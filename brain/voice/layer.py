"""Voice layer (PLAN.md §11): ears and mouth on top of the agent, never its hands.

Run the agent with a display server, then this:
    .venv/bin/python -m brain --display 8770
    .venv/bin/python -m brain.voice.layer --agent http://127.0.0.1:8770

Hold the button (or Space) and speak. Each pause sends what was said to the agent's
POST /goal, so work starts while the user keeps talking; starting to speak while the
agent is busy sends POST /stop first. Later segments of the same hold continue that
request ("go to youtube" ... "and search lofi", "actually go back"): they join it at its
next step without cancelling the step in progress. Narration comes from the
agent's event stream: questions (including irreversible-action confirmations) are
always spoken, plan stages briefly, then the result; a request that runs silently gets
"On it." and then a progress line every few seconds, so a hard task never leaves dead air.
With the page's Describe toggle on, Gemini Live describes the screen after each request
instead of the bare result line (brain/voice/describe.py). The page can also show the
target's live screen, relayed from the agent's GET /screen.jpg. This process never
writes HID.

Mic audio is relayed to Deepgram server-side, so DEEPGRAM_API_KEY never reaches
the browser. The page and relay only answer to this machine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import threading
import time
import urllib.request
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any

from dotenv import load_dotenv

from brain.voice import button
from brain.voice.describe import DESCRIBE_MODEL, Describer

# A 200 ms pause ends a segment: the short break after "open chrome" in "open chrome and search cats" is
# enough, so Chrome opens while the user is still talking.
DG_URL = "wss://api.deepgram.com/v1/listen?model=nova-3&interim_results=true&endpointing=200&smart_format=true"
KEEPALIVE_S = 15.0
MAX_BODY = 4096
ACK_S = 2.5  # a request still running this long with nothing said gets "On it."
PROGRESS_S = 10.0  # then each this long of silence gets a progress line: a hard task never leaves dead air


def narrate(event: dict[str, Any]) -> str | None:
    """What to say for one agent event; None stays silent (fast actions need no commentary)."""
    kind = event.get("ev")
    if kind == "status":
        return str(event.get("message", "")).strip() or None
    if kind == "plan":
        reasoning = str(event.get("reasoning", ""))
        if reasoning.startswith("Stage "):  # "Stage 1/3: <expected visible result>"
            return reasoning.split(":", 1)[-1].strip() or None
        return None
    if kind != "result":
        return None
    status, message = event.get("status"), str(event.get("message", "")).strip()
    if status == "ask_user":
        return message or "Can you clarify?"
    if status == "done":
        message = message.removeprefix("Verified: ").strip()
        return message if 0 < len(message) <= 120 else "Done."
    if status == "cancelled":
        return "Stopped."
    return "Sorry, I couldn't do that."


def filler(quiet_s: float, spoken: bool) -> str | None:
    """What to say while a request runs silently: an acknowledgement, then progress."""
    if not spoken:
        return "On it." if quiet_s >= ACK_S else None
    return "Still working on it." if quiet_s >= PROGRESS_S else None


class Agent:
    """The agent's display server, reached as a local program (no Origin header)."""

    def __init__(self, base_url: str, describer: Describer | None = None) -> None:
        self.base = base_url.rstrip("/")
        self.describer = describer
        self.describe_on = False  # the page's Describe toggle
        self.busy = False
        self._goal = ""
        self._said_at = 0.0  # monotonic time of the last line for the running request
        self._spoken = False  # anything said for it yet
        self._last = ""  # its last line: a result that repeats the final stage is said once

    def post(self, path: str, body: dict[str, Any]) -> None:
        request = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()

    def submit(self, text: str, continues: bool = False) -> None:
        self.post("/goal", {"text": text, "continues": continues})

    def screen(self) -> bytes | None:
        """The latest capture-card frame as JPEG, or None when the agent has none or is down."""
        try:
            with urllib.request.urlopen(self.base + "/screen.jpg", timeout=2) as response:
                return response.read() or None  # 204: no frame yet
        except (OSError, HTTPException):  # an agent that dies mid-frame raises IncompleteRead
            return None

    def interrupt(self) -> None:
        if self.busy:
            self.post("/stop", {})

    def follow(self, lines: queue.Queue[dict[str, Any]]) -> None:
        """Track busy state and queue narration from the agent's server-sent events, forever."""
        while True:
            try:
                with urllib.request.urlopen(self.base + "/events", timeout=KEEPALIVE_S * 2) as stream:
                    # The agent replays recent events: they set busy, but saying them again would be stale.
                    # shortcut: compares the agent's clock with ours; send a sequence number if the agent moves to another machine.
                    since = time.time()
                    for raw in stream:
                        if not raw.startswith(b"data: "):
                            continue
                        event = json.loads(raw[6:])
                        fresh = event.get("t", since) >= since
                        if event.get("ev") == "run":
                            self.busy, self._goal = True, str(event.get("goal", ""))
                            self._said_at, self._spoken, self._last = time.monotonic(), False, ""
                        elif event.get("ev") == "result":
                            self.busy = False
                        line = narrate(event)
                        if not (line and fresh):
                            continue
                        if self.describing(event):
                            threading.Thread(target=self._describe, args=(lines, line), daemon=True).start()
                        else:
                            self._say(lines, line)
            except OSError:
                threading.Event().wait(1.0)  # agent restarting; reconnect


    def describing(self, event: dict[str, Any]) -> bool:
        """A finished request is described; questions are always read word for word."""
        return self.describe_on and self.describer is not None and event.get("ev") == "result" \
            and event.get("status") in ("done", "fail")

    def _describe(self, lines: queue.Queue[dict[str, Any]], line: str) -> None:
        lines.put({"show": line})
        jpeg = self.screen()
        if not (jpeg and self.describer and self.describer.describe(jpeg, self._goal, line, lines.put)):
            lines.put({"say": line})

    def pace(self, lines: queue.Queue[dict[str, Any]]) -> None:
        """Fill silence while a request runs, forever."""
        while True:
            time.sleep(0.5)
            line = filler(time.monotonic() - self._said_at, self._spoken) if self.busy else None
            if line:
                self._say(lines, line)

    def _say(self, lines: queue.Queue[dict[str, Any]], line: str) -> None:
        self._said_at, self._spoken = time.monotonic(), True
        if line != self._last:
            self._last = line
            lines.put({"say": line})


def page_server(agent: Agent, port: int, ws_port: int, lines: queue.Queue[dict[str, Any]]) -> ThreadingHTTPServer:
    """The voice page, narration stream and agent forwarding (no audio). Port 0 picks a free port."""
    page = resources.files("brain.voice").joinpath("layer.html").read_text().replace("__WS_PORT__", str(ws_port))
    hosts: set[str] = set()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            if self.headers.get("Host") not in hosts:
                return self._reply(403, b"unknown host")
            if self.path == "/":
                self._reply(200, page.encode(), "text/html; charset=utf-8")
            elif self.path == "/narration":
                self._narration()
            elif self.path.split("?", 1)[0] == "/screen.jpg":
                jpeg = agent.screen()
                if jpeg:
                    self._reply(200, jpeg, "image/jpeg")
                else:
                    self._reply(503, b"no screen yet", "text/plain")
            else:
                self._reply(404, b"not found")

        def do_POST(self) -> None:
            # Host + same-origin + JSON body: other sites can't drive the agent through this page.
            origin = self.headers.get("Origin", "").removeprefix("http://")
            if self.headers.get("Host") not in hosts or origin not in hosts or \
                    self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self._reply(403, b'{"error":"forbidden"}')
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length)) if 0 < length <= MAX_BODY else {}
            try:
                if self.path == "/talk":  # the user started speaking: barge in on running work
                    if agent.describer is not None:
                        agent.describer.cancel()
                    agent.interrupt()
                elif self.path == "/describe":
                    agent.describe_on = body.get("on") is True and agent.describer is not None
                    return self._reply(200, json.dumps({"ok": True, "on": agent.describe_on}).encode())
                elif self.path == "/segment":
                    text = str(body.get("text", "")).strip()
                    if not 0 < len(text) <= 500:
                        return self._reply(400, b'{"error":"text must be 1-500 characters"}')
                    continues = body.get("continues") is True
                    if not continues:  # a new request barges in; more words of this one join it at the next step
                        agent.interrupt()
                    agent.submit(text, continues)
                else:
                    return self._reply(404, b'{"error":"not found"}')
            except OSError as exc:
                return self._reply(502, json.dumps({"error": f"agent unreachable: {exc}"}).encode())
            self._reply(200, b'{"ok":true}')

        def _narration(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    try:
                        message = lines.get(timeout=KEEPALIVE_S)
                        self.wfile.write(b"data: " + json.dumps(message).encode() + b"\n\n")
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _reply(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    bound = server.server_address[1]
    hosts.update({f"127.0.0.1:{bound}", f"localhost:{bound}"})
    threading.Thread(target=server.serve_forever, name="voice-page", daemon=True).start()
    return server


def serve(agent: Agent, port: int, ws_port: int, deepgram_key: str) -> None:
    lines: queue.Queue[dict[str, Any]] = queue.Queue()
    threading.Thread(target=agent.follow, args=(lines,), name="agent-events", daemon=True).start()
    threading.Thread(target=agent.pace, args=(lines,), name="agent-pace", daemon=True).start()
    if button.available():  # the Pi's push-to-talk button: the page holds and lets go of its orb
        threading.Thread(target=button.watch, args=(lambda down: lines.put({"button": "down" if down else "up"}),),
                         name="ptt-button", daemon=True).start()
    page_server(agent, port, ws_port, lines)
    print(f"Voice layer on http://127.0.0.1:{port} -> agent {agent.base}", flush=True)
    asyncio.run(_relay_forever(port, ws_port, deepgram_key))


async def _relay_forever(port: int, ws_port: int, key: str) -> None:
    """Pipe mic audio from the page to Deepgram and transcripts back."""
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve as ws_serve
    from websockets.exceptions import ConnectionClosed

    async def relay(browser) -> None:
        async with connect(DG_URL, additional_headers={"Authorization": f"Token {key}"}) as dg:
            async def upload() -> None:
                try:
                    async for chunk in browser:
                        await dg.send(chunk)
                finally:
                    await dg.close()

            uploader = asyncio.create_task(upload())
            try:
                async for message in dg:
                    await browser.send(message)
            except ConnectionClosed:
                pass
            finally:
                uploader.cancel()

    # Origin check: WebSockets skip CORS, so only this page may stream through the relay.
    origins = [f"http://127.0.0.1:{port}", f"http://localhost:{port}"]
    async with ws_serve(relay, "127.0.0.1", ws_port, origins=origins):
        await asyncio.Future()


def main() -> int:
    parser = argparse.ArgumentParser(description="Exospace voice layer")
    parser.add_argument("--agent", default="http://127.0.0.1:8770", help="the agent's --display server")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--ws-port", type=int, default=8781)
    args = parser.parse_args()
    load_dotenv()
    key = os.environ.get("DEEPGRAM_API_KEY", "")
    if not key:
        print("Set DEEPGRAM_API_KEY in the project .env (see .env.example).", flush=True)
        return 1
    gemini = os.environ.get("GEMINI_API_KEY", "")
    describer = Describer(gemini, os.environ.get("DESCRIBE_MODEL") or DESCRIBE_MODEL) if gemini else None
    serve(Agent(args.agent, describer), args.port, args.ws_port, key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
