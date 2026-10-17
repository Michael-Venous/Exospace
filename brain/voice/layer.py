"""Voice layer (PLAN.md §11): ears and mouth on top of the agent, never its hands.

Run the agent with a display server, then this:
    .venv/bin/python -m brain --display 8770
    .venv/bin/python -m brain.voice.layer --agent http://127.0.0.1:8770

Hold the button (or Space) and speak. Each finished utterance goes to the agent's
POST /goal; starting to speak while the agent is busy sends POST /stop first, so the
next utterance becomes a correction ("actually go back"). Narration comes from the
agent's event stream: questions (including irreversible-action confirmations) are
always spoken, plan stages briefly, then the result. The page can also show the
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any

from dotenv import load_dotenv

DG_URL = "wss://api.deepgram.com/v1/listen?model=nova-3&interim_results=true&endpointing=300&smart_format=true"
KEEPALIVE_S = 15.0
MAX_BODY = 4096


def narrate(event: dict[str, Any]) -> str | None:
    """What to say for one agent event; None stays silent (fast actions need no commentary)."""
    kind = event.get("ev")
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


class Agent:
    """The agent's display server, reached as a local program (no Origin header)."""

    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")
        self.busy = False

    def post(self, path: str, body: dict[str, Any]) -> None:
        request = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()

    def submit(self, text: str) -> None:
        self.post("/goal", {"text": text})

    def screen(self) -> bytes | None:
        """The latest capture-card frame as JPEG, or None when the agent has none or is down."""
        try:
            with urllib.request.urlopen(self.base + "/screen.jpg", timeout=2) as response:
                return response.read() or None  # 204: no frame yet
        except OSError:
            return None

    def interrupt(self) -> None:
        if self.busy:
            self.post("/stop", {})

    def follow(self, lines: queue.Queue[str]) -> None:
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
                        if event.get("ev") == "run":
                            self.busy = True
                        elif event.get("ev") == "result":
                            self.busy = False
                        line = narrate(event)
                        if line and event.get("t", since) >= since:
                            lines.put(line)
            except OSError:
                threading.Event().wait(1.0)  # agent restarting; reconnect


def page_server(agent: Agent, port: int, ws_port: int, lines: queue.Queue[str]) -> ThreadingHTTPServer:
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
                    agent.interrupt()
                elif self.path == "/segment":
                    text = str(body.get("text", "")).strip()
                    if not 0 < len(text) <= 500:
                        return self._reply(400, b'{"error":"text must be 1-500 characters"}')
                    agent.interrupt()  # speaking while it works ("stop", a correction) barges in
                    agent.submit(text)
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
                        line = lines.get(timeout=KEEPALIVE_S)
                        self.wfile.write(b"data: " + json.dumps({"say": line}).encode() + b"\n\n")
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
    lines: queue.Queue[str] = queue.Queue()
    threading.Thread(target=agent.follow, args=(lines,), name="agent-events", daemon=True).start()
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
    serve(Agent(args.agent), args.port, args.ws_port, key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
