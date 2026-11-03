"""The Qwen queue agent behind Jev's agent API (POST /goal, /stop; GET /events, /screen.jpg) on :8770,
so the unchanged voice page (brain.voice.layer) and its orb drive it: the page hears, this acts and answers.

On the Pi (the default) the target is the capture card and the Pico; `EXO_TARGET=laptop` drives the laptop rig
instead (`serve_laptop.sh` starts this and the page there). The page sends each request whole, once the talk
button is released.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

REPORT_S = 0.2  # how often the reporter looks for what the agent said and which requests ended
QUIET = {"verified": "done", "replied": "done", "cancelled": "cancelled"}  # a request that ended without speaking


class Events:
    """The page's event stream: each connected page gets every event."""

    def __init__(self) -> None:
        self.clients: list[queue.Queue] = []

    def emit(self, ev: str, **fields) -> None:
        for client in list(self.clients):
            client.put({"ev": ev, "t": time.time(), **fields})


class Reporter:
    """Per request: what the agent says goes to the page as a result it speaks word for word; a request that
    ends without speaking still ends the page's wait ("On it." and progress lines otherwise run on); and, as
    talk.py does, its timings are appended to `timings` and the control state is saved in the session."""

    def __init__(self, agent, events: Events, session: Path, timings: Path) -> None:
        self.agent, self.events, self.session, self.timings = agent, events, session, timings
        self.spoken = 0  # the agent's lines handled so far
        self.spoke: set[int] = set()  # requests that said something
        self.reported: set[int] = set()  # requests whose end was handled

    def step(self) -> None:
        agent = self.agent
        with agent.cond:
            said, owners = list(agent.brain.said), list(agent.brain.said_requests)
            current, speaking = agent.shared.request, agent.brain.speaking
            states = {request: dict(state) for request, state in agent.control.requests.items()}
        for line, request in list(zip(said, owners))[self.spoken:]:
            if request == current and not speaking:  # a line for a request already replaced stays unsaid
                print(f"  agent: {line}", flush=True)
                self.events.emit("result", status="ask_user", message=line)
                self.spoke.add(request)
        self.spoken = len(said)
        for request, state in states.items():
            if request in self.reported or "terminal_at" not in state:
                continue
            self.reported.add(request)
            if state.get("status") != "interrupted" and request not in self.spoke:
                self.events.emit("result", status=QUIET.get(state.get("status"), "fail"), message="")
            self._record(request, state)

    def _record(self, request: int, state: dict) -> None:
        released = state.get("released_at")
        since = lambda key: round(state[key] - released, 3) if released is not None and key in state else None
        row = {"session": self.session.name, "via": "page", "request": request, "status": state.get("status"),
               "first_input_s": since("first_input_at"), "terminal_s": since("terminal_at"),
               "stt_finalize_s": since("transcript_final_at")}
        print("  timing: " + json.dumps(row), flush=True)
        with self.timings.open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.agent.control.save(self.session / "control.json")

    def run(self) -> None:
        while True:
            self.step()
            time.sleep(REPORT_S)


def server(agent, target, events: Events, port: int = 8770) -> ThreadingHTTPServer:
    """The agent API on 127.0.0.1:`port` (0: any free port)."""
    brain, lock = agent.brain, threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            with lock:
                if self.path == "/goal":
                    text = str(body.get("text", "")).strip()
                    if not text:
                        return self._reply(400, b'{"error":"no text"}')
                    brain.end_turn()  # each request arrives whole: it replaces whatever ran before
                    events.emit("run", goal=text)
                    print(f"you: {text}", flush=True)
                    brain.hear(text, speaking=False)
                elif self.path == "/stop":
                    brain.cancel_turn()
                else:
                    return self._reply(404, b"{}")
            self._reply(200, b'{"ok":true}')

        def do_GET(self) -> None:
            if self.path.split("?")[0] == "/screen.jpg":
                try:
                    ok, jpeg = cv2.imencode(".jpg", target.grab())
                    return self._reply(200, jpeg.tobytes(), "image/jpeg")
                except Exception:  # noqa: BLE001 - no frame right now: the page keeps its last one
                    return self._reply(204, b"", "image/jpeg")
            if self.path != "/events":
                return self._reply(404, b"{}")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            mine: queue.Queue = queue.Queue()
            events.clients.append(mine)
            try:
                while True:
                    try:
                        self.wfile.write(b"data: " + json.dumps(mine.get(timeout=15)).encode() + b"\n\n")
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                events.clients.remove(mine)

        def _reply(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.daemon_threads = True
    return httpd


def make_target(kind: str):
    """"pi": the capture card and the Pico (the rig's own path); "laptop": the laptop rig, brain and target at once."""
    if kind == "laptop":
        from bench.arch.laptop import LaptopTarget
        return LaptopTarget()
    if kind != "pi":
        raise SystemExit(f"EXO_TARGET must be pi or laptop, not {kind!r}")
    from bench.arch.pi_target import PiTarget
    from brain.config import Config
    return PiTarget(Config.load())


def main() -> None:
    from bench.arch.optimized_queue import create
    from bench.arch.talk import LOGS, MODEL  # talk.py's model and logs folder, so both front ends log alike

    kind = os.environ.get("EXO_TARGET", "pi")
    session = LOGS / time.strftime("%Y%m%d-%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    target = make_target(kind)
    agent = create(target, MODEL, "", os.environ["TYPESAFE_API_KEY"], session, "qwen")
    events = Events()
    threading.Thread(target=Reporter(agent, events, session, LOGS / "timings-optimized.jsonl").run, daemon=True).start()
    httpd = server(agent, target, events)
    print(f"Qwen agent ({kind}) on http://127.0.0.1:8770 (start brain.voice.layer for the page)", flush=True)
    try:
        httpd.serve_forever()
    finally:
        agent.close()
        agent.controller.hid.close()


if __name__ == "__main__":
    main()
