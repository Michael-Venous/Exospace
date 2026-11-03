"""The Qwen queue agent behind Jev's agent API (POST /goal, /stop; GET /events, /screen.jpg) on :8770,
so the unchanged voice page (brain.voice.layer) and its orb drive it: the page hears, this acts and answers."""

import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from bench.arch.optimized_queue import create
from bench.arch.pi_target import PiTarget
from bench.arch.talk import LOGS, MODEL
from brain.config import Config

clients: list[queue.Queue] = []


def emit(ev: str, **fields) -> None:
    for client in list(clients):
        client.put({"ev": ev, "t": time.time(), **fields})


def main() -> None:
    session = LOGS / time.strftime("%Y%m%d-%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    target = PiTarget(Config.load())
    agent = create(target, MODEL, "", os.environ["TYPESAFE_API_KEY"], session, "qwen")
    brain, lock = agent.brain, threading.Lock()
    # Words said after a request's first clause wait until it has acted once (or ended): applying them at once
    # throws away the plan in flight, so "open chrome and go to youtube" opened Chrome only after the whole sentence.
    pending = [None]
    acted = lambda: agent.shared.finished or len(agent.shared.log) > agent.shared.turn_start

    def pump() -> None:
        while True:
            with lock:
                if pending[0] and acted():
                    print(f"you (more): {pending[0]}", flush=True)
                    brain.hear(pending[0], speaking=False)
                    pending[0] = None
            time.sleep(0.05)

    threading.Thread(target=pump, daemon=True).start()

    def answers() -> None:  # what the agent says, as a result line the page speaks word for word
        spoken = 0
        while True:
            said = brain.said
            while spoken < len(said):
                stale = brain.said_requests[spoken] != agent.shared.request or brain.speaking
                if not stale:
                    print(f"  agent: {said[spoken]}", flush=True)
                    emit("result", status="ask_user", message=said[spoken])
                spoken += 1
            time.sleep(0.2)

    threading.Thread(target=answers, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            with lock:
                if self.path == "/goal":
                    text = str(body.get("text", "")).strip()
                    if body.get("continues") and brain.words:  # more words of the same request
                        pending[0] = f"{pending[0] or brain.words} {text}"
                        if not acted():
                            return self._reply(200, b'{"ok":true}')
                        text, pending[0] = pending[0], None
                    else:
                        pending[0] = None
                        brain.end_turn()
                        emit("run", goal=text)
                    print(f"you: {text}", flush=True)
                    brain.hear(text, speaking=False)
                elif self.path == "/stop":
                    pending[0] = None
                    brain.cancel_turn()
                else:
                    return self._reply(404, b"{}")
            self._reply(200, b'{"ok":true}')

        def do_GET(self) -> None:
            if self.path.split("?")[0] == "/screen.jpg":
                try:
                    ok, jpeg = cv2.imencode(".jpg", target.grab())
                    return self._reply(200, jpeg.tobytes(), "image/jpeg")
                except Exception:
                    return self._reply(204, b"", "image/jpeg")
            if self.path != "/events":
                return self._reply(404, b"{}")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            mine: queue.Queue = queue.Queue()
            clients.append(mine)
            try:
                while True:
                    try:
                        self.wfile.write(b"data: " + json.dumps(mine.get(timeout=15)).encode() + b"\n\n")
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                clients.remove(mine)

        def _reply(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 8770), Handler)
    server.daemon_threads = True
    print("Qwen agent on http://127.0.0.1:8770 (start brain.voice.layer for the page)", flush=True)
    try:
        server.serve_forever()
    finally:
        agent.close()
        agent.controller.hid.close()


if __name__ == "__main__":
    main()
