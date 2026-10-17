"""Local web server for the touch display and projector (brain/agent/DESIGN.md §5, §5b).

GET  /            the display page
GET  /events      server-sent events from the EventBus
GET  /screen.jpg  latest screen image (active rect, so tap coordinates map 1:1)
POST /goal        {"text": "..."}                 a typed or transcribed request
POST /tap         {"x": 0-1, "y": 0-1, "mode": "click"|"point"}
POST /stop        cancel whatever is running (same path as barge-in)

The server never touches HID: goals and taps are queued for the main loop,
which stays the single HID writer.
"""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Callable, Literal

from brain.events import EventBus

KEEPALIVE_S = 15.0
MAX_BODY = 16_384
SHUTDOWN_POLL_S = 0.05


@dataclass(frozen=True)
class Command:
    kind: Literal["goal", "click", "point"]
    text: str = ""
    x: float = 0.0
    y: float = 0.0


class DisplayServer:
    def __init__(self, bus: EventBus, commands: queue.Queue[Command],
                 stop: Callable[[str], None], host: str = "127.0.0.1", port: int = 8770) -> None:
        page = resources.files("brain.display").joinpath("page.html").read_bytes()
        handler = type("Handler", (_Handler,), {"bus": bus, "commands": commands, "stop": staticmethod(stop),
                                                "page": page})
        self._httpd = ThreadingHTTPServer((host, port), handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": SHUTDOWN_POLL_S},
                                        name="display", daemon=True)

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


class _Handler(BaseHTTPRequestHandler):
    bus: EventBus
    commands: queue.Queue[Command]
    stop: Callable[[str], None]
    page: bytes

    def log_message(self, format: str, *args: object) -> None:  # keep the REPL readable
        pass

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._reply(200, "text/html; charset=utf-8", self.page)
        elif path == "/screen.jpg":
            _, jpeg = self.bus.frame()
            if jpeg:
                self._reply(200, "image/jpeg", jpeg, cache=False)
            else:
                self._reply(204, "text/plain", b"")
        elif path == "/events":
            self._stream_events()
        else:
            self._reply(404, "text/plain", b"not found")

    def do_POST(self) -> None:
        try:
            body = self._json_body()
            if self.path == "/goal":
                text = str(body.get("text", "")).strip()
                if not text:
                    raise ValueError("text is required")
                self.commands.put(Command("goal", text=text))
            elif self.path == "/tap":
                mode = body.get("mode", "click")
                if mode not in ("click", "point"):
                    raise ValueError("mode must be click or point")
                self.commands.put(Command(mode, x=_unit(body.get("x")), y=_unit(body.get("y"))))
            elif self.path == "/stop":
                self.stop("barge-in")
                self.bus.publish("stopped")
            else:
                self._reply(404, "application/json", b'{"error":"not found"}')
                return
        except ValueError as exc:
            self._reply(400, "application/json", json.dumps({"error": str(exc)}).encode())
            return
        self._reply(200, "application/json", b'{"ok":true}')

    def _stream_events(self) -> None:
        client = self.bus.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                try:
                    event = client.get(timeout=KEEPALIVE_S)
                    self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.bus.unsubscribe(client)

    def _json_body(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY:
            raise ValueError("JSON body required")
        try:
            body = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSON") from exc
        if not isinstance(body, dict):
            raise ValueError("JSON object required")
        return body

    def _reply(self, code: int, ctype: str, body: bytes, cache: bool = True) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _unit(value: object) -> float:
    if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError("x and y must be numbers in [0, 1]")
    return float(value)
