"""Local web server for the touch display and projector (brain/README.md, "Touch display").

GET  /            the display page
GET  /events      server-sent events from the EventBus
GET  /screen.jpg  latest screen image (active rect, so tap coordinates map 1:1)
GET  /frame/N.jpg  the screen a step was decided on (N from its plan event; the last KEPT_FRAMES steps)
POST /goal        {"text": "...", "continues": bool} a typed or transcribed request; continues: it
                                                  extends the last request (more of the same utterance)
                  "test": true marks a test run, recorded apart from the user's (brain/README.md, "Recordings")
POST /tap         {"x": 0-1, "y": 0-1, "mode": "click"|"point"}
POST /stop        cancel whatever is running (same path as barge-in) and drop queued requests

The server never touches HID: goals and taps are queued for the main loop,
which stays the single HID writer.

Any web page in a browser on this machine can reach a localhost port, and this
server can drive the target computer. So every request must name this server
in its Host header (blocks DNS rebinding), browser POSTs must come from this
page's own origin, and POST bodies must be application/json (cross-site pages
can't send that without a CORS preflight, which this server never approves).
Local programs without an Origin header (curl, the voice server) are allowed.
"""

from __future__ import annotations

import json
import queue
import re
import threading
from contextlib import suppress
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
    continues: bool = False  # more of the same spoken request
    test: bool = False  # a test run, not the user's
    x: float = 0.0
    y: float = 0.0


class DisplayServer:
    def __init__(self, bus: EventBus, commands: queue.Queue[Command],
                 stop: Callable[[str], None], host: str = "127.0.0.1", port: int = 8770) -> None:
        if host in ("", "0.0.0.0", "::"):
            raise ValueError("bind the display to a specific address, not all interfaces")
        page = resources.files("brain.display").joinpath("page.html").read_bytes()
        handler = type("Handler", (_Handler,), {"bus": bus, "commands": commands, "stop": staticmethod(stop),
                                                "page": page, "hosts": frozenset()})
        self._httpd = ThreadingHTTPServer((host, port), handler)
        bound = self._httpd.server_address[1]
        names = {host, "localhost", "127.0.0.1"} if host in ("127.0.0.1", "localhost") else {host}
        handler.hosts = frozenset(f"{name}:{bound}" for name in names)
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
    hosts: frozenset[str]  # "host:port" values this server answers to

    def log_message(self, format: str, *args: object) -> None:  # keep the REPL readable
        pass

    def do_GET(self) -> None:
        if not self._trusted_host():
            return
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
        elif (kept := re.fullmatch(r"/frame/([0-9]{1,9})\.jpg", path)) and (jpeg := self.bus.kept_frame(int(kept[1]))):
            self._reply(200, "image/jpeg", jpeg, cache=False)  # ids restart with the brain
        else:
            self._reply(404, "text/plain", b"not found")

    def do_POST(self) -> None:
        # Read the body before any refusal: closing a socket with unread data makes some systems
        # (Windows) send a reset, so the client sees "connection reset" instead of our reply.
        raw = self._read_body()
        if not self._trusted_host():
            return
        origin = self.headers.get("Origin")
        if origin is not None and origin.removeprefix("http://") not in self.hosts:
            self._reply(403, "application/json", b'{"error":"cross-origin request refused"}')
            return
        try:
            body = self._json_body(raw)
            if self.path == "/goal":
                text = str(body.get("text", "")).strip()
                if not text:
                    raise ValueError("text is required")
                self.commands.put(Command("goal", text=text, continues=body.get("continues") is True,
                                          test=body.get("test") is True))
            elif self.path == "/tap":
                mode = body.get("mode", "click")
                if mode not in ("click", "point"):
                    raise ValueError("mode must be click or point")
                self.commands.put(Command(mode, x=_unit(body.get("x")), y=_unit(body.get("y"))))
            elif self.path == "/stop":
                # The newest request wins: what was queued before the stop is stale. The voice layer stops
                # before the new speech is sent, so this never drops the request that follows.
                with suppress(queue.Empty):
                    while True:
                        self.commands.get_nowait()
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

    def _trusted_host(self) -> bool:
        if self.headers.get("Host") in self.hosts:
            return True
        self._reply(403, "text/plain", b"unknown host")
        return False

    def _read_body(self) -> bytes | None:
        """The request body, or None when it's missing or too large (the connection then closes)."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if not 0 < length <= MAX_BODY:
            self.close_connection = True
            return None
        return self.rfile.read(length)

    def _json_body(self, raw: bytes | None) -> dict[str, object]:
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            raise ValueError("Content-Type must be application/json")
        if raw is None:
            raise ValueError("JSON body required")
        try:
            body = json.loads(raw)
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
