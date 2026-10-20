"""Recordings of every request, played back on a phone (brain/README.md, "Recordings").

The recorder follows the event bus and saves each request from `run` to `result`: its events and
every screen frame, so the run plays back as a video with Jev's boxes and the plan on top. The
server only lists and serves what was saved; it never touches the agent or HID, so it may listen
on a network address (the Pi's Tailscale address) where a phone can reach it.
"""

from __future__ import annotations

import json
import re
import shutil
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import TextIO

from brain.events import Event, EventBus

RECORDINGS = Path("work/recordings")
KEEP_RUNS = 50  # older recordings are deleted when a new request starts
MAX_FRAMES = 2400  # live frames per run: 10 min of the Pi's 4 fps view; step screens are always saved
NAME = re.compile(r"[0-9]{8}-[0-9]{6}-[0-9a-z]{1,12}")  # <date>-<time>-<run id>, the recording's folder
SHUTDOWN_POLL_S = 0.05


class Recorder:
    """Saves each run to `root/<name>/`: `log.jsonl` (the bus events, plus a line per saved frame)
    and `<frame id>.jpg`. A disk error drops the current recording, never the run."""

    def __init__(self, bus: EventBus, root: Path) -> None:
        self.bus, self.root = bus, root
        self._client = bus.subscribe()
        self._log: TextIO | None = None
        self._dir = root
        self._frames: set[int] = set()
        self._live = 0

    def start(self) -> None:
        threading.Thread(target=self._follow, name="recorder", daemon=True).start()

    def _follow(self) -> None:
        while True:
            self.handle(self._client.get())

    def handle(self, event: Event) -> None:
        try:
            self._record(event)
        except OSError as exc:
            print(f"Recording stopped: {exc}", file=sys.stderr, flush=True)
            self._close()

    def _record(self, event: Event) -> None:
        if event["ev"] == "run":
            self._close()  # a run cut short by a crash has no result
            self._start(event)
        if self._log is None:
            return
        if event["ev"] == "frame":
            if self._live < MAX_FRAMES:
                frame_id, jpeg = self.bus.frame()  # may be newer than the notice: the newest is what's on screen
                self._live += self._save(frame_id, jpeg, time.time())
            return
        if event["ev"] in ("plan", "discard") and event.get("frame"):
            # The screen it was decided (or discarded) on, in case this recorder fell behind its notice
            self._save(event["frame"], self.bus.kept_frame(event["frame"]), event["t"] - event.get("t_model", 0))
        self._write(event)
        if event["ev"] == "result":
            self._close()

    def _start(self, event: Event) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for old in recordings(self.root)[KEEP_RUNS - 1:]:
            shutil.rmtree(self.root / old, ignore_errors=True)
        run_id = re.sub(r"[^0-9a-z]", "", str(event.get("run_id", "")).lower())[:12] or "run"
        self._dir = self.root / f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(event['t']))}-{run_id}"
        self._dir.mkdir()
        self._log = (self._dir / "log.jsonl").open("a", buffering=1)  # line-buffered: a live run is viewable
        self._frames, self._live = set(), 0

    def _save(self, frame_id: int, jpeg: bytes, t: float) -> bool:
        if not jpeg or frame_id in self._frames:
            return False
        (self._dir / f"{frame_id}.jpg").write_bytes(jpeg)
        self._frames.add(frame_id)
        self._write({"ev": "frame", "id": frame_id, "t": t})
        return True

    def _write(self, event: Event) -> None:
        assert self._log is not None
        self._log.write(json.dumps(event) + "\n")

    def _close(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None


def recordings(root: Path) -> list[str]:
    """Recording names, newest first."""
    if not root.is_dir():
        return []
    return sorted((p.name for p in root.iterdir() if NAME.fullmatch(p.name) and p.is_dir()), reverse=True)


def read_log(path: Path) -> list[Event]:
    """A recording's events; a line still being written is skipped."""
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return events


def summary(root: Path, name: str) -> dict[str, object]:
    events = read_log(root / name / "log.jsonl")
    run = next((e for e in events if e["ev"] == "run"), {})
    result = next((e for e in reversed(events) if e["ev"] == "result"), None)
    return {"name": name, "goal": run.get("goal", ""), "t": run.get("t"),
            "status": result["status"] if result else "running",
            "message": result["message"] if result else "",
            "steps": result["steps"] if result else sum(e["ev"] == "plan" for e in events),
            "total_s": result["total_s"] if result else None}


class RecordingsServer:
    """GET /                    the player page
    GET /runs.json            every recording, newest first
    GET /runs/NAME/log.json   one recording's events
    GET /runs/NAME/N.jpg      one of its frames"""

    def __init__(self, root: Path, host: str, port: int) -> None:
        if host in ("", "0.0.0.0", "::"):
            raise ValueError("bind the recordings to a specific address (e.g. the Tailscale one), not all interfaces")
        page = resources.files("brain.display").joinpath("recordings.html").read_bytes()
        handler = type("Handler", (_Handler,), {"root": root, "page": page, "hosts": frozenset()})
        self._httpd = ThreadingHTTPServer((host, port), handler)
        bound = self._httpd.server_address[1]
        names = {host, socket.gethostname()}  # the hostname: Tailscale's MagicDNS name for the Pi
        if host in ("127.0.0.1", "localhost"):
            names |= {"127.0.0.1", "localhost"}
        handler.hosts = frozenset(f"{name}:{bound}" for name in names)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": SHUTDOWN_POLL_S},
                                        name="recordings", daemon=True)

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
    root: Path
    page: bytes
    hosts: frozenset[str]  # "host:port" values this server answers to (blocks DNS rebinding)

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        if self.headers.get("Host") not in self.hosts:
            self._reply(403, "text/plain", b"unknown host")
            return
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._reply(200, "text/html; charset=utf-8", self.page)
        elif path == "/runs.json":
            runs = [summary(self.root, name) for name in recordings(self.root)]
            self._reply(200, "application/json", json.dumps(runs).encode())
        elif (m := re.fullmatch(r"/runs/([^/]+)/(log\.json|[0-9]{1,9}\.jpg)", path)) and NAME.fullmatch(m[1]):
            folder = self.root / m[1]
            try:
                if m[2] == "log.json":
                    self._reply(200, "application/json", json.dumps(read_log(folder / "log.jsonl")).encode())
                else:  # a frame never changes once saved
                    self._reply(200, "image/jpeg", (folder / m[2]).read_bytes(), cache="max-age=86400, immutable")
            except FileNotFoundError:
                self._reply(404, "text/plain", b"not found")
        else:
            self._reply(404, "text/plain", b"not found")

    def _reply(self, code: int, ctype: str, body: bytes, cache: str = "no-store") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)
