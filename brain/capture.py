"""Seeing the target: frame sources and frame analysis.

`V4L2Capture` reads the HDMI capture card on the Pi. `ScreenCapture` and
`CommandCapture` are development stand-ins that capture the machine you're on;
`CommandCapture` runs a screenshot tool, for desktops (Wayland) that mss cannot read.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

from brain.hid import window_rects
from brain.types import Frame, Rect

STALE_S = 0.5  # a frame older than this counts as no signal
BLANK_MAX = 12  # brightest pixel below this (0-255) = blank / no signal
BAR_MIN_FRACTION = 0.05  # pillarbox bars must each be at least this wide
ASPECT_RANGE = (0.3, 3.4)  # width/height of any real screen; a tall phone is ~0.43
THUMB_SIZE = (64, 36)


class CaptureError(RuntimeError):
    """The capture card is missing or unusable."""


class Capture(Protocol):
    def grab(self) -> Frame | None:
        """Latest frame, or None if there's no fresh signal."""
        ...

    def signal_present(self) -> bool: ...

    def close(self) -> None: ...


class V4L2Capture:
    """HDMI capture card via V4L2, read continuously on a background thread."""

    def __init__(self, device: str, width: int, height: int) -> None:
        self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise CaptureError(f"cannot open capture device {device}")
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self._frame: Frame | None = None
        self._stamp = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, name="capture", daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while self._running:
            try:
                ok, frame = self._cap.read()
            except cv2.error:
                # Some cards (the Monster/UltraSemi one: 3 of 40 frames right after opening) send an
                # empty MJPG frame, which OpenCV raises on instead of returning False. Uncaught, it
                # would end this thread and leave the agent with no video for the rest of the run.
                ok, frame = False, None
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._stamp = time.monotonic()

    def grab(self) -> Frame | None:
        with self._lock:
            if self._frame is None or time.monotonic() - self._stamp > STALE_S:
                return None
            return self._frame

    def signal_present(self) -> bool:
        frame = self.grab()
        return frame is not None and not is_blank(frame)

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()


class ScreenCapture:
    """Development stand-in: captures this machine's desktop, or one of its monitors (mss numbering: 0 is
    all of them, 1 the first). Windows whose title starts with what `hide` returns are greyed out of each frame."""

    def __init__(self, monitor: int = 0, hide: Callable[[], str] | None = None) -> None:
        import mss  # optional dependency, only needed for the local backend

        if hide and not shutil.which("wmctrl"):
            raise CaptureError("hiding a window needs wmctrl (sudo apt install wmctrl)")
        self._mss = mss
        self._local = threading.local()
        self.monitor, self.hide = monitor, hide

    def area(self) -> dict[str, int]:
        """The captured area in desktop pixels: the space LocalHID's clicks map onto."""
        # shortcut: read once at startup; restart the agent after rearranging monitors.
        with self._mss.mss() as sct:
            return dict(sct.monitors[self.monitor])

    def grab(self) -> Frame | None:
        # mss handles are per-thread.
        sct = getattr(self._local, "sct", None)
        if sct is None:
            sct = self._local.sct = self._mss.mss()
        try:
            shot = sct.grab(sct.monitors[self.monitor])
        except self._mss.exception.ScreenShotError:
            # The handle caches the monitor layout; a screen added or removed makes that area invalid.
            sct.close()
            sct = self._local.sct = self._mss.mss()
            shot = sct.grab(sct.monitors[self.monitor])
        frame = np.ascontiguousarray(np.asarray(shot)[:, :, :3])
        title = self.hide() if self.hide else ""
        if title:
            _grey_out(frame, title, sct.monitors[self.monitor])
        return frame

    def signal_present(self) -> bool:
        return True

    def close(self) -> None:
        pass


def _grey_out(frame: Frame, title: str, area: dict[str, int]) -> None:
    """Grey out each window whose title starts with `title`, its frame (title bar) too."""
    # shortcut: greys the whole window even under other windows, minimized or on another workspace; read
    # `_NET_CLIENT_LIST_STACKING` and `wmctrl -d` if that hides a target.
    for x, y, w, h in window_rects(title):
        x, y = x - area["left"], y - area["top"]
        frame[max(y, 0):max(y + h, 0), max(x, 0):max(x + w, 0)] = 128


class CommandCapture:
    """Development stand-in: runs `command` (with `{path}` for the output image) per frame,
    e.g. `spectacle -b -n -f -o {path}` on KDE Wayland."""

    def __init__(self, command: str) -> None:
        if "{path}" not in command:
            raise ValueError("LOCAL_CAPTURE needs {path} where the screenshot is written")
        self._command = command
        self._dir = tempfile.TemporaryDirectory(prefix="exospace-capture-")
        self._lock = threading.Lock()  # one screenshot tool run at a time

    def grab(self) -> Frame | None:
        path = Path(self._dir.name) / "screen.png"
        with self._lock:
            args = [part.replace("{path}", str(path)) for part in shlex.split(self._command)]
            subprocess.run(args, capture_output=True, timeout=10, check=False)
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            path.unlink(missing_ok=True)
        return frame

    def signal_present(self) -> bool:
        return True

    def close(self) -> None:
        self._dir.cleanup()


def is_blank(frame: Frame) -> bool:
    """No picture at all. Sampled at full resolution: a thumbnail averaged away the mouse
    cursor on a black desktop, and a live but dark screen read as lost video."""
    return int(frame[::4, ::4].max()) < BLANK_MAX


def thumbnail(frame: Frame) -> np.ndarray:
    """Tiny grayscale copy for cheap change detection."""
    small = cv2.resize(frame, THUMB_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def frame_change(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference between two thumbnails (0-255)."""
    return float(np.mean(cv2.absdiff(a, b)))


def active_rect(frame: Frame) -> Rect:
    """Find the real screen inside the frame.

    A mirrored phone arrives as a portrait image with black bars on both sides
    (pillarbox). Only symmetric bars on one axis count: a scaled-to-fit picture
    always touches two opposite edges, while a window centred on a black wallpaper
    has bars on all four sides and must not become the whole "screen".
    """
    h, w = frame.shape[:2]
    full = Rect(0, 0, w, h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cols = np.flatnonzero(gray.max(axis=0) >= BLANK_MAX)
    rows = np.flatnonzero(gray.max(axis=1) >= BLANK_MAX)
    if cols.size == 0 or rows.size == 0:
        return full

    left, right = int(cols[0]), w - 1 - int(cols[-1])
    top, bottom = int(rows[0]), h - 1 - int(rows[-1])
    sides, ends = _symmetric_bars(left, right, w), _symmetric_bars(top, bottom, h)
    if sides == ends:
        return full
    rect = Rect(left, 0, w - right, h) if sides else Rect(0, top, w, h - bottom)
    # A dark loading page lit only by the cursor once "cropped" the screen to an 18 px strip.
    aspect = (rect.x1 - rect.x0) / (rect.y1 - rect.y0)
    return rect if ASPECT_RANGE[0] <= aspect <= ASPECT_RANGE[1] else full


def _symmetric_bars(a: int, b: int, size: int) -> bool:
    min_bar = size * BAR_MIN_FRACTION
    return a >= min_bar and b >= min_bar and abs(a - b) <= size * 0.02
