"""Seeing the target: frame sources and frame analysis.

`V4L2Capture` reads the HDMI capture card on the Pi. `ScreenCapture` is the
development stand-in that captures the machine you're on. Both keep only the
latest frame in memory, so callers always see a fresh screen.
"""

from __future__ import annotations

import threading
import time
from typing import Protocol

import cv2
import numpy as np

from brain.types import Frame, Rect

STALE_S = 0.5  # a frame older than this counts as no signal
BLANK_MAX = 12  # brightest pixel below this (0-255) = blank / no signal
BAR_MIN_FRACTION = 0.05  # pillarbox bars must each be at least this wide
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
    """Development stand-in: captures this machine's whole desktop."""

    def __init__(self) -> None:
        import mss  # optional dependency, only needed for the local backend

        self._mss = mss
        self._local = threading.local()

    def grab(self) -> Frame | None:
        # mss handles are per-thread.
        sct = getattr(self._local, "sct", None)
        if sct is None:
            sct = self._local.sct = self._mss.mss()
        try:
            shot = sct.grab(sct.monitors[0])  # every monitor, the same space LocalHID's clicks map onto
        except self._mss.exception.ScreenShotError:
            # The handle caches the monitor layout; a screen added or removed makes that area invalid.
            sct.close()
            sct = self._local.sct = self._mss.mss()
            shot = sct.grab(sct.monitors[0])
        return np.ascontiguousarray(np.asarray(shot)[:, :, :3])

    def signal_present(self) -> bool:
        return True

    def close(self) -> None:
        pass


def is_blank(frame: Frame) -> bool:
    return int(thumbnail(frame).max()) < BLANK_MAX


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
    (pillarbox). Only symmetric bars count, so a dark desktop wallpaper doesn't
    get cropped by mistake.
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
    x0, x1, y0, y1 = 0, w, 0, h
    if _symmetric_bars(left, right, w):
        x0, x1 = left, w - right
    if _symmetric_bars(top, bottom, h):
        y0, y1 = top, h - bottom
    return Rect(x0, y0, x1, y1)


def _symmetric_bars(a: int, b: int, size: int) -> bool:
    min_bar = size * BAR_MIN_FRACTION
    return a >= min_bar and b >= min_bar and abs(a - b) <= size * 0.02
