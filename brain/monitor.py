"""Watches the target connection and fires the cancel path on unplug."""

from __future__ import annotations

import threading
import time
from typing import Callable

from brain.capture import Capture
from brain.hid import HID

POLL_S = 0.05
BLANK_GRACE_S = 1.0  # video can flicker black briefly (mode switches, rotation)


class ConnectionMonitor:
    """A target counts as connected when the Pico is mounted and video shows.

    The Pico unmounting cancels immediately; lost video cancels after a short
    grace period. Each new connection bumps `generation`, so callers can tell
    the device may have changed and re-detect it.
    """

    def __init__(self, capture: Capture, hid: HID, on_disconnect: Callable[[str], None]) -> None:
        self._capture = capture
        self._hid = hid
        self._on_disconnect = on_disconnect
        self._cond = threading.Condition()
        self._connected = False
        self._generation = 0
        self._last_video = time.monotonic()
        self._seen_video = False
        self._stop = threading.Event()
        hid.on_event(self._on_hid_event)
        self._thread = threading.Thread(target=self._loop, name="monitor", daemon=True)
        self._thread.start()

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def generation(self) -> int:
        return self._generation

    def wait_connected(self) -> int:
        """Block until a target is connected; returns its generation."""
        with self._cond:
            self._cond.wait_for(lambda: self._connected or self._stop.is_set())
            if self._stop.is_set():
                raise RuntimeError("connection monitor closed")
            return self._generation

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=1.0)

    def _on_hid_event(self, event: str) -> None:
        if event == "unmounted":
            self._mark_disconnected("target disconnected")

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            mounted = self._hid.is_mounted()
            if self._capture.signal_present():
                self._last_video = now
                self._seen_video = True
            video_ok = self._seen_video and now - self._last_video <= BLANK_GRACE_S
            if mounted and video_ok:
                self._mark_connected()
            elif self._connected:
                self._mark_disconnected("target disconnected" if not mounted else "video lost")
            self._stop.wait(POLL_S)

    def _mark_connected(self) -> None:
        with self._cond:
            if not self._connected:
                self._connected = True
                self._generation += 1
                self._cond.notify_all()

    def _mark_disconnected(self, reason: str) -> None:
        with self._cond:
            if not self._connected:
                return
            self._connected = False
        self._on_disconnect(reason)
