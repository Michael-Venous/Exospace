"""Identify a connected target from its first usable frame."""

from __future__ import annotations

import time
from collections.abc import Callable

from brain.agent.executor import Executor
from brain.calibrate import pointer_rect
from brain.cancel import Cancelled, CancelToken
from brain.capture import Capture, active_rect, is_blank
from brain.imaging import prepare
from brain.types import DeviceProfile, OSName


def detect_profile(capture: Capture, executor: Executor, override_os: OSName | None = None,
                   *, cancel: CancelToken | None = None, timeout_s: float = 5.0,
                   pointer: Callable[[float, float], None] | None = None) -> DeviceProfile:
    """`pointer(x, y)` moves the target's absolute mouse; with it, a mouse target's screen is found
    from where its cursor appears (brain/calibrate.py), not from black bars."""
    token = cancel or CancelToken()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        token.check()
        frame = capture.grab()
        if frame is not None and not is_blank(frame):
            rect = active_rect(frame)
            os = override_os or _classify(executor, prepare(frame, rect, grid=False))
            token.check()
            mode = "touch" if os == "android" else "mouse"
            if pointer is not None and mode == "mouse":
                rect = pointer_rect(capture, pointer) or rect
            # "unknown" (BIOS, boot splash, full-screen app) still works as keyboard + mouse.
            return DeviceProfile(os, rect, mode, time.time())
        token.sleep(0.05)
    raise RuntimeError("No usable video frame; check the capture card and target display")


def _classify(executor: Executor, jpeg: bytes) -> OSName:
    """A model error (network blip after a device swap) must not make the target unusable."""
    try:
        return executor.classify_os(jpeg)
    except Cancelled:
        raise
    except Exception:  # noqa: BLE001 - fall back to a generic keyboard + mouse target
        return "unknown"
