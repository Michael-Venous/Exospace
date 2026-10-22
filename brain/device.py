"""Identify a connected target from its first usable frame."""

from __future__ import annotations

import time
from collections.abc import Callable

from brain.agent.executor import Executor
from brain.calibrate import pointer_rect
from brain.cancel import Cancelled, CancelToken
from brain.capture import Capture, active_rect, frame_change, is_blank, thumbnail
from brain.imaging import prepare
from brain.types import DeviceProfile, OSName


PROBE_S = 3.0  # longest wait for the launcher or Start menu to appear after the OS key
PROBE_SETTLE_S = 0.5  # then for its opening animation to finish
PROBE_CHANGE = 1.5  # mean thumbnail change (0-255) that counts as the launcher appearing, as in settling


def detect_profile(capture: Capture, executor: Executor, override_os: OSName | None = None,
                   *, cancel: CancelToken | None = None, timeout_s: float = 5.0,
                   pointer: Callable[[float, float], None] | None = None,
                   press: Callable[[list[str]], None] | None = None) -> DeviceProfile:
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
            if os == "unknown" and press is not None:
                os = _probe(capture, executor, rect, press, token)
            mode = "touch" if os == "android" else "mouse"
            if pointer is not None and mode == "mouse":
                rect = pointer_rect(capture, pointer) or rect
            # "unknown" (BIOS, boot splash, full-screen app) still works as keyboard + mouse.
            return DeviceProfile(os, rect, mode, time.time())
        token.sleep(0.05)
    raise RuntimeError("No usable video frame; check the capture card and target display")


def _probe(capture: Capture, executor: Executor, rect, press: Callable[[list[str]], None],
           token: CancelToken) -> OSName:
    """A screen that shows nothing to tell the OS by (a black, empty desktop) gets the OS key once: a
    Start menu or desktop launcher is unmistakable, and Escape closes it again. Without an OS the app
    launcher is off and every request fell to the planner guessing at a black screen."""
    before = capture.grab()
    press(["GUI"])
    # Wait for the launcher itself: after a fixed 1.2 s the Linux laptop's startup still classified "unknown".
    deadline = time.monotonic() + PROBE_S
    frame = capture.grab()
    while (before is not None and frame is not None and time.monotonic() < deadline
           and frame_change(thumbnail(before), thumbnail(frame)) < PROBE_CHANGE):
        token.sleep(0.05)
        frame = capture.grab()
    token.sleep(PROBE_SETTLE_S)
    frame = capture.grab()
    os = _classify(executor, prepare(frame, rect, grid=False)) if frame is not None and not is_blank(frame) else "unknown"
    press(["ESCAPE"])
    return os


def _classify(executor: Executor, jpeg: bytes) -> OSName:
    """A model error (network blip after a device swap) must not make the target unusable."""
    try:
        return executor.classify_os(jpeg)
    except Cancelled:
        raise
    except Exception:  # noqa: BLE001 - fall back to a generic keyboard + mouse target
        return "unknown"
