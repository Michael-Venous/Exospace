"""Identify a connected target from its first usable frame."""

from __future__ import annotations

import time

from brain.agent.executor import Executor
from brain.cancel import Cancelled, CancelToken
from brain.capture import Capture, active_rect, is_blank
from brain.imaging import prepare
from brain.types import DeviceProfile, OSName


def detect_profile(capture: Capture, executor: Executor, override_os: OSName | None = None,
                   *, home_os: OSName | None = None, cancel: CancelToken | None = None,
                   timeout_s: float = 5.0) -> DeviceProfile:
    """`home_os` is the target's own OS: any classification but firmware becomes it, since one
    screenshot of a full-screen app shows no OS (Linux with Chrome was read as windows and chromeos)."""
    token = cancel or CancelToken()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        token.check()
        frame = capture.grab()
        if frame is not None and not is_blank(frame):
            rect = active_rect(frame)
            os = override_os or _classify(executor, prepare(frame, rect, grid=False))
            token.check()
            if home_os and os != "bios":
                os = home_os
            # "unknown" (BIOS, boot splash, full-screen app) still works as keyboard + mouse.
            return DeviceProfile(os, rect, "touch" if os == "android" else "mouse", time.time())
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
