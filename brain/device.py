"""Identify a connected target from its first usable frame."""

from __future__ import annotations

import time

from brain.agent.executor import Executor
from brain.cancel import CancelToken
from brain.capture import Capture, active_rect, is_blank
from brain.imaging import prepare
from brain.types import DeviceProfile, OSName


def detect_profile(capture: Capture, executor: Executor, override_os: OSName | None = None,
                   *, cancel: CancelToken | None = None, timeout_s: float = 5.0) -> DeviceProfile:
    token = cancel or CancelToken()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        token.check()
        frame = capture.grab()
        if frame is not None and not is_blank(frame):
            rect = active_rect(frame)
            os = override_os or executor.classify_os(prepare(frame, rect, grid=False))
            token.check()
            if os == "unknown":
                raise RuntimeError("Cannot identify target OS; supply --os explicitly")
            return DeviceProfile(os, rect, "touch" if os == "android" else "mouse", time.time())
        token.sleep(0.05)
    raise RuntimeError("No usable video frame; check the capture card and target display")
