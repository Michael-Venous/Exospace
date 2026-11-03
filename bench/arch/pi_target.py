"""The real rig as a queue target (bench only): frames from the capture card, input through the Pico.

The laptop rig (`laptop.py`) is the brain and the target at once; on the Pi the target is another computer. The
queue needs one object that grabs frames and sends PLAN.md §5.1 commands, as `LaptopTarget` is. Frames are cropped
to the screen inside any black border (`active_rect`, as the runtime does), so 0-1 coordinates of a frame are 0-1
of the target's screen, the Pico's absolute range. Not yet run on the hardware.
"""

from __future__ import annotations

import time

from brain.capture import CaptureError, V4L2Capture, active_rect
from brain.config import Config
from brain.hid import PicoHID

FIRST_FRAME_S = 5.0  # how long to wait for the capture card's first frame


class PiTarget:
    # The Pico's protocol has no held keys during a click and no triple click (PLAN.md §5.1: count 1 or 2).
    features: frozenset[str] = frozenset()
    launcher_settles = True  # the target's Cinnamon menu takes ~0.3 s to accept typing after Super

    def __init__(self, cfg: Config) -> None:
        self.capture = V4L2Capture(cfg.capture_device, cfg.capture_width, cfg.capture_height)
        self.hid = PicoHID(cfg.pico_serial, cfg.pico_baud)
        deadline = time.monotonic() + FIRST_FRAME_S
        while self.capture.grab() is None:
            if time.monotonic() > deadline:
                self.close()
                raise CaptureError(f"no frame from {cfg.capture_device} in {FIRST_FRAME_S:g} s")
            time.sleep(0.05)

    def grab(self):
        frame = self.capture.grab()
        if frame is None:
            raise CaptureError("the capture card has no current frame (signal lost?)")
        r = active_rect(frame)
        return frame[r.y0:r.y1, r.x0:r.x1]

    def send(self, command: dict) -> None:
        self.hid.send(command)

    def signal_present(self) -> bool:
        return self.capture.signal_present()

    def is_mounted(self) -> bool:
        return self.hid.is_mounted()

    def on_event(self, callback) -> None:
        self.hid.on_event(callback)

    def close(self) -> None:
        self.hid.close()
        self.capture.close()
