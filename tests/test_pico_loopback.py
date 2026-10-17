"""Pi <-> Pico protocol over a pseudo-terminal, before the hardware exists.

The Pi side is the real `PicoHID` + `Controller`; the Pico side is the real firmware
`Commands` handler behind the same newline-JSON framing as firmware/pico/code.py,
with mock USB devices standing in for the host.
"""

import json
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "firmware" / "pico"))

from commands import Commands  # noqa: E402
from reports import BUTTON_LEFT, BUTTON_RIGHT  # noqa: E402

from brain.cancel import CancelToken  # noqa: E402
from brain.controller import Controller  # noqa: E402
from brain.hid import HIDError, PicoHID  # noqa: E402
from brain.types import Action, DeviceProfile, Rect  # noqa: E402


class FakePico:
    """firmware/pico/code.py's loop on the far end of a pty."""

    def __init__(self, fd: int) -> None:
        self.fd, self.mounted = fd, True
        self.keyboard, self.mouse, self.touch = Mock(), Mock(), Mock()
        self.commands = Commands(self.keyboard, self.mouse, self.touch, lambda: self.mounted, lambda: False)
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def send(self, message: dict) -> None:
        os.write(self.fd, json.dumps(message).encode() + b"\n")

    def _loop(self) -> None:
        buffer = b""
        while not self._stop.is_set():
            buffer += os.read(self.fd, 4096)
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                if line.strip():
                    self.send(self.commands.handle(json.loads(line)))

    def close(self) -> None:
        self._stop.set()


class LoopbackTests(unittest.TestCase):
    def setUp(self):
        master, slave = os.openpty()
        self.pico = FakePico(master)
        self.hid = PicoHID(os.ttyname(slave), 115200)
        self.addCleanup(self.hid.close)
        self.addCleanup(self.pico.close)

    def test_ping_reports_usb_state(self):
        self.assertTrue(self.hid.is_mounted())

    def test_gestures_reach_the_mouse(self):
        controller = Controller(self.hid, Mock(), CancelToken())
        profile = DeviceProfile("windows", Rect(0, 0, 1920, 1080), "mouse", 1)
        controller.execute(Action("double_click", x=0.5, y=0.25), profile)
        controller.execute(Action("right_click", x=0.1, y=0.9), profile)
        (x, y, button, count), _ = self.pico.mouse.click.call_args_list[0]
        self.assertEqual((button, count), (BUTTON_LEFT, 2))
        self.assertAlmostEqual(x / 32767, 0.5, places=3)
        self.assertEqual(self.pico.mouse.click.call_args_list[1].args[2:], (BUTTON_RIGHT, 1))

    def test_touch_profile_taps(self):
        controller = Controller(self.hid, Mock(), CancelToken())
        controller.execute(Action("click", x=0.3, y=0.7), DeviceProfile("android", Rect(0, 0, 9, 16), "touch", 1))
        self.pico.touch.tap.assert_called_once()

    def test_firmware_errors_surface_as_hid_errors(self):
        with self.assertRaises(HIDError):
            self.hid.send({"op": "click", "x": 1, "y": 1, "count": 3})

    def test_unplug_event_reaches_the_pi(self):
        events = []
        self.hid.on_event(events.append)
        self.pico.send({"ev": "unmounted"})
        deadline = time.monotonic() + 2
        while not events and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(events, ["unmounted"])
        self.assertFalse(self.hid.is_mounted())


if __name__ == "__main__":
    unittest.main()
