"""The PC <-> laptop input contract: whatever the controller sends, the laptop helper can carry out.

The PC side is the real `Controller` (and its text folding); the laptop side is the real `laptop_hid.handle`
with its uinput writes recorded instead of sent. No devices are created and nothing is typed.
"""

import string
import unittest
from unittest.mock import patch

import numpy as np

from bench.arch import laptop_hid
from brain.cancel import CancelToken
from brain.controller import Controller, SettleConfig, _ascii
from brain.types import Action, DeviceProfile, Rect

EV_KEY = laptop_hid.EV_KEY


class Target:
    """What the controller talks to: records each command, shows a still screen."""

    def __init__(self):
        self.sent = []

    def send(self, command):
        self.sent.append(command)

    def grab(self):
        return np.zeros((100, 160, 3), np.uint8)

    def signal_present(self):
        return True

    def is_mounted(self):
        return True

    def on_event(self, callback):
        pass


class LaptopContractTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        for name, value in (("emit", lambda fd, *events: self.events.extend(events)), ("time", _NoSleep())):
            patcher = patch.object(laptop_hid, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.target = Target()
        fast = SettleConfig(min_s=0, react_s=0, load_stable_s=0, load_react_s=0, stable_s=0, timeout_s=.01, poll_s=.001)
        self.controller = Controller(self.target, self.target, CancelToken(), fast)
        self.profile = DeviceProfile("linux", Rect(0, 0, 160, 100), "mouse", 1)

    def typed(self, text):
        """The characters the laptop's keyboard produced for `text`."""
        self.events.clear()
        laptop_hid.type_text(text)
        codes = {code: char for char, (code, shift) in laptop_hid.CHARS.items() if not shift}
        out, shift = [], False
        for kind, code, value in self.events:
            if kind != EV_KEY:
                continue
            if code == laptop_hid.NAMED["SHIFT"]:
                shift = value == 1
            elif value == 1:
                out.append({c for c, (k, s) in laptop_hid.CHARS.items() if k == code and s == shift}.pop()
                           if shift else codes[code])
        return "".join(out)

    def test_every_command_the_executor_sends_is_carried_out(self):
        c, p = self.controller, self.profile
        for action in (Action("click", x=.5, y=.5), Action("double_click", x=.1, y=.9), Action("right_click", x=1, y=0),
                       Action("swipe", x=.2, y=.2, x2=.8, y2=.8, ms=600), Action("scroll", dy=3),
                       Action("scroll", dy=-800)):
            c.execute(action, p)
        for keys in (["CTRL", "L"], ["ALT", "F4"], ["GUI", "PAGEUP"], ["CTRL", "SHIFT", "S"], ["ENTER"], ["ESC"],
                     ["F12"], ["HOME"], ["a"], ["7"]):
            c.hotkey(keys)
        c.hover(.3, .4)
        c.type_text("Hello, world! 41261 main-dash ~/notes.txt")
        c.hid.send({"op": "click", "x": 100, "y": 200, "button": "left", "count": 1, "hold": ["CTRL"]})  # held-key
        c.release_all()
        ops = {command["op"] for command in self.target.sent}
        self.assertTrue({"click", "drag", "scroll", "key", "move", "type", "release_all"} <= ops, ops)
        for command in self.target.sent:
            laptop_hid.handle(command)  # raises on anything the laptop can't do

    def test_any_text_the_pc_side_sends_types_exactly(self):
        text = _ascii(string.printable.replace("\x0b", "").replace("\x0c", "").replace("\r", ""))
        self.assertEqual(self.typed(text), text)
        self.assertEqual(self.typed(_ascii("café — “quoted” ‘x’")), 'cafe - "quoted" \'x\'')

    def test_a_windows_line_break_is_one_enter(self):
        sent = _ascii("first\r\nsecond\rthird")  # the PC side keeps \r; this once failed mid-text on the laptop
        self.assertEqual(self.typed(sent), "first\nsecond\nthird")

    def test_untypable_text_types_nothing(self):
        self.events.clear()
        with self.assertRaises(ValueError):
            laptop_hid.type_text("half typed\x0b")
        self.assertEqual(self.events, [])

    def test_scrolls_and_clicks_reach_the_pointer(self):
        self.controller.execute(Action("click", x=.5, y=.5), self.profile)
        self.controller.execute(Action("scroll", dy=2), self.profile)
        self.events.clear()
        for command in self.target.sent:
            laptop_hid.handle(command)
        self.assertIn((EV_KEY, laptop_hid.BTN["left"], 1), self.events)
        self.assertEqual([e for e in self.events if e[0] == laptop_hid.EV_REL],
                         [(laptop_hid.EV_REL, laptop_hid.REL_WHEEL, -1)] * 2)


class _NoSleep:
    """`time` for the helper module: clicks and typing don't wait in tests."""

    @staticmethod
    def sleep(seconds):
        pass

    @staticmethod
    def time():
        return 0.0


if __name__ == "__main__":
    unittest.main()
