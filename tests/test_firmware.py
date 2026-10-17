"""Pico protocol logic, run on a PC with fake devices (no hardware needed)."""

import struct
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "firmware" / "pico"))

from commands import Commands  # noqa: E402
from keys import resolve  # noqa: E402
from reports import BUTTON_LEFT, BUTTON_RIGHT, mouse_report, touch_report  # noqa: E402


class FakeKeycode:
    CONTROL = 0xE0
    GUI = 0xE3
    ESCAPE = 0x29
    ONE = 0x1E
    L = 0x0F
    name = "not a key"


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.keyboard, self.mouse, self.touch = Mock(), Mock(), Mock()
        self.touch.swipe.return_value = True
        self.mounted = True
        self.commands = Commands(
            self.keyboard, self.mouse, self.touch, lambda: self.mounted, lambda: False
        )

    def run_cmd(self, **cmd):
        return self.commands.handle({"id": 7, **cmd})

    def test_ping_reports_mount_state(self):
        self.assertEqual(self.run_cmd(op="ping"), {"id": 7, "ok": True, "mounted": True})
        self.mounted = False
        self.assertFalse(self.run_cmd(op="ping")["mounted"])

    def test_key_and_type(self):
        self.assertTrue(self.run_cmd(op="key", keys=["CTRL", "L"])["ok"])
        self.keyboard.press_keys.assert_called_once_with(["CTRL", "L"])
        self.assertTrue(self.run_cmd(op="type", text="hello")["ok"])
        self.keyboard.type_text.assert_called_once_with("hello")

    def test_type_rejects_long_text(self):
        reply = self.run_cmd(op="type", text="x" * 33)
        self.assertFalse(reply["ok"])
        self.keyboard.type_text.assert_not_called()

    def test_pointer_ops(self):
        self.run_cmd(op="move", x=10, y=20)
        self.mouse.move.assert_called_once_with(10, 20)
        self.run_cmd(op="click", x=1, y=2, button="right", count=2)
        self.mouse.click.assert_called_once_with(1, 2, BUTTON_RIGHT, 2)
        self.run_cmd(op="click", x=1, y=2)
        self.mouse.click.assert_called_with(1, 2, BUTTON_LEFT, 1)
        self.run_cmd(op="scroll", dy=-3)
        self.mouse.scroll.assert_called_once_with(-3)

    def test_touch_ops(self):
        self.run_cmd(op="tap", x=5, y=6)
        self.touch.tap.assert_called_once_with(5, 6)
        self.run_cmd(op="swipe", x1=0, y1=1, x2=2, y2=3, ms=300)
        self.touch.swipe.assert_called_once_with(0, 1, 2, 3, 300, self.commands._should_abort)

    def test_swipe_duration_is_clamped(self):
        self.run_cmd(op="swipe", x1=0, y1=1, x2=2, y2=3, ms=9999)
        self.assertEqual(self.touch.swipe.call_args.args[4], 2000)
        self.run_cmd(op="swipe", x1=0, y1=1, x2=2, y2=3, ms=0)
        self.assertEqual(self.touch.swipe.call_args.args[4], 1)

    def test_aborted_swipe_is_an_error(self):
        self.touch.swipe.return_value = False
        reply = self.run_cmd(op="swipe", x1=0, y1=1, x2=2, y2=3, ms=300)
        self.assertEqual((reply["ok"], reply["err"]), (False, "aborted"))

    def test_release_all_clears_every_device(self):
        self.assertTrue(self.run_cmd(op="release_all")["ok"])
        for device in (self.keyboard, self.mouse, self.touch):
            device.release_all.assert_called_once_with()

    def test_bad_input_returns_error_with_id(self):
        for cmd in (
            {"op": "nope"},
            {"op": "move", "x": 40000, "y": 0},
            {"op": "move", "x": "1", "y": 0},
            {"op": "move", "y": 0},
            {"op": "click", "x": 0, "y": 0, "count": 3},
            {"op": "click", "x": 0, "y": 0, "button": "middle"},
            {"op": "key", "keys": []},
            {"op": "swipe", "x1": 0, "y1": 0, "x2": 1, "y2": 1, "ms": "slow"},
        ):
            reply = self.run_cmd(**cmd)
            self.assertFalse(reply["ok"], cmd)
            self.assertEqual(reply["id"], 7)
            self.assertTrue(reply["err"])

    def test_usb_failure_becomes_error_reply(self):
        self.mouse.move.side_effect = OSError()
        self.assertEqual(self.run_cmd(op="move", x=1, y=1)["err"], "usb send failed")

    def test_device_value_error_becomes_error_reply(self):
        self.keyboard.press_keys.side_effect = ValueError("unknown key FOO")
        self.assertEqual(self.run_cmd(op="key", keys=["FOO"])["err"], "unknown key FOO")


class KeyTests(unittest.TestCase):
    def test_names_and_aliases(self):
        self.assertEqual(resolve("CTRL", FakeKeycode), FakeKeycode.CONTROL)
        self.assertEqual(resolve("CMD", FakeKeycode), FakeKeycode.GUI)
        self.assertEqual(resolve("esc", FakeKeycode), FakeKeycode.ESCAPE)
        self.assertEqual(resolve("1", FakeKeycode), FakeKeycode.ONE)
        self.assertEqual(resolve("l", FakeKeycode), FakeKeycode.L)

    def test_unknown_key_raises(self):
        for name in ("FOO", "NAME"):
            with self.assertRaises(ValueError):
                resolve(name, FakeKeycode)


class ReportTests(unittest.TestCase):
    def test_mouse_report_layout(self):
        report = mouse_report(BUTTON_LEFT, 32767, 1, -3)
        self.assertEqual(struct.unpack("<BHHb", report), (1, 32767, 1, -3))
        self.assertEqual(len(report), 6)

    def test_mouse_wheel_is_clamped(self):
        self.assertEqual(mouse_report(0, 0, 0, 500)[-1], 127)

    def test_touch_report_layout(self):
        self.assertEqual(struct.unpack("<BBHHB", touch_report(True, 4, 5)), (3, 0, 4, 5, 1))
        self.assertEqual(struct.unpack("<BBHHB", touch_report(False, 4, 5)), (0, 0, 4, 5, 1))
        self.assertEqual(len(touch_report(True, 0, 0)), 7)


if __name__ == "__main__":
    unittest.main()
