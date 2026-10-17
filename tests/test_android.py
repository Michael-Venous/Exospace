import subprocess
import unittest
from unittest import mock

from brain.android import AndroidTarget
from brain.hid import HID_MAX, HIDError


class AndroidTargetTests(unittest.TestCase):
    def test_commands_become_adb_input(self):
        shell = []

        def run(args, **kwargs):
            shell.append(args[2])
            out = b"Physical size: 1080x2400\n" if args[2] == "wm size" else b""
            return subprocess.CompletedProcess(args, 0, out, b"")

        with mock.patch("brain.android.subprocess.run", side_effect=run):
            target = AndroidTarget()
            for cmd in ({"op": "key", "keys": ["ENTER"]}, {"op": "key", "keys": ["GUI", "H"]},
                        {"op": "type", "text": "it's 2+2"}, {"op": "tap", "x": HID_MAX, "y": 0},
                        {"op": "swipe", "x1": 0, "y1": HID_MAX, "x2": 0, "y2": 0, "ms": 300}):
                target.send(cmd)
            with self.assertRaises(HIDError):
                target.send({"op": "click", "x": 0, "y": 0, "button": "right", "count": 1})
        self.assertEqual(shell[1:], [
            "input keyevent KEYCODE_ENTER",
            "input keycombination KEYCODE_META_LEFT KEYCODE_H",
            "input text 'it'\"'\"'s 2+2'",
            "input tap 1079 0",
            "input swipe 0 2399 0 0 300",
        ])


if __name__ == "__main__":
    unittest.main()
