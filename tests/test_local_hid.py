"""LocalHID routing without real input."""

import sys
import unittest
from unittest.mock import Mock, patch

from brain.hid import LocalHID


class LocalHIDTests(unittest.TestCase):
    def test_keys_go_to_the_keyboard_and_clicks_stay_local(self):
        gui, keyboard = Mock(), Mock()
        gui.size.return_value = (100, 100)
        with patch.dict(sys.modules, {"pyautogui": gui}):
            hid = LocalHID(keyboard)
        for cmd in ({"op": "key", "keys": ["GUI"]}, {"op": "type", "text": "hi"},
                    {"op": "click", "x": 0, "y": 0}, {"op": "release_all"}):
            hid.send(cmd)
        self.assertEqual([c.args[0]["op"] for c in keyboard.send.call_args_list], ["key", "type", "release_all"])
        gui.hotkey.assert_not_called()
        gui.write.assert_not_called()
        gui.click.assert_called_once()
        gui.mouseUp.assert_called_once()  # release_all still lets go of the local mouse


if __name__ == "__main__":
    unittest.main()
