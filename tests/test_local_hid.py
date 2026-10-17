"""LocalHID routing without real input."""

import sys
import unittest
from unittest.mock import Mock, patch

from brain.hid import VOICE_WINDOW, LocalHID
from brain.types import ActionError


@patch("brain.hid._focused_title", return_value="New Tab - Google Chrome")
class LocalHIDTests(unittest.TestCase):
    def test_keys_go_to_the_keyboard_and_clicks_stay_local(self, _title):
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

    def test_keys_meant_for_the_target_never_reach_the_voice_window(self, title):
        title.return_value = VOICE_WINDOW
        with patch.dict(sys.modules, {"pyautogui": Mock(**{"size.return_value": (100, 100)})}):
            hid = LocalHID()
        typing = {"op": "type", "text": "Genghis Khan"}
        self.assertRaises(ActionError, hid.send, typing)
        for cmd in ({"op": "key", "keys": ["GUI"]}, typing, {"op": "key", "keys": ["ENTER"]}):
            hid.send(cmd)  # open_app: the menu has focus though X still names the voice window
        self.assertRaises(ActionError, hid.send, typing)
        hid.send({"op": "key", "keys": ["GUI"]})
        hid.send({"op": "click", "x": 0, "y": 0})  # a click closes the menu
        self.assertRaises(ActionError, hid.send, typing)


if __name__ == "__main__":
    unittest.main()
