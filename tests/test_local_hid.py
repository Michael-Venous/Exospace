"""LocalHID routing without real input."""

import sys
import unittest
from unittest.mock import Mock, patch

from brain.hid import VOICE_WINDOW, LocalHID, _focus_window_below
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

    def test_clicks_land_on_the_captured_monitor(self, _title):
        gui = Mock()
        with patch.dict(sys.modules, {"pyautogui": gui}):
            hid = LocalHID(area={"left": 0, "top": 1800, "width": 2880, "height": 1800})
        hid.send({"op": "click", "x": 0, "y": 0})
        hid.send({"op": "click", "x": 32767, "y": 32767})
        self.assertEqual([c.args for c in gui.click.call_args_list], [(0, 1800), (2879, 3599)])

    @patch("brain.hid._focus_window_below", return_value=False)
    def test_keys_meant_for_the_target_never_reach_the_voice_window(self, _below, title):
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

    def test_keys_go_to_the_window_the_user_was_in(self, title):
        title.return_value = VOICE_WINDOW
        gui = Mock(**{"size.return_value": (100, 100)})
        with patch.dict(sys.modules, {"pyautogui": gui}):
            hid = LocalHID()
        with patch("brain.hid._focus_window_below", return_value=True) as below:
            hid.send({"op": "key", "keys": ["CTRL", "T"]})
        below.assert_called_once()
        gui.hotkey.assert_called_once_with("ctrl", "t")

    def test_keys_never_go_to_a_terminal_below(self, title):
        windows = {"0x1": ("Exospace - Visual Studio Code", "code"), "0x2": ("New Tab - Google Chrome", "google-chrome"),
                   "0x3": ("claude", "gnome-terminal-server"), "0x4": (VOICE_WINDOW, "google-chrome")}

        def run(order):
            def fake(cmd, **_):
                out = {"_NET_CURRENT_DESKTOP": "_NET_CURRENT_DESKTOP(CARDINAL) = 0",
                       "_NET_CLIENT_LIST_STACKING": "_NET_CLIENT_LIST_STACKING(WINDOW): window id # " + ", ".join(order)}
                if cmd[0] == "wmctrl":
                    out = "\n".join(f"{w}  0 host {t}" for w, (t, _) in windows.items()) if cmd[1] == "-l" else ""
                    activated.extend(cmd[3:])
                elif cmd[1] == "-id":
                    out = f'WM_CLASS(STRING) = "{windows[cmd[2]][1]}"\n_NET_WM_WINDOW_TYPE(ATOM) = _NET_WM_WINDOW_TYPE_NORMAL'
                else:
                    out = out[cmd[-1]]
                return Mock(stdout=out)
            activated = []
            with patch("brain.hid.subprocess.run", fake):
                return _focus_window_below(), activated

        title.return_value = "New Tab - Google Chrome"
        self.assertEqual(run(["0x1", "0x3", "0x2", "0x4"]), (True, ["0x2"]))  # the browser the user was in
        self.assertEqual(run(["0x2", "0x3", "0x4"]), (False, []))  # never past it to a browser further down
        self.assertEqual(run(["0x2", "0x1", "0x4"]), (False, []))


if __name__ == "__main__":
    unittest.main()
