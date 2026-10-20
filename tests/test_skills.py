"""Shortcut checks without model calls or real keyboard input."""

import unittest

import numpy as np
from unittest.mock import Mock

from brain.cancel import Cancelled, CancelToken
from brain.skills import SkillUnavailable, run_skill
from brain.types import ActionError, DeviceProfile, Rect


class SkillTests(unittest.TestCase):
    def setUp(self):
        self.controller = Mock()
        self.controller.cancel = CancelToken()
        self.controller.capture.grab.return_value = None  # no frame: open_app skips waiting for the window

    def profile(self, os):
        return DeviceProfile(os, Rect(0, 0, 100, 100), "mouse", 0)

    def test_platform_shortcuts(self):
        for os, keys in (("macos", ["GUI", "SPACE"]), ("windows", ["GUI"]), ("linux", ["GUI"]),
                         ("chromeos", ["GUI"])):
            with self.subTest(os=os):
                self.controller.reset_mock()
                run_skill("open_app", {"name": " Notes "}, self.profile(os), self.controller)
                self.assertEqual(self.controller.method_calls, [
                    unittest.mock.call.capture.grab(),
                    unittest.mock.call.hotkey(keys),
                    unittest.mock.call.type_text("Notes"),
                    unittest.mock.call.wait_settled(react=os != "linux"),
                    unittest.mock.call.capture.grab(),
                    unittest.mock.call.hotkey(["ENTER"]),
                ])

    def test_open_app_types_only_once_the_launcher_shows(self):
        self.controller.capture.grab.return_value = np.zeros((10, 10, 3), np.uint8)
        run_skill("open_app", {"name": "Notes"}, self.profile("macos"), self.controller)
        calls = [c[0] for c in self.controller.method_calls]
        self.assertEqual(calls[calls.index("hotkey"):calls.index("type_text")],
                         ["hotkey", "wait_for_change"])
        self.assertEqual(self.controller.method_calls[calls.index("wait_for_change")],
                         unittest.mock.call.wait_for_change(2.0, since=unittest.mock.ANY))

    def test_open_app_types_nothing_when_the_launcher_never_shows(self):
        self.controller.capture.grab.return_value = np.zeros((10, 10, 3), np.uint8)
        self.controller.wait_for_change.return_value = False
        with self.assertRaises(ActionError):
            run_skill("open_app", {"name": "Notes"}, self.profile("linux"), self.controller)
        self.controller.type_text.assert_not_called()
        self.controller.hotkey.assert_called_once_with(["GUI"])

    def test_open_app_waits_for_the_window_not_just_the_launcher(self):
        self.controller.capture.grab.return_value = np.zeros((10, 10, 3), np.uint8)
        run_skill("open_app", {"name": "Notes"}, self.profile("macos"), self.controller)
        calls = self.controller.method_calls
        # the launcher must go before the app counts as shown: a menu still up after Enter differs from the desktop
        self.assertEqual(calls[calls.index(unittest.mock.call.hotkey(["ENTER"])) + 1:], [
            unittest.mock.call.wait_for_change(4.0, since=unittest.mock.ANY), unittest.mock.call.wait_settled(),
            unittest.mock.call.wait_for_change(4.0, since=unittest.mock.ANY)])

    def test_wait_only_watches_the_screen(self):
        run_skill("wait", {}, self.profile("windows"), self.controller)
        self.assertEqual(self.controller.method_calls, [unittest.mock.call.wait_for_change()])

    def test_select_triple_clicks_with_no_settle_between(self):
        run_skill("select", {"x": .4, "y": .2}, self.profile("linux"), self.controller)
        self.assertEqual([(c[0], c[1][0].brief()) for c in self.controller.method_calls],
                         [("execute", "click(400,200)")] * 3)
        with self.assertRaises(ActionError):
            run_skill("select", {"x": 2, "y": .2}, self.profile("linux"), self.controller)

    def test_unsupported_has_no_input(self):
        for os in ("ios", "unknown"):
            with self.assertRaises(SkillUnavailable):
                run_skill("open_app", {"name": "Notes"}, self.profile(os), self.controller)
        with self.assertRaises(SkillUnavailable):
            run_skill("missing", {}, self.profile("macos"), self.controller)
        self.assertEqual(self.controller.method_calls, [])

    def test_invalid_names_have_no_input(self):
        for name in (None, 5, "", "  ", "Notes\n", "Notes\t"):
            with self.assertRaises(ActionError):
                run_skill("open_app", {"name": name}, self.profile("macos"), self.controller)
        self.assertEqual(self.controller.method_calls, [])

    def test_cancel_after_search_prevents_typing(self):
        self.controller.hotkey.side_effect = lambda keys: self.controller.cancel.cancel("barge-in")
        with self.assertRaises(Cancelled):
            run_skill("open_app", {"name": "Notes"}, self.profile("macos"), self.controller)
        self.controller.type_text.assert_not_called()
        self.controller.hotkey.assert_called_once_with(["GUI", "SPACE"])

    def test_cancel_after_typing_prevents_enter(self):
        self.controller.type_text.side_effect = lambda text: self.controller.cancel.cancel("disconnect")
        with self.assertRaises(Cancelled):
            run_skill("open_app", {"name": "Notes"}, self.profile("windows"), self.controller)
        self.controller.hotkey.assert_called_once_with(["GUI"])


if __name__ == "__main__":
    unittest.main()
