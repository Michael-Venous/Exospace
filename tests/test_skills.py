"""Shortcut checks without model calls or real keyboard input."""

import unittest
from unittest.mock import Mock

from brain.cancel import Cancelled, CancelToken
from brain.skills import SkillUnavailable, run_skill
from brain.types import ActionError, DeviceProfile, Rect


class SkillTests(unittest.TestCase):
    def setUp(self):
        self.controller = Mock()
        self.controller.cancel = CancelToken()

    def profile(self, os):
        return DeviceProfile(os, Rect(0, 0, 100, 100), "mouse", 0)

    def test_platform_shortcuts(self):
        for os, keys in (("macos", ["GUI", "SPACE"]), ("ios", ["GUI", "SPACE"]), ("windows", ["GUI"]),
                         ("linux", ["GUI"]), ("chromeos", ["GUI"])):
            with self.subTest(os=os):
                self.controller.reset_mock()
                run_skill("open_app", {"name": " Notes "}, self.profile(os), self.controller)
                self.assertEqual(self.controller.method_calls, [
                    unittest.mock.call.hotkey(keys),
                    unittest.mock.call.wait_settled(),
                    unittest.mock.call.type_text("Notes"),
                    unittest.mock.call.wait_settled(),
                    unittest.mock.call.hotkey(["ENTER"]),
                ])

    def test_android_picks_the_app_not_a_web_search(self):
        run_skill("open_app", {"name": "Clock"}, self.profile("android"), self.controller)
        self.assertEqual(self.controller.method_calls[-2:], [
            unittest.mock.call.hotkey(["DOWN"]), unittest.mock.call.hotkey(["ENTER"])])

    def test_unsupported_has_no_input(self):
        for os in ("bios", "unknown"):
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
        self.controller.wait_settled.side_effect = lambda: self.controller.cancel.cancel("barge-in")
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
