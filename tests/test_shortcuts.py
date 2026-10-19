"""Keyboard first: one-command requests become that OS's shortcut, with no model aiming."""

import unittest

from brain.agent.shortcuts import shortcut_for


class ShortcutTests(unittest.TestCase):
    def test_closing_an_app_uses_each_os_own_keys(self):
        self.assertEqual(shortcut_for("Close this app.", "macos"), ("close the app", ("GUI", "Q")))
        self.assertEqual(shortcut_for("quit the application", "windows"), ("close the app", ("ALT", "F4")))
        self.assertEqual(shortcut_for("close the app", "ios"), ("close the app", ("GUI", "H")))  # iPhone: home

    def test_common_commands(self):
        for said, os, keys in (("new tab", "linux", ("CTRL", "T")), ("Go back.", "windows", ("ALT", "LEFT")),
                               ("refresh the page", "macos", ("GUI", "R")), ("close this tab", "chromeos", ("CTRL", "W")),
                               ("undo that", "linux", ("CTRL", "Z")), ("go home", "android", ("GUI", "H")),
                               ("new message", "ios", ("GUI", "N")), ("can you save it please", "macos", ("GUI", "S"))):
            with self.subTest(said=said):
                self.assertEqual(shortcut_for(said, os)[1], keys)

    def test_only_a_whole_one_command_request_matches(self):
        for said in ("close this app and open Safari", "go back to the inbox", "save the file as report.pdf",
                     "open a new tab and go to youtube", "close the cookie banner"):
            with self.subTest(said=said):
                self.assertIsNone(shortcut_for(said, "macos"))

    def test_no_shortcut_where_the_os_has_none(self):
        self.assertIsNone(shortcut_for("refresh the page", "ios"))
        self.assertIsNone(shortcut_for("go home", "windows"))  # a computer's "home" is ambiguous
        self.assertIsNone(shortcut_for("close this app", "unknown"))


if __name__ == "__main__":
    unittest.main()
