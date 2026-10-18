"""Model-free quick commands: what fires instantly and what must reach the model."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from brain.agent import quick
from brain.agent.executor import Plan, StepContext
from brain.agent.memory import Memory
from brain.agent.quick import QuickStart, quick_actions, spoken_commands
from brain.skills import desktop_entries, installed_apps
from brain.types import Action, DeviceProfile, Rect


def briefs(goal, os="windows"):
    return [action.brief() for action in quick_actions(goal, os)]


class QuickActionTests(unittest.TestCase):
    def setUp(self):
        for name, apps in (("INSTALLED", {"google chrome", "firefox web browser", "settings", "discord"}),
                           ("BROWSERS", {"google chrome", "firefox web browser", "safari"})):
            installed = patch.object(quick, name, frozenset(apps))
            installed.start()
            self.addCleanup(installed.stop)

    def test_unambiguous_leading_commands_fire(self):
        cases = {
            "open calculator": ["skill(open_app {'name': 'calculator'})"],
            "Open the Calculator app.": ["skill(open_app {'name': 'Calculator'})"],
            "launch VS Code": ["skill(open_app {'name': 'VS Code'})"],
            "open settings, then turn on wifi": ["skill(open_app {'name': 'settings'})"],
            "open calculator and type 2+2": ["skill(open_app {'name': 'calculator'})"],
            'open notes and type "hi"': ["skill(open_app {'name': 'notes'})", "type_text('hi')"],
            "open chrome and go to youtube.com/feed": ["skill(open_app {'name': 'chrome'})", "hotkey(CTRL+L)",
                                                       "type_text('youtube.com/feed')", "hotkey(DELETE)", "hotkey(ENTER)"],
            "open firefox, open a new tab, then visit example.org": [
                "skill(open_app {'name': 'firefox'})", "hotkey(CTRL+T)", "hotkey(CTRL+L)",
                "type_text('example.org')", "hotkey(DELETE)", "hotkey(ENTER)"],
            "Open up Chrome and go to youtube.com.": ["skill(open_app {'name': 'Chrome'})", "hotkey(CTRL+L)",
                                                      "type_text('youtube.com')", "hotkey(DELETE)", "hotkey(ENTER)"],
            "open chrome, go to youtube.com and search for horses": [
                "skill(open_app {'name': 'chrome'})", "hotkey(CTRL+L)", "type_text('youtube.com')",
                "hotkey(DELETE)", "hotkey(ENTER)"],
            "Open a spreadsheet, enter 12 in A1": ["skill(open_app {'name': 'spreadsheet'})"],
            "open a text editor and type hi": ["skill(open_app {'name': 'text editor'})"],
            "press ctrl c": ["hotkey(CTRL+C)"],
            "press ctrl-shift+t": ["hotkey(CTRL+SHIFT+T)"],
            "press page down": ["hotkey(PAGEDOWN)"],
            "press the windows key": ["hotkey(GUI)"],
            "press ctrl a then press delete": ["hotkey(CTRL+A)", "hotkey(DELETE)"],
            'type "hello world" and press enter': ["type_text('hello world')", "hotkey(ENTER)"],
            "type “hi there”, then press tab.": ["type_text('hi there')", "hotkey(TAB)"],
            "scroll down": ["scroll(5)"],
            "scroll up and click the first result": ["scroll(-5)"],  # a scroll is easily undone
            "scroll down, please": ["scroll(5)"],
            "open a new tab": ["hotkey(CTRL+T)"],
            # a site said without ".com" is left to the browser's own completion; Enter loads it, so its own search
            # box takes the search
            "Open up a new tab and go to YouTube and search for horses.": [
                "hotkey(CTRL+T)", "hotkey(CTRL+L)", "type_text('YouTube')", "hotkey(ENTER)"],
            "Go to Google Chrome, open up a new tab, and go to YouTube.": [
                "skill(open_app {'name': 'Google Chrome'})", "hotkey(CTRL+T)", "hotkey(CTRL+L)",
                "type_text('YouTube')", "hotkey(ENTER)"],
            "switch to firefox and click the second link": ["skill(open_app {'name': 'firefox'})"],
            # follows on from a request that already ran; after a launch its window is in front
            "Just did: open a new tab\npress ctrl l": ["hotkey(CTRL+L)"],
            "Just did: open chrome\ngo to wikipedia.org": ["hotkey(CTRL+L)", "type_text('wikipedia.org')",
                                                           "hotkey(DELETE)", "hotkey(ENTER)"],
            # a last search after a launch or a new tab goes in the address bar
            "Just did: open up a new tab\nsearch Genghis Khan.": ["hotkey(CTRL+L)", "type_text('Genghis Khan')",
                                                                  "hotkey(DELETE)", "hotkey(ENTER)"],
            "Open up Chrome and search up Genghis Khan.": ["skill(open_app {'name': 'Chrome'})", "hotkey(CTRL+L)",
                                                           "type_text('Genghis Khan')", "hotkey(DELETE)", "hotkey(ENTER)"],
            "open a new tab and search for cats": ["hotkey(CTRL+T)", "hotkey(CTRL+L)", "type_text('cats')",
                                                   "hotkey(DELETE)", "hotkey(ENTER)"],
            # where to search, or more after it, is the model's
            "Open up Chrome and search up Genghis Khan. On YouTube.": ["skill(open_app {'name': 'Chrome'})"],
            "open chrome and search up youtube and search up cats": ["skill(open_app {'name': 'chrome'})"],
            "open chrome and search cats on youtube": ["skill(open_app {'name': 'chrome'})"],
        }
        for goal, expected in cases.items():
            with self.subTest(goal=goal):
                self.assertEqual(briefs(goal), expected)

    def test_vague_or_compound_goals_go_to_the_model(self):
        for goal in ("open a document", "open an email", "open up the file menu", "open file.txt", "open youtube.com",
                     "open the file menu", "open my documents", "open it", "press enter twice",
                     "press the button", "press enter to submit", "press and hold power",
                     "press a b", "type hello", "scroll down to the bottom", "Create the event",
                     "open four word app name here",
                     "open calculator and type 2+2\nUser correction: I meant calendar",
                     "Just did: press ctrl l\ngo to wikipedia.org", "x\nJust did: open chrome\npress enter",
                     "search cats", "Just did: press ctrl l\nsearch cats", "Just did: open chrome and go to youtube.com\nsearch cats",
                     # half a combination, or a qualified command, must not act alone
                     "press control and c", "press windows and r", "press alt and f4",
                     "press shift and press enter", "press control and then press s",
                     "press delete, but only on the second email", "press enter, not yet",
                     'type "cats", after clicking the search box',
                     # things on the screen, not apps
                     "open the second result", "open the attachment", "open them", "open history",
                     "open his profile"):
            with self.subTest(goal=goal):
                self.assertEqual(quick_actions(goal, "windows"), [])

    def test_open_app_only_where_the_os_has_a_launcher(self):
        self.assertTrue(quick_actions("open calculator", "linux"))
        self.assertEqual(quick_actions("open camera", "android"), [])
        self.assertEqual(quick_actions("open calculator", "unknown"), [])
        self.assertEqual(briefs("press F10", "unknown"), ["hotkey(F10)"])
        self.assertEqual(briefs("press home", "android"), ["hotkey(GUI+H)"])

    def test_go_to_an_app_opens_it_only_when_it_is_installed(self):
        self.assertEqual(briefs("go to firefox"), ["skill(open_app {'name': 'firefox'})"])  # "Firefox Web Browser"
        self.assertEqual(briefs("go back to the settings"), ["skill(open_app {'name': 'settings'})"])
        self.assertEqual(briefs("go to the chrome settings"), [])
        self.assertEqual(briefs("go to fire"), [])  # whole words only
        with patch.object(quick, "INSTALLED", frozenset()):  # a target that isn't this machine
            self.assertEqual(briefs("go to firefox"), [])

    def test_installed_apps_are_the_launcher_entries_shown(self):
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as system:
            for root, name, body in ((home, "a.desktop", "Name=Firefox Web Browser\nCategories=Network;WebBrowser;\n"
                                                         "[Desktop Action new]\nName=New Window"),
                                     (system, "sub/b.desktop", "Name=Discord\nStartupWMClass=discord\nCategories=Network;"),
                                     (system, "c.desktop", "Name=Helper\nNoDisplay=true")):
                path = Path(root, "applications", name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("[Desktop Entry]\n" + body + "\n")
            with patch.dict(os.environ, {"XDG_DATA_HOME": home, "XDG_DATA_DIRS": system}):
                self.assertEqual(installed_apps(), {"firefox web browser", "discord"})
                self.assertEqual(installed_apps("WebBrowser"), {"firefox web browser"})
                self.assertIn(("discord", {"b", "discord"}, {"Network"}), desktop_entries())  # the ids its windows carry

    def test_browser_keys_only_follow_a_launch_on_a_computer(self):
        self.assertEqual(quick_actions("go to youtube.com", "windows"), [])  # whatever is in front gets them
        self.assertEqual(quick_actions("go to youtube", "windows"), [])
        self.assertEqual(briefs("open chrome and go to youtubers", "windows")[1:], [
            "hotkey(CTRL+L)", "type_text('youtubers')", "hotkey(ENTER)"])  # the browser searches what isn't a site
        self.assertEqual(briefs("open chrome and switch to youtubers", "windows")[1:], [])
        self.assertEqual(quick_actions("go to the chrome settings", "windows"), [])
        self.assertEqual(briefs("open safari and go to apple.com", "macos")[1:], [
            "hotkey(GUI+L)", "type_text('apple.com')", "hotkey(DELETE)", "hotkey(ENTER)"])
        self.assertEqual(quick_actions("open chrome and go to youtube.com", "android"), [])
        self.assertEqual(quick_actions("open a new tab", "android"), [])
        # Another app's launch puts it in front: the address would be typed into its message box and sent.
        self.assertEqual(briefs("open discord and go to swampax", "linux"), ["skill(open_app {'name': 'discord'})"])
        self.assertEqual(briefs("open chrome, open discord and go to x.com", "linux")[1:], ["skill(open_app {'name': 'discord'})"])
        self.assertEqual(quick_actions("Just did: open discord\ngo to swampax", "linux"), [])
        with patch.object(quick, "BROWSERS", frozenset()):  # a target that isn't this machine: the model checks
            self.assertEqual(briefs("open chrome and go to youtube.com", "linux"), ["skill(open_app {'name': 'chrome'})"])


class SpokenCommandTests(unittest.TestCase):
    def test_a_command_is_finished_once_more_speech_follows_it(self):
        cases = {
            "open a new tab": [],  # "... in incognito" may still come
            "open a new tab in incognito": [],
            "open a new tab and": [("open a new tab", "")],
            "Open a new tab, and search Genghis Khan.": [("Open a new tab", "search Genghis Khan.")],
            "Open up a new tab and search up Genghis Khan.": [("Open up a new tab", "search up Genghis Khan.")],
            "open chrome and go to wikipedia.org and": [("open chrome", "go to wikipedia.org and")],
            "press ctrl c then press escape.": [("press ctrl c", "press escape."), ("press escape", "")],
            "open calculator and type 2+2": [("open calculator", "type 2+2")],
            # any other step is finished once the next one starts
            "open up chrome and go to youtube, search for embedded systems videos and click the third reel": [
                ("open up chrome", "go to youtube, search for embedded systems videos and click the third reel"),
                ("go to youtube", "search for embedded systems videos and click the third reel"),
                ("search for embedded systems videos", "click the third reel")],
            "click the red button and press enter": [("click the red button", "press enter")],
            # until then it, a scroll and half a combination wait; "and" inside a step doesn't end it
            "click the red button and": [], "open the second result and": [], "scroll down and click it": [],
            "press control and c": [], "search for salt and pepper and": [],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(spoken_commands(text), expected)


class QuickStartTests(unittest.TestCase):
    def setUp(self):
        self.inner = Mock()
        self.inner.next_actions.return_value = Plan("model", [Action("done")], [])
        self.executor = QuickStart(self.inner)
        self.profile = DeviceProfile("macos", Rect(0, 0, 10, 10), "mouse", 0)

    def ctx(self, goal, step):
        return StepContext(goal, self.profile, Mock(spec=Memory), [], step, 30)

    def test_first_step_skips_the_model(self):
        plan = self.executor.next_actions(b"jpeg", self.ctx("open notes", 1))
        self.assertEqual(plan.actions, [Action("skill", name="open_app", args={"name": "notes"})])
        self.inner.next_actions.assert_not_called()

    def test_later_steps_and_other_goals_use_the_model(self):
        for goal, step in (("open notes", 2), ("draft an invite", 1)):
            self.assertEqual(self.executor.next_actions(b"jpeg", self.ctx(goal, step)).reasoning, "model")
        self.assertEqual(self.inner.next_actions.call_count, 2)

    def test_delegates_everything_else(self):
        self.inner.classify_os.return_value = "android"
        self.inner.image_grid = False
        self.assertEqual(self.executor.classify_os(b"jpeg"), "android")
        self.assertIs(self.executor.image_grid, False)


if __name__ == "__main__":
    unittest.main()
