"""Model-free quick commands: what fires instantly and what must reach the model."""

import unittest
from unittest.mock import Mock

from brain.agent.executor import Plan, StepContext
from brain.agent.memory import Memory
from brain.agent.quick import QuickStart, quick_actions, spoken_commands
from brain.types import Action, DeviceProfile, Rect


def briefs(goal, os="windows"):
    return [action.brief() for action in quick_actions(goal, os)]


class QuickActionTests(unittest.TestCase):
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
            "open chrome, go to youtube.com and search for horses": ["skill(open_app {'name': 'chrome'})"],
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

    def test_browser_keys_only_follow_a_launch_on_a_computer(self):
        self.assertEqual(quick_actions("go to youtube.com", "windows"), [])  # whatever is in front gets them
        self.assertEqual(briefs("open safari and go to apple.com", "macos")[1:], [
            "hotkey(GUI+L)", "type_text('apple.com')", "hotkey(DELETE)", "hotkey(ENTER)"])
        self.assertEqual(quick_actions("open chrome and go to youtube.com", "android"), [])
        self.assertEqual(quick_actions("open a new tab", "android"), [])


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
            # model work, pointer work and half a combination wait for the pause
            "click the red button and": [], "open the second result and": [], "scroll down and": [],
            "press control and c": [],
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
