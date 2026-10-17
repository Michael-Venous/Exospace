"""Model-free quick commands: what fires instantly and what must reach the model."""

import unittest
from unittest.mock import Mock

from brain.agent.executor import Plan, StepContext
from brain.agent.memory import Memory
from brain.agent.quick import QuickStart, quick_actions
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
            "press ctrl c": ["hotkey(CTRL+C)"],
            "press ctrl-shift+t": ["hotkey(CTRL+SHIFT+T)"],
            "press page down": ["hotkey(PAGEDOWN)"],
            "press the windows key": ["hotkey(GUI)"],
            "press ctrl a then press delete": ["hotkey(CTRL+A)", "hotkey(DELETE)"],
            'type "hello world" and press enter': ["type_text('hello world')", "hotkey(ENTER)"],
            "type “hi there”, then press tab.": ["type_text('hi there')", "hotkey(TAB)"],
            "scroll down": ["scroll(5)"],
        }
        for goal, expected in cases.items():
            with self.subTest(goal=goal):
                self.assertEqual(briefs(goal), expected)

    def test_vague_or_compound_goals_go_to_the_model(self):
        for goal in ("open a new tab", "open up spotify", "open file.txt", "open youtube.com",
                     "open the file menu", "open my documents", "open it", "press enter twice",
                     "press the button", "press enter to submit", "press and hold power",
                     "press a b", "type hello", "scroll down to the bottom", "Create the event",
                     "open four word app name here",
                     "open calculator and type 2+2\nUser correction: I meant calendar",
                     # half a combination, or a qualified command, must not act alone
                     "press control and c", "press windows and r", "press alt and f4",
                     "press delete, but only on the second email", "press enter, not yet",
                     'type "cats", after clicking the search box', "scroll up and click the first result",
                     # things on the screen, not apps
                     "open the second result", "open the attachment", "open them", "open history",
                     "open his profile"):
            with self.subTest(goal=goal):
                self.assertEqual(quick_actions(goal, "windows"), [])

    def test_open_app_only_where_the_os_has_a_launcher(self):
        self.assertTrue(quick_actions("open camera", "android"))
        self.assertEqual(quick_actions("open calculator", "bios"), [])
        self.assertEqual(quick_actions("open calculator", "unknown"), [])
        self.assertEqual(briefs("press F10", "bios"), ["hotkey(F10)"])
        self.assertEqual(quick_actions("scroll down", "bios"), [])
        self.assertEqual(briefs("press home", "android"), ["hotkey(GUI+H)"])


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
        self.inner.classify_os.return_value = "bios"
        self.inner.image_grid = False
        self.assertEqual(self.executor.classify_os(b"jpeg"), "bios")
        self.assertIs(self.executor.image_grid, False)


if __name__ == "__main__":
    unittest.main()
