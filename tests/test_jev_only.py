"""The Jev-first planner that falls back to Flash (Jev's own step is tested in test_race.JevOnlyTests)."""

import unittest
from unittest.mock import Mock

from brain.agent.executor import Plan, StepContext
from brain.agent.jev_first import JevFirstPlanner
from brain.agent.jev_only import FAILED
from brain.types import Action, DeviceProfile, Rect


class JevFirstTests(unittest.TestCase):
    def setUp(self):
        self.planner = object.__new__(JevFirstPlanner)
        self.planner.flash, self.planner.steps, self.planner._flash_run = Mock(), Mock(), None
        self.planner.flash.next_actions.return_value = Plan("Stage 1/1: planned", [Action("hotkey", keys=("TAB",))], [])

    def ctx(self, run):
        return StepContext("summarize this page", DeviceProfile("linux", Rect(0, 0, 4, 3), "mouse", 1), Mock(), [], 1,
                           10, run_id=run)

    def test_a_recovery_note_hands_the_run_to_flash(self):
        ctx = StepContext("open calculadora", DeviceProfile("linux", Rect(0, 0, 4, 3), "mouse", 1), Mock(), [], 2,
                          10, note="the launch landed elsewhere", run_id="a")
        self.assertEqual(self.planner.next_actions(b"img", ctx).reasoning, "Stage 1/1: planned")
        self.planner.steps.next_actions.assert_not_called()

    def test_jev_steps_need_no_flash_call(self):
        self.planner.steps.next_actions.return_value = Plan("Jev next step: click", [Action("click", x=.1, y=.1)], [])
        self.assertEqual(self.planner.next_actions(b"img", self.ctx("a")).reasoning, "Jev next step: click")
        self.planner.flash.next_actions.assert_not_called()

    def test_a_step_jev_cant_take_hands_the_run_to_flash(self):
        self.planner.steps.next_actions.return_value = Plan(FAILED, [Action("fail", reason="stuck")], [])
        self.assertEqual(self.planner.next_actions(b"img", self.ctx("a")).reasoning, "Stage 1/1: planned")
        self.planner.next_actions(b"img", self.ctx("a"))
        self.assertEqual(self.planner.steps.next_actions.call_count, 1)  # the rest of that run stays on Flash
        self.planner.next_actions(b"img", self.ctx("b"))
        self.assertEqual(self.planner.steps.next_actions.call_count, 2)  # a new request starts with Jev again


if __name__ == "__main__":
    unittest.main()
