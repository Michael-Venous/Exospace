"""Deep mode: race until it is stuck, then the deep agent for that request only, behind the side-effect guard."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from brain.agent.deep import DeepFallback
from brain.agent.executor import Plan, StepContext
from brain.agent.memory import Memory
from brain.config import Config
from brain.types import Action, DeviceProfile, Rect

STUCK = Plan("Plan could not be grounded", [], ["Visual checks: ..."])


class DeepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        cfg = replace(Config.load(), deep_model="deep-model", deep_after=2)
        self.inner = Mock()
        self.inner.jev.evaluate.return_value = ({"side_effects": {"noul": .05}}, None, None)
        with patch("brain.agent.deep.ModelExecutor") as model:
            self.agent = DeepFallback(self.inner, cfg, Mock())
        self.deep = model.return_value
        self.deep.next_actions.return_value = Plan("deep", [Action("hotkey", keys=("TAB",))], [])
        self.memory = Memory.load(Path(self.temp.name) / "m.json")

    def ctx(self, step, run="a", goal="make me a house in blender"):
        profile = DeviceProfile("linux", Rect(0, 0, 10, 10), "mouse", 1)
        return StepContext(goal, profile, self.memory, [], step, 60, run_id=run)

    def test_race_keeps_requests_it_handles(self):
        self.inner.next_actions.return_value = Plan("Stage 1/1", [Action("click", x=.1, y=.1)], [])
        for step in (1, 2, 3):
            self.assertEqual(self.agent.next_actions(b"img", self.ctx(step)).reasoning, "Stage 1/1")
        self.deep.next_actions.assert_not_called()

    def test_two_stuck_steps_switch_this_request_to_deep_mode(self):
        self.inner.next_actions.return_value = STUCK
        self.assertEqual(self.agent.next_actions(b"img", self.ctx(1)).reasoning, STUCK.reasoning)
        self.assertEqual(self.agent.next_actions(b"img", self.ctx(2)).reasoning, "deep")
        self.assertEqual(self.agent.next_actions(b"img", self.ctx(3)).reasoning, "deep")
        self.inner.next_actions.return_value = Plan("fast", [Action("click", x=.1, y=.1)], [])
        self.assertEqual(self.agent.next_actions(b"img", self.ctx(1, run="b")).reasoning, "fast")  # next request

    def test_a_deep_model_error_hands_that_step_to_race(self):
        self.inner.next_actions.return_value = STUCK
        for step in (1, 2):
            self.agent.next_actions(b"img", self.ctx(step))
        self.deep.next_actions.side_effect = RuntimeError("504 DEADLINE_EXCEEDED")
        self.inner.next_actions.return_value = Plan("Stage 1/1", [Action("click", x=.1, y=.1)], [])
        self.assertEqual(self.agent.next_actions(b"img", self.ctx(3)).reasoning, "Stage 1/1")
        self.agent.acknowledge(True)
        self.inner.acknowledge.assert_called_with(True)
        self.assertTrue(self.agent.active)  # the next step tries the deep model again

    def test_deep_actions_with_side_effects_are_blocked(self):
        self.inner.next_actions.return_value = STUCK
        self.inner.jev.evaluate.return_value = ({"side_effects": {"noul": .9}}, None, None)
        self.agent.next_actions(b"img", self.ctx(1))
        plan = self.agent.next_actions(b"img", self.ctx(2))
        self.assertEqual(plan.actions, [])
        self.assertIn("Blocked", plan.rejected[0])


    def test_alternating_progress_and_vetoes_still_escalates(self):
        ok = Plan("ok", [Action("click", x=.1, y=.1)], [])
        self.inner.next_actions.side_effect = [STUCK, ok, STUCK, ok]
        reasons = [self.agent.next_actions(b"img", self.ctx(step)).reasoning for step in (1, 2, 3, 4)]
        self.assertEqual(reasons, [STUCK.reasoning, "ok", "deep", "deep"])  # 2 of the last steps (DEEP_AFTER=2 here)

if __name__ == "__main__":
    unittest.main()
