"""Replay cache: record a successful run, replay it only on the same screen, never record approvals."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np

from brain.agent.executor import Plan, StepContext
from brain.agent.memory import Memory
from brain.agent.replay import ReplayCache
from brain.types import Action, DeviceProfile, Rect


def screen(value: int) -> bytes:
    image = np.full((200, 320, 3), value, np.uint8)
    return cv2.imencode(".jpg", image)[1].tobytes()


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "replay.json"
        self.inner = Mock()
        self.inner.jev.completion.return_value = .9
        self.memory = Memory.load(Path(self.temp.name) / "memory.json")
        self.profile = DeviceProfile("linux", Rect(0, 0, 320, 200), "mouse", 1)

    def ctx(self, goal, step, run_id):
        return StepContext(goal, self.profile, self.memory, [], step, 10, run_id=run_id)

    def record(self, cache, goal="open the talk page", run_id="a"):
        self.inner.next_actions.side_effect = [Plan("click", [Action("click", x=.3, y=.2)], []),
                                               Plan("done", [Action("done", summary="Talk page open")], [])]
        cache.next_actions(screen(40), self.ctx(goal, 1, run_id))
        cache.acknowledge(True)
        cache.next_actions(screen(200), self.ctx(goal, 2, run_id))

    def test_a_repeat_on_the_same_screen_replays_without_the_models(self):
        self.record(ReplayCache(self.inner, self.path))
        cache = ReplayCache(self.inner, self.path)  # reloaded from disk
        self.inner.next_actions.reset_mock()
        first = cache.next_actions(screen(40), self.ctx("Open the Talk page!", 1, "b"))
        self.assertEqual((first.actions[0].type, first.actions[0].x), ("click", .3))
        cache.acknowledge(True)
        last = cache.next_actions(screen(200), self.ctx("Open the Talk page!", 2, "b"))
        self.assertEqual(last.actions[0].type, "done")
        self.inner.next_actions.assert_not_called()

    def test_a_different_screen_goes_to_the_executor(self):
        self.record(ReplayCache(self.inner, self.path))
        self.inner.next_actions.side_effect = None
        self.inner.next_actions.return_value = Plan("planned", [Action("hotkey", keys=("ENTER",))], [])
        cache = ReplayCache(self.inner, self.path)
        self.assertEqual(cache.next_actions(screen(120), self.ctx("open the talk page", 1, "b")).reasoning, "planned")

    def test_an_unconfirmed_replay_hands_over(self):
        self.record(ReplayCache(self.inner, self.path))
        cache = ReplayCache(self.inner, self.path)
        cache.next_actions(screen(40), self.ctx("open the talk page", 1, "b"))
        cache.acknowledge(True)
        self.inner.jev.completion.return_value = .1
        self.inner.next_actions.side_effect = None
        self.inner.next_actions.return_value = Plan("planned", [], [])
        self.assertEqual(cache.next_actions(screen(200), self.ctx("open the talk page", 2, "b")).reasoning, "planned")

    def test_approved_runs_are_never_recorded(self):
        self.record(ReplayCache(self.inner, self.path), goal="email john\nUser correction: yes")
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
