"""Learned cache: screen fingerprints, matching, forgetting, persistence and race replay."""

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from brain.agent.executor import Plan, StepContext
from brain.agent.learned import LearnedCache, fingerprint, normalize
from brain.agent.memory import Memory
from brain.agent.race import RaceExecutor
from brain.agent.triage import Triage
from brain.cancel import CancelToken
from brain.types import Action, DeviceProfile, Rect

CLICK = Triage("act", (.3, .4), False)
OPEN = Triage("keys", None, False, skill="open_app", text="spotify")


def screen(seed: int, mark: bool = False) -> bytes:
    """A screen-like image; `mark` adds a small change such as a clock tick or a cursor."""
    image = np.random.default_rng(seed).integers(0, 256, (90, 160, 3), dtype=np.uint8)
    image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_NEAREST)
    if mark:
        cv2.rectangle(image, (1200, 690), (1270, 715), (255, 255, 255), -1)
    return cv2.imencode(".jpg", image)[1].tobytes()


class LearnedCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "learned.json"
        self.cache = LearnedCache(self.path)

    def test_fingerprint_tolerates_small_changes_but_not_other_screens(self):
        base = fingerprint(screen(1))
        self.assertLessEqual(bin(base ^ fingerprint(screen(1, mark=True))).count("1"), 10)
        self.assertGreater(bin(base ^ fingerprint(screen(2))).count("1"), 30)

    def test_a_stored_click_replays_on_the_same_request_and_screen_only(self):
        self.cache.store("windows", "Open John's email!", screen(1), CLICK)
        self.assertEqual(self.cache.lookup("windows", "open john's email", screen(1, mark=True)), CLICK)
        self.assertIsNone(self.cache.lookup("windows", "open john's email", screen(2)))  # other screen
        self.assertIsNone(self.cache.lookup("macos", "open john's email", screen(1)))  # other OS
        self.assertIsNone(self.cache.lookup("windows", "open the email", screen(1)))  # other request

    def test_screen_independent_skills_match_any_screen(self):
        self.cache.store("windows", "open spotify", screen(1), OPEN)
        self.assertEqual(self.cache.lookup("windows", "open spotify", screen(2)), OPEN)

    def test_irreversible_and_planner_decisions_are_never_stored(self):
        self.cache.store("windows", "delete it", screen(1), Triage("act", (.1, .1), True))
        self.cache.store("windows", "write an essay", screen(1), Triage("plan", None, False))
        self.assertEqual(self.cache.entries, [])

    def test_forget_and_persistence(self):
        self.cache.store("windows", "open spotify", screen(1), OPEN)
        self.assertEqual(LearnedCache(self.path).lookup("windows", "open spotify", screen(1)), OPEN)
        self.cache.forget("windows", "open spotify", screen(1))
        self.assertIsNone(LearnedCache(self.path).lookup("windows", "open spotify", screen(1)))

    def test_oldest_entries_are_dropped_beyond_the_cap(self):
        cache = LearnedCache(self.path, max_entries=2)
        for app in ("a", "b", "c"):
            cache.store("windows", f"open {app}", screen(1), Triage("keys", None, False, skill="open_app", text=app))
        self.assertEqual([e["utterance"] for e in cache.entries], ["open b", "open c"])

    def test_corrupt_file_starts_empty(self):
        self.path.write_text("{not json")
        self.assertEqual(LearnedCache(self.path).entries, [])

    def test_normalize_ignores_case_punctuation_and_spacing(self):
        self.assertEqual(normalize("  Go to YouTube.com, please! "), "go to youtube.com please")


class RaceReplayTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        race = object.__new__(RaceExecutor)
        race.planner, race.jev, race.cancel, race.solo, race._keys = Mock(), Mock(), CancelToken(), True, []
        race.learned = LearnedCache(Path(temp.name) / "learned.json")
        race._pool = ThreadPoolExecutor(max_workers=1)
        race._plan, race._key, race.mode, race._point, race._clicked, race._pending = None, None, "", None, False, None
        race._gesture, race._utterance, race._triage_step, race._approved_goal = "click", "", 0, None
        self.addCleanup(race._pool.shutdown)
        race.planner.next_actions.return_value = Plan("planned", [Action("hotkey", keys=("ENTER",))], [])
        self.race, self.memory = race, Memory(Path(temp.name) / "m.json")

    def ctx(self, goal, run, note=""):
        return StepContext(goal, DeviceProfile("windows", Rect(0, 0, 1280, 720), "mouse", 1), self.memory,
                           [], 1, 10, note=note, run_id=run)

    def run_once(self, run, triaged, note=""):
        with patch("brain.agent.race.triage", return_value=triaged) as triage:
            first = self.race.next_actions(screen(1), self.ctx("open john's email", run))
            self.race.acknowledge(True)
            second = self.race.next_actions(screen(1), self.ctx("open john's email", run, note))
        return first, second, triage

    def test_second_time_replays_without_jev_or_planner(self):
        self.run_once("a", CLICK)
        self.race.planner.next_actions.reset_mock()
        first, second, triage = self.run_once("b", CLICK)
        triage.assert_not_called()
        self.race.planner.next_actions.assert_not_called()
        self.assertTrue(first.reasoning.startswith("Remembered: "))
        self.assertEqual((first.actions[0].type, first.actions[0].x), ("click", .3))
        self.assertEqual(second.actions[0].type, "done")

    def test_a_replay_with_no_effect_is_forgotten_and_the_planner_recovers(self):
        self.run_once("a", CLICK)
        _, second, _ = self.run_once("b", CLICK, note="Screen unchanged; try something different.")
        self.assertEqual(second.reasoning, "planned")
        self.assertEqual(self.race.learned.entries, [])

    def test_a_failed_fresh_decision_is_not_stored(self):
        self.run_once("a", CLICK, note="Screen unchanged; try something different.")
        self.assertEqual(self.race.learned.entries, [])


if __name__ == "__main__":
    unittest.main()
