"""Jev solo skills: span options, gating, keyboard actions, and the race executor's keys mode."""

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from brain.agent.executor import Plan, StepContext
from brain.agent.memory import Memory
from brain.agent.race import RaceExecutor
from brain.agent.solo import solo_actions, spans
from brain.agent.triage import INTENT_CRITERIA, Triage, triage, triage_questions
from brain.cancel import CancelToken
from brain.types import Action, DeviceProfile, Rect

SIZE = ((400, 300), (400, 300))


def jpeg() -> bytes:
    return cv2.imencode(".jpg", np.full((300, 400, 3), 200, np.uint8))[1].tobytes()


def answers(intent, intent_p=.95, span=None, span_p=.9, browser=.95, irreversible=.02):
    return ({"intent": {"choice": intent, "probabilities": {intent: intent_p}},
             "present": {"noul": .1}, "unique": {"noul": .1}, "gesture": {"choice": "click"},
             "irreversible": {"noul": irreversible}, "target": {"box": {}},
             "name_span": {"choice": span or "none_here", "probabilities": {span or "none_here": span_p}},
             "browser_front": {"noul": browser}}, *SIZE)


def span_id(utterance, text):
    return next(k for k, v in spans(utterance).items() if v == text)


class SpanTests(unittest.TestCase):
    def test_spans_are_exact_slices_without_edge_punctuation(self):
        options = spans("Go to google.com, please.")
        self.assertIn("google.com", options.values())
        self.assertIn("please", options.values())
        self.assertNotIn("google.com,", options.values())

    def test_multiword_names_are_offered(self):
        self.assertIn("visual studio code", spans("open visual studio code").values())

    def test_long_utterances_offer_no_solo_skills(self):
        self.assertEqual(spans("word " * 41), {})


class TriageSoloTests(unittest.TestCase):
    def setUp(self):
        self.jev = Mock()

    def test_off_by_default_asks_exactly_the_original_questions(self):
        self.jev.evaluate.return_value = answers("plan")
        triage(self.jev, jpeg(), "open spotify", CancelToken())
        questions = self.jev.evaluate.call_args.args[1]
        self.assertEqual(set(questions["intent"]["criteria"]), set(INTENT_CRITERIA))
        self.assertNotIn("name_span", questions)

    def test_confident_open_app_becomes_keys_with_the_users_words(self):
        utterance = "open spotify please"
        self.jev.evaluate.return_value = answers("open_app", span=span_id(utterance, "spotify"))
        result = triage(self.jev, jpeg(), utterance, CancelToken(), solo=True)
        self.assertEqual((result.mode, result.skill, result.text), ("keys", "open_app", "spotify"))
        self.jev.evaluate.assert_called_once()  # one Jev call, no zoom

    def test_unsure_intent_or_words_go_to_the_planner(self):
        utterance = "open spotify"
        for kwargs in ({"intent_p": .5, "span": span_id(utterance, "spotify")},
                       {"span": span_id(utterance, "spotify"), "span_p": .3},
                       {"span": None}):
            self.jev.evaluate.return_value = answers("open_app", **kwargs)
            self.assertEqual(triage(self.jev, jpeg(), utterance, CancelToken(), solo=True).mode, "plan", kwargs)

    def test_go_to_site_needs_a_browser_in_front(self):
        utterance = "go to youtube"
        self.jev.evaluate.return_value = answers("go_to_site", span=span_id(utterance, "youtube"), browser=.2)
        self.assertEqual(triage(self.jev, jpeg(), utterance, CancelToken(), solo=True).mode, "plan")

    def test_solo_questions_join_only_when_on(self):
        self.assertIn("go_to_site", triage_questions(spans("go to youtube"))["intent"]["criteria"])
        self.assertNotIn("go_to_site", triage_questions({})["intent"]["criteria"])


class SoloActionTests(unittest.TestCase):
    def test_open_app_uses_the_launcher_skill(self):
        self.assertEqual(solo_actions("open_app", "spotify", "windows"),
                         [Action("skill", name="open_app", args={"name": "spotify"})])

    def test_go_to_site_uses_each_os_address_bar_key(self):
        mac = solo_actions("go_to_site", "youtube", "macos")
        self.assertEqual([a.brief() for a in mac], ["hotkey(GUI+L)", "type_text('youtube')", "hotkey(ENTER)"])
        self.assertEqual(solo_actions("go_to_site", "youtube", "linux")[0].keys, ("CTRL", "L"))

    def test_no_keyboard_route_or_unsafe_text_returns_none(self):
        self.assertIsNone(solo_actions("go_to_site", "youtube", "android"))
        self.assertIsNone(solo_actions("open_app", "evil\x1b[", "windows"))
        self.assertIsNone(solo_actions("open_app", "", "windows"))


class RaceKeysTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        race = object.__new__(RaceExecutor)
        race.planner, race.jev, race.cancel, race.solo, race._keys = Mock(), Mock(), CancelToken(), True, []
        race._pool = ThreadPoolExecutor(max_workers=1)
        race._plan, race._key, race.mode, race._point, race._clicked, race._pending = None, None, "", None, False, None
        race._gesture, race._utterance, race._triage_step, race._approved_goal = "click", "", 0, None
        self.addCleanup(race._pool.shutdown)
        race.planner.next_actions.return_value = Plan("planned", [Action("hotkey", keys=("ENTER",))], [])
        self.race, self.memory = race, Memory(Path(temp.name) / "m.json")

    def ctx(self, goal, os="windows", note="", step=1):
        return StepContext(goal, DeviceProfile(os, Rect(0, 0, 400, 300), "mouse", 1), self.memory,
                           [], step, 10, note=note, run_id="a")

    def test_keys_run_without_waiting_for_the_plan_then_finish(self):
        with patch("brain.agent.race.triage", return_value=Triage("keys", None, False, skill="open_app",
                                                                  text="spotify")) as triaged:
            first = self.race.next_actions(b"img", self.ctx("open spotify"))
        self.assertTrue(triaged.call_args.kwargs["solo"])
        self.assertEqual(first.actions, [Action("skill", name="open_app", args={"name": "spotify"})])
        self.race.acknowledge(True)
        self.assertEqual(self.race.next_actions(b"img", self.ctx("open spotify")).actions[0].type, "done")

    def test_no_visible_effect_hands_over_to_the_planner(self):
        with patch("brain.agent.race.triage", return_value=Triage("keys", None, False, skill="open_app",
                                                                  text="spotify")):
            self.race.next_actions(b"img", self.ctx("open spotify"))
        self.race.acknowledge(True)
        plan = self.race.next_actions(b"img", self.ctx("open spotify", note="Screen unchanged; try something"))
        self.assertEqual(plan.reasoning, "planned")

    def test_os_without_a_keyboard_route_uses_the_plan(self):
        with patch("brain.agent.race.triage", return_value=Triage("keys", None, False, skill="go_to_site",
                                                                  text="youtube")):
            plan = self.race.next_actions(b"img", self.ctx("go to youtube", os="android"))
        self.assertEqual((self.race.mode, plan.reasoning), ("plan", "planned"))


if __name__ == "__main__":
    unittest.main()
