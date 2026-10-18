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
from brain.agent.solo import first_ends, rest_after, solo_actions, spans
from brain.agent.triage import INTENT_CRITERIA, Triage, triage, triage_questions
from brain.cancel import CancelToken
from brain.types import Action, DeviceProfile, Rect

SIZE = ((400, 300), (400, 300))


def jpeg() -> bytes:
    return cv2.imencode(".jpg", np.full((300, 400, 3), 200, np.uint8))[1].tobytes()


BOX = {"box": {"left": .2, "right": .4, "top": .2, "bottom": .4}, "confidence": .9}


def choice(value, p=.9):
    return {"choice": value, "probabilities": {value: p}}


def answers(intent, intent_p=.95, span=None, span_p=.9, browser=.95, irreversible=.02, key="enter", key_p=.9,
            direction="down", direction_p=.9, field=.1, first_end=None, first_end_p=.9):
    return ({"intent": choice(intent, intent_p), "present": {"noul": field}, "unique": {"noul": field},
             "gesture": {"choice": "click"}, "irreversible": {"noul": irreversible},
             "target": BOX if field > .5 else {"box": {}}, "value_span": choice(span or "none_here", span_p),
             "browser_front": {"noul": browser}, "key": choice(key, key_p),
             "scroll_direction": choice(direction, direction_p),
             **({"first_end": choice(first_end, first_end_p)} if first_end else {})}, *SIZE)


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

    def test_the_rest_of_a_chain_is_the_words_after_the_first_instruction(self):
        utterance = "open chrome, then go to youtube."
        ends = first_ends(utterance)
        self.assertEqual(rest_after(utterance, ends["e1"]), "then go to youtube")
        self.assertEqual(rest_after(utterance, ends[max(ends, key=ends.get)]), "")

    def test_long_utterances_offer_no_solo_skills(self):
        self.assertEqual(spans("word " * 31), {})

    def test_span_options_stay_under_jevs_option_limit(self):
        self.assertLessEqual(len(spans("word " * 30)) + 1, 254)


class TriageSoloTests(unittest.TestCase):
    def setUp(self):
        self.jev = Mock()

    def test_off_by_default_asks_exactly_the_original_questions(self):
        self.jev.evaluate.return_value = answers("plan")
        triage(self.jev, jpeg(), "open spotify", CancelToken())
        questions = self.jev.evaluate.call_args.args[1]
        self.assertEqual(set(questions["intent"]["criteria"]), set(INTENT_CRITERIA))
        self.assertNotIn("value_span", questions)

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

    def test_type_into_needs_one_visible_field_and_is_zoom_refined(self):
        utterance = "type hello world in the search box"
        zoom = ({"present": {"noul": .9}, "target": {"box": {"left": .4, "right": .6, "top": .4, "bottom": .6}}},
                *SIZE)
        self.jev.evaluate.side_effect = [answers("type_into", span=span_id(utterance, "hello world"), field=.95), zoom]
        result = triage(self.jev, jpeg(), utterance, CancelToken(), solo=True)
        self.assertEqual((result.mode, result.skill, result.text), ("keys", "type_into", "hello world"))
        self.assertIsNotNone(result.point)
        self.jev.evaluate.reset_mock(side_effect=True)
        self.jev.evaluate.return_value = answers("type_into", span=span_id(utterance, "hello world"), field=.3)
        self.assertEqual(triage(self.jev, jpeg(), utterance, CancelToken(), solo=True).mode, "plan")

    def test_press_key_and_scroll_use_their_closed_choices(self):
        self.jev.evaluate.return_value = answers("press_key", key="undo")
        self.assertEqual(triage(self.jev, jpeg(), "undo that", CancelToken(), solo=True).text, "undo")
        self.jev.evaluate.return_value = answers("scroll", direction="up")
        self.assertEqual(triage(self.jev, jpeg(), "scroll up", CancelToken(), solo=True).text, "up")
        self.jev.evaluate.return_value = answers("press_key", key="enter", key_p=.4)
        self.assertEqual(triage(self.jev, jpeg(), "hit it", CancelToken(), solo=True).mode, "plan")

    def test_a_chained_request_runs_the_first_instruction_and_keeps_the_rest(self):
        utterance = "open chrome and go to youtube"
        self.jev.evaluate.return_value = answers("open_app", span=span_id(utterance, "chrome"), first_end="e1")
        result = triage(self.jev, jpeg(), utterance, CancelToken(), solo=True)
        self.assertEqual((result.mode, result.text, result.rest), ("keys", "chrome", "and go to youtube"))

    def test_an_unsure_split_or_words_from_a_later_instruction_go_to_the_planner(self):
        utterance = "open chrome and go to youtube"
        self.jev.evaluate.return_value = answers("open_app", span=span_id(utterance, "chrome"), first_end="e1",
                                                 first_end_p=.3)
        self.assertEqual(triage(self.jev, jpeg(), utterance, CancelToken(), solo=True).mode, "plan")
        self.jev.evaluate.return_value = answers("open_app", span=span_id(utterance, "youtube"), first_end="e1")
        self.assertEqual(triage(self.jev, jpeg(), utterance, CancelToken(), solo=True).mode, "plan")

    def test_action_questions_judge_the_first_instruction_but_safety_reads_it_all(self):
        questions = triage_questions(spans("open chrome and delete my files"), {}, "open chrome and delete my files")
        self.assertIn("FIRST instruction", questions["intent"]["instructions"])
        self.assertNotIn("FIRST instruction", questions["irreversible"]["instructions"])

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

    def test_find_type_press_and_scroll_actions(self):
        self.assertEqual([a.brief() for a in solo_actions("find_on_page", "pricing", "windows")],
                         ["hotkey(CTRL+F)", "type_text('pricing')", "hotkey(ENTER)"])
        typed = solo_actions("type_into", "hello", "windows", (.5, .25))
        self.assertEqual([(a.type, a.x, a.text) for a in typed], [("click", .5, ""), ("type_text", None, "hello")])
        self.assertIsNone(solo_actions("type_into", "hello", "windows"))  # no field point
        self.assertEqual(solo_actions("press_key", "undo", "macos")[0].keys, ("GUI", "Z"))
        self.assertEqual(solo_actions("press_key", "back", "windows")[0].keys, ("ALT", "LEFT"))
        self.assertEqual(solo_actions("press_key", "enter", "android")[0].keys, ("ENTER",))
        self.assertIsNone(solo_actions("press_key", "undo", "android"))
        self.assertEqual(solo_actions("scroll", "down", "linux")[0].dy, 5)
        self.assertEqual(solo_actions("scroll", "up", "macos")[0].dy, -5)

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

    def test_a_chain_triages_the_rest_on_the_next_screen_then_finishes(self):
        first = Triage("keys", None, False, skill="open_app", text="chrome", rest="and go to youtube")
        second = Triage("keys", None, False, skill="go_to_site", text="youtube")
        with patch("brain.agent.race.triage", side_effect=[first, second]) as triaged:
            plan = self.race.next_actions(b"img", self.ctx("open chrome and go to youtube"))
            self.assertEqual(plan.actions[0].name, "open_app")
            self.race.acknowledge(True)
            plan = self.race.next_actions(b"img", self.ctx("open chrome and go to youtube", step=2))
            self.assertEqual(triaged.call_args.args[2], "and go to youtube")
            self.assertEqual([a.type for a in plan.actions], ["hotkey", "type_text", "hotkey"])
            self.race.acknowledge(True)
            plan = self.race.next_actions(b"img", self.ctx("open chrome and go to youtube", step=3))
        self.assertEqual(plan.actions[0].type, "done")

    def test_the_planner_gets_only_what_is_left_of_a_chain(self):
        first = Triage("keys", None, False, skill="open_app", text="chrome", rest="and write a poem in it")
        with patch("brain.agent.race.triage", side_effect=[first, Triage("plan", None, False)]):
            self.race.next_actions(b"img", self.ctx("open chrome and write a poem in it"))
            self.race.acknowledge(True)
            self.race.next_actions(b"img", self.ctx("open chrome and write a poem in it", step=2))
        planned_goal = self.race.planner.next_actions.call_args.args[1].goal
        self.assertEqual(planned_goal, "and write a poem in it")

    def test_os_without_a_keyboard_route_uses_the_plan(self):
        with patch("brain.agent.race.triage", return_value=Triage("keys", None, False, skill="go_to_site",
                                                                  text="youtube")):
            plan = self.race.next_actions(b"img", self.ctx("go to youtube", os="android"))
        self.assertEqual((self.race.mode, plan.reasoning), ("plan", "planned"))


if __name__ == "__main__":
    unittest.main()
