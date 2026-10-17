"""Triage and the race executor: routing, zoom refinement, irreversible approval."""

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from brain.agent.executor import Plan, StepContext
from brain.agent.jev_only import JevOnlyPlanner, spans
from brain.agent.memory import Memory
from brain.agent.race import RaceExecutor
from brain.agent.triage import Triage, triage
from brain.cancel import CancelToken
from brain.types import Action, DeviceProfile, Rect

BOX = {"box": {"left": .2, "right": .4, "top": .2, "bottom": .4}, "confidence": .9}


def jpeg(width=400, height=300) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((height, width, 3), 200, np.uint8))
    return buf.tobytes()


class TriageTests(unittest.TestCase):
    def setUp(self):
        self.jev = Mock()
        self.size = ((400, 300), (400, 300))

    def answer(self, choice, present=.95, unique=.95, irreversible=.02):
        return ({"intent": {"choice": choice, "confidence": .9}, "present": {"noul": present}, "gesture": {"choice": "click"},
                 "unique": {"noul": unique}, "irreversible": {"noul": irreversible}, "target": BOX}, *self.size)

    def test_remark_is_no_action_without_zoom(self):
        self.jev.evaluate.return_value = self.answer("no_action")
        self.assertEqual(triage(self.jev, jpeg(), "nice", CancelToken()), Triage("none", None, False))
        self.jev.evaluate.assert_called_once()

    def test_confident_target_is_zoom_refined(self):
        zoom = ({"present": {"noul": .9}, "target": {"box": {"left": .4, "right": .6, "top": .4, "bottom": .6}}},
                *self.size)
        self.jev.evaluate.side_effect = [self.answer("act_now"), zoom]
        result = triage(self.jev, jpeg(), "open john's email", CancelToken())
        self.assertEqual(result.mode, "act")
        self.assertEqual(self.jev.evaluate.call_count, 2)
        x, y = result.point
        self.assertTrue(0 <= x <= 1 and 0 <= y <= 1)

    def test_ambiguous_target_goes_to_the_planner(self):
        self.jev.evaluate.return_value = self.answer("act_now", unique=.3)
        self.assertEqual(triage(self.jev, jpeg(), "the other one", CancelToken()).mode, "plan")
        self.jev.evaluate.assert_called_once()

    def test_target_lost_on_zoom_goes_to_the_planner(self):
        zoom = ({"present": {"noul": .1}, "target": BOX}, *self.size)
        self.jev.evaluate.side_effect = [self.answer("act_now"), zoom]
        self.assertEqual(triage(self.jev, jpeg(), "open it", CancelToken()).mode, "plan")

    def test_irreversible_is_flagged(self):
        self.jev.evaluate.return_value = self.answer("plan", irreversible=.95)
        self.assertTrue(triage(self.jev, jpeg(), "delete the downloads folder", CancelToken()).irreversible)


class RaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        race = object.__new__(RaceExecutor)
        race.planner, race.jev, race.cancel = Mock(), Mock(), CancelToken()
        race._pool = ThreadPoolExecutor(max_workers=1)
        race._plan, race._key, race.mode, race._point, race._clicked, race._pending = None, None, "", None, False, None
        race._gesture, race._utterance, race._triage_step, race._approved_goal = "click", "", 0, None
        self.addCleanup(race._pool.shutdown)
        race.planner.next_actions.return_value = Plan("planned", [Action("hotkey", keys=("ENTER",))], [])
        self.race = race
        self.memory = Memory(Path(self.temp.name) / "m.json")

    def ctx(self, goal, run="a", note="", step=1, connected_at=1):
        return StepContext(goal, DeviceProfile("linux", Rect(0, 0, 400, 300), "mouse", connected_at), self.memory,
                           [], step, 10, note=note, run_id=run)

    def test_confident_jev_clicks_without_waiting_for_the_plan(self):
        with patch("brain.agent.race.triage", return_value=Triage("act", (.3, .4), False)):
            first = self.race.next_actions(b"img", self.ctx("open john's email"))
        self.assertEqual((first.actions[0].type, first.actions[0].x), ("click", .3))
        self.race.acknowledge(True)
        self.assertEqual(self.race.next_actions(b"img", self.ctx("open john's email")).actions[0].type, "done")

    def test_click_without_visible_effect_falls_back_to_the_planner(self):
        with patch("brain.agent.race.triage", return_value=Triage("act", (.3, .4), False)):
            self.race.next_actions(b"img", self.ctx("open it"))
        self.race.acknowledge(True)
        result = self.race.next_actions(b"img", self.ctx("open it", note="Screen unchanged; try something different."))
        self.assertEqual(result.reasoning, "planned")

    def test_remark_is_remembered_without_input(self):
        with patch("brain.agent.race.triage", return_value=Triage("none", None, False)):
            result = self.race.next_actions(b"img", self.ctx("oh john wants a report"))
        self.assertEqual([a.type for a in result.actions], ["remember", "done"])

    def test_plan_uses_the_planner_started_in_parallel(self):
        with patch("brain.agent.race.triage", return_value=Triage("plan", None, False)):
            result = self.race.next_actions(b"img", self.ctx("draft a reply"))
        self.assertEqual(result.reasoning, "planned")
        self.race.planner.next_actions.assert_called_once()

    def test_irreversible_request_waits_for_approval(self):
        with patch("brain.agent.race.triage", return_value=Triage("act", (.3, .4), True)):
            asked = self.race.next_actions(b"img", self.ctx("delete the downloads folder"))
        self.assertEqual(asked.actions[0].type, "ask_user")
        with patch("brain.agent.race.answer_to_pending", return_value="approve"), \
             patch("brain.agent.race.triage", return_value=Triage("act", (.3, .4), True)) as tri:
            go = self.race.next_actions(b"img", self.ctx("delete the downloads folder\nUser correction: yes", run="b"))
        self.assertEqual(go.actions[0].type, "click")
        self.assertEqual(tri.call_args.args[2], "delete the downloads folder")

    def test_declined_irreversible_request_does_nothing(self):
        with patch("brain.agent.race.triage", return_value=Triage("plan", None, True)):
            self.race.next_actions(b"img", self.ctx("send it"))
        with patch("brain.agent.race.answer_to_pending", return_value="decline"), \
             patch("brain.agent.race.triage", return_value=Triage("none", None, False)):
            result = self.race.next_actions(b"img", self.ctx("send it\nUser correction: no wait", run="b"))
        self.assertEqual([a.type for a in result.actions], ["done"])
        self.assertIn("won't", result.actions[0].summary)

    def test_decline_with_a_new_instruction_carries_it_out(self):
        with patch("brain.agent.race.triage", return_value=Triage("plan", None, True)):
            self.race.next_actions(b"img", self.ctx("delete all of these emails"))
        with patch("brain.agent.race.answer_to_pending", return_value="decline"), \
             patch("brain.agent.race.triage", return_value=Triage("act", (.4, .2), False)):
            result = self.race.next_actions(b"img", self.ctx(
                "delete all of these emails\nUser correction: no, open john's email instead", run="b"))
        self.assertEqual(result.actions[0].type, "click")


    def test_reply_that_is_neither_yes_nor_no_runs_as_a_new_request(self):
        with patch("brain.agent.race.triage", return_value=Triage("plan", None, True)):
            self.race.next_actions(b"img", self.ctx("delete all of these emails"))
        with patch("brain.agent.race.answer_to_pending", return_value="other"), \
             patch("brain.agent.race.triage", return_value=Triage("act", (.2, .2), False)) as tri:
            result = self.race.next_actions(b"img", self.ctx("delete all of these emails\nUser correction: open "
                                                             "calendar", run="b"))
        self.assertEqual(result.actions[0].type, "click")
        self.assertEqual(tri.call_args.args[2], "User correction: open calendar")

    def test_pending_approval_is_dropped_after_a_device_swap(self):
        with patch("brain.agent.race.triage", return_value=Triage("plan", None, True)):
            self.race.next_actions(b"img", self.ctx("delete all of these emails"))
        with patch("brain.agent.race.answer_to_pending") as answer, \
             patch("brain.agent.race.triage", return_value=Triage("act", (.2, .2), False)):
            self.race.next_actions(b"img", self.ctx("yes", run="b", connected_at=2))
        answer.assert_not_called()

    def test_approved_run_plans_only_the_approved_request(self):
        with patch("brain.agent.race.triage", return_value=Triage("plan", None, True)):
            self.race.next_actions(b"img", self.ctx("delete this email"))
        with patch("brain.agent.race.answer_to_pending", return_value="approve"), \
             patch("brain.agent.race.triage", return_value=Triage("plan", None, True)):
            self.race.next_actions(b"img", self.ctx("delete this email\nUser correction: yes and empty the trash",
                                                    run="b"))
        planned_goal = self.race.planner.next_actions.call_args.args[1].goal
        self.assertNotIn("empty the trash", planned_goal)
        self.assertIn("delete this email", planned_goal)

    def test_discarded_click_is_aimed_again_on_the_new_screen(self):
        with patch("brain.agent.race.triage", return_value=Triage("act", (.3, .4), False)):
            self.race.next_actions(b"img", self.ctx("open john's email", step=1))
        # No acknowledgement: the loop discarded the click because the screen moved.
        with patch("brain.agent.race.triage", return_value=Triage("act", (.6, .7), False)) as tri:
            again = self.race.next_actions(b"moved", self.ctx("open john's email", step=2))
        tri.assert_called_once()
        self.assertEqual((again.actions[0].x, again.actions[0].y), (.6, .7))

    def test_gesture_from_triage_is_used(self):
        with patch("brain.agent.race.triage", return_value=Triage("act", (.3, .4), False, "double_click")):
            result = self.race.next_actions(b"img", self.ctx("open up chrome"))
        self.assertEqual(result.actions[0].type, "double_click")


if __name__ == "__main__":
    unittest.main()


class JevOnlyTests(unittest.TestCase):
    def setUp(self):
        self.planner = object.__new__(JevOnlyPlanner)
        self.planner.jev = Mock(min_presence=.85)

    def plan(self, goal, complete=.1, next="click", present=.95, text="none", replace=.1, note=""):
        self.planner.jev.evaluate.return_value = ({
            "complete": {"noul": complete}, "next": {"choice": next}, "present": {"noul": present},
            "target": BOX, "text": {"choice": text}, "replace": {"noul": replace}}, (400, 300), (400, 300))
        profile = DeviceProfile("linux", Rect(0, 0, 400, 300), "mouse", 1)
        return self.planner.next_actions(b"img", StepContext(goal, profile, Mock(), [], 2, 10, note=note)).actions

    def step(self, goal, **answers):
        return self.plan(goal, **answers)[0]

    def test_goal_jev_sees_done_is_done(self):
        self.assertEqual(self.step("open calculator", complete=.9).type, "done")

    def test_next_click_lands_on_the_target(self):
        action = self.step("open calculator, then click 7")
        self.assertEqual((action.type, round(action.x, 2), round(action.y, 2)), ("click", .3, .3))

    def test_unclear_target_fails_instead_of_guessing(self):
        self.assertEqual(self.step("click the other one", present=.4).type, "fail")

    def test_typed_text_is_a_run_of_the_goals_words(self):
        goal = 'type "hello world", then press enter'
        action = self.step(goal, next="type", text=str(spans(goal).index("hello world")))
        self.assertEqual((action.type, action.text), ("type_text", "hello world"))
        self.assertEqual(self.step("write a poem", next="type").type, "fail")
        self.assertEqual(self.step("type hi", next="type", text="99").type, "fail")

    def test_filled_box_is_selected_before_typing(self):
        actions = self.plan("search for cats", next="type", text=str(spans("search for cats").index("cats")),
                            replace=.9)
        self.assertEqual([a.brief() for a in actions], ["hotkey(CTRL+A)", "type_text('cats')"])

    def test_fill_clicks_the_field_then_types(self):
        goal = 'enter "Morgan" in Name'
        actions = self.plan(goal, next="fill", text=str(spans(goal).index("Morgan")))
        self.assertEqual([a.type for a in actions], ["click", "type_text"])
        self.assertEqual(self.step("enter it in Name", next="fill", present=.3).type, "fail")

    def test_spans_are_unique_and_unquoted(self):
        self.assertEqual(spans('enter "a b" here.'), ["enter", "a", "b", "here", 'enter "a', 'a b', 'b" here',
                                                      'enter "a b', 'a b" here', 'enter "a b" here'])

    def test_step_with_no_visible_effect_fails(self):
        self.assertEqual(self.step("open it", note="Screen unchanged; try something different.").type, "fail")

    def test_reasoning_needs_the_language_model(self):
        self.assertIn("EXECUTOR=race", self.step("summarize this page", next="stuck").reason)

    def test_os_comes_from_a_jev_choice(self):
        self.planner.jev.evaluate.return_value = ({"os": {"choice": "android"}}, None, None)
        self.assertEqual(self.planner.classify_os(b"img"), "android")
        self.planner.jev.evaluate.return_value = ({"os": {"choice": "beos"}}, None, None)
        with self.assertRaises(ValueError):
            self.planner.classify_os(b"img")
