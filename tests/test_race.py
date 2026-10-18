"""Triage and the race executor: routing, zoom refinement, irreversible approval."""

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
import brain.agent.triage as triage_module
from brain.agent.triage import Triage, answer_to_pending, spans, triage
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

    def answer(self, choice, present=.95, unique=.95, irreversible=.02, app="none"):
        return ({"intent": {"choice": choice, "confidence": .9}, "present": {"noul": present}, "gesture": {"choice": "click"},
                 "unique": {"noul": unique}, "irreversible": {"noul": irreversible}, "target": BOX,
                 "app": {"choice": app}}, *self.size)

    def test_an_app_to_open_is_named_by_the_utterance_itself(self):
        words = spans("open the calculator and work out 12 times 9")
        self.jev.evaluate.return_value = self.answer("launch", app=str(words.index("calculator")))
        result = triage(self.jev, jpeg(), "open the calculator and work out 12 times 9", CancelToken())
        self.assertEqual((result.mode, result.app), ("launch", "calculator"))
        self.jev.evaluate.assert_called_once()  # no zoom: nothing to click

    def test_a_launch_with_no_named_app_goes_to_the_planner(self):
        self.jev.evaluate.return_value = self.answer("launch")
        self.assertEqual(triage(self.jev, jpeg(), "open it", CancelToken()).mode, "plan")

    def test_remark_is_no_action_without_zoom(self):
        self.jev.evaluate.return_value = self.answer("no_action")
        self.assertEqual(triage(self.jev, jpeg(), "nice", CancelToken()), Triage("none", None, False))
        self.jev.evaluate.assert_called_once()

    def test_confident_target_is_zoom_refined(self):
        zoom = ({"present": {"noul": .9}, "target": {"box": {"left": .4, "right": .6, "top": .4, "bottom": .6}}},
                *self.size)
        self.jev.evaluate.side_effect = [self.answer("act_now"), zoom, zoom]
        result = triage(self.jev, jpeg(), "open john's email", CancelToken())
        self.assertEqual(result.mode, "act")
        self.assertEqual(self.jev.evaluate.call_count, 3)  # the screen, a zoom, a tighter zoom
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

    def test_correction_is_judged_with_the_request_it_corrects(self):
        self.jev.evaluate.return_value = self.answer("plan")
        triage(self.jev, jpeg(), "User correction: actually make it six", CancelToken(), earlier="reply saying five")
        questions, state = self.jev.evaluate.call_args.args[1], self.jev.evaluate.call_args.kwargs["state"]
        self.assertEqual(state["earlier_request"], "reply saying five")
        self.assertTrue(questions["irreversible"]["instructions"].endswith(triage_module.CORRECTION_SCOPE))

    def test_plain_command_question_is_unchanged(self):
        self.jev.evaluate.return_value = self.answer("no_action")
        triage(self.jev, jpeg(), "open john's email", CancelToken())
        questions, state = self.jev.evaluate.call_args.args[1], self.jev.evaluate.call_args.kwargs["state"]
        self.assertEqual(state, {"utterance": "open john's email"})
        self.assertEqual(questions["irreversible"]["instructions"], triage_module.IRREVERSIBLE)

    def test_a_reply_is_judged_against_the_held_step(self):
        self.jev.evaluate.return_value = ({"approve": {"noul": .9}, "decline": {"noul": .1}}, *self.size)
        self.assertEqual(answer_to_pending(self.jev, jpeg(), "reply saying five", "yes", held="click Send button"),
                         "approve")
        state = self.jev.evaluate.call_args.kwargs["state"]
        self.assertEqual(state["held_step"], "click Send button")
        answer_to_pending(self.jev, jpeg(), "delete this email", "yes")
        self.assertEqual(self.jev.evaluate.call_args.kwargs["state"],
                         {"pending_request": "delete this email", "reply": "yes"})


class RaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        race = object.__new__(RaceExecutor)
        race.planner, race.jev, race.cancel = Mock(), Mock(), CancelToken()
        race._pool = ThreadPoolExecutor(max_workers=1)
        race._plan, race._key, race.mode, race._point, race._clicked, race._pending = None, None, "", None, False, None
        race._gesture, race._utterance, race._triage_step, race._approved_goal = "click", "", 0, None
        race._scroll, race._launched = 0, ""
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
        self.race.jev.completion.return_value = .9
        self.assertEqual(self.race.next_actions(b"img", self.ctx("open john's email")).actions[0].type, "done")

    def test_a_click_that_misses_the_goal_falls_back_to_the_planner(self):
        with patch("brain.agent.race.triage", return_value=Triage("act", (.3, .4), False)):
            self.race.next_actions(b"img", self.ctx("close the calculator"))
        self.race.acknowledge(True)
        self.race.jev.completion.return_value = .05  # e.g. the neighbouring minimize button
        result = self.race.next_actions(b"img", self.ctx("close the calculator"))
        self.assertEqual(result.reasoning, "planned")

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

    def held_send(self):
        """The planner reaches Send and holds it unless the run is approved, as JevFlashExecutor does."""
        def planner(image, ctx):
            if self.race.approved_run():
                return Plan("Stage 1/1", [Action("click", x=.5, y=.9)], [])
            return Plan("Irreversible step needs approval", [], [], held="click Send button")

        self.race.planner.next_actions.side_effect = planner
        return "open john's email and reply saying five\nUser correction: actually make it six"

    def test_the_planner_asks_race_whether_the_run_is_approved(self):
        for jev_first, name in ((False, "JevFlashExecutor"), (True, "JevFirstPlanner")):
            with patch(f"brain.agent.race.{name}") as planner:
                race = RaceExecutor(Mock(jev_first=jev_first), CancelToken())
            self.addCleanup(race.close)
            approved = planner.call_args.kwargs["approved"]
            self.assertFalse(approved())
            race._approved_goal = "delete this email\nThe user approved this request."
            self.assertTrue(approved())

    def test_correction_carries_the_request_it_corrects(self):
        goal = "reply to john and send it saying five\nUser correction: actually make it six"
        with patch("brain.agent.race.triage", return_value=Triage("plan", None, True)) as tri:
            asked = self.race.next_actions(b"img", self.ctx(goal))
        self.assertEqual(tri.call_args.args[2], "User correction: actually make it six")
        self.assertEqual(tri.call_args.kwargs["earlier"], "reply to john and send it saying five")
        self.assertEqual(asked.actions[0].type, "ask_user")
        self.assertEqual(self.race._pending.goal, goal)

    def test_correction_after_a_draft_request_asks_before_the_send(self):
        draft = self.held_send()
        with patch("brain.agent.race.triage", return_value=Triage("plan", None, False)):
            asked = self.race.next_actions(b"img", self.ctx(draft))
        self.assertEqual(asked.actions[0].question, "This can't be undone: click Send button. Should I go ahead?")
        with patch("brain.agent.race.answer_to_pending", return_value="approve") as answer, \
             patch("brain.agent.race.triage", return_value=Triage("plan", None, True)):
            sent = self.race.next_actions(b"img", self.ctx(draft + "\nUser correction: yes", run="b"))
        self.assertEqual(answer.call_args.kwargs["held"], "click Send button")
        self.assertEqual(sent.actions[0].type, "click")

    def test_a_stage_held_after_earlier_stages_asks(self):
        self.race.planner.next_actions.side_effect = [
            Plan("Stage 1/2", [Action("type_text", text="six")], []),
            Plan("Irreversible step needs approval", [], [], held="click Send button")]
        with patch("brain.agent.race.triage", return_value=Triage("plan", None, False)):
            self.race.next_actions(b"img", self.ctx("reply saying six", step=1))
        self.race.acknowledge(True)
        asked = self.race.next_actions(b"typed", self.ctx("reply saying six", step=2))
        self.assertEqual([a.type for a in asked.actions], ["ask_user"])
        self.assertEqual(self.race._pending.step, "click Send button")

    def test_held_step_approval_is_bound_to_the_device(self):
        draft = self.held_send()
        with patch("brain.agent.race.triage", return_value=Triage("plan", None, False)):
            self.race.next_actions(b"img", self.ctx(draft))
        with patch("brain.agent.race.answer_to_pending") as answer, \
             patch("brain.agent.race.triage", return_value=Triage("plan", None, False)):
            again = self.race.next_actions(b"img", self.ctx(draft + "\nUser correction: yes", run="b", connected_at=2))
        answer.assert_not_called()
        self.assertEqual(again.actions[0].type, "ask_user")

    def test_a_correction_to_a_held_step_is_planned_again_not_approved(self):
        draft = self.held_send()
        with patch("brain.agent.race.triage", return_value=Triage("plan", None, False)):
            self.race.next_actions(b"img", self.ctx(draft))
        seven = draft + "\nUser correction: actually make it seven"
        with patch("brain.agent.race.answer_to_pending", return_value="other"), \
             patch("brain.agent.race.triage", return_value=Triage("plan", None, False)) as tri:
            again = self.race.next_actions(b"img", self.ctx(seven, run="b"))
        self.assertEqual(tri.call_args.kwargs["earlier"], draft)
        self.assertFalse(self.race.approved_run())
        self.assertEqual(again.actions[0].type, "ask_user")

    def test_send_it_after_a_correction_asks(self):
        goal = "reply to john saying five\nUser correction: make it six\nUser correction: send it"
        with patch("brain.agent.race.triage", return_value=Triage("act", (.5, .9), True)) as tri:
            asked = self.race.next_actions(b"img", self.ctx(goal))
        self.assertEqual(asked.actions[0].type, "ask_user")
        self.assertEqual(tri.call_args.kwargs["earlier"], "reply to john saying five\nUser correction: make it six")

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


    def test_scroll_needs_no_plan_and_ends_when_the_view_moves(self):
        with patch("brain.agent.race.triage", return_value=Triage("scroll", None, False, scroll=-5)):
            first = self.race.next_actions(b"img", self.ctx("scroll up"))
        self.assertEqual((first.actions[0].type, first.actions[0].dy), ("scroll", -5))
        self.race.acknowledge(True)
        self.race.jev.completion.return_value = .9
        self.assertEqual(self.race.next_actions(b"img", self.ctx("scroll up")).actions[0].type, "done")

    def test_a_launch_needs_no_plan_and_the_rest_is_planned_on_the_app(self):
        goal = "open the calculator and work out 12 times 9"
        with patch("brain.agent.race.triage", return_value=Triage("launch", None, False, app="calculator")):
            first = self.race.next_actions(b"img", self.ctx(goal))
        self.assertEqual((first.actions[0].name, first.actions[0].args), ("open_app", {"name": "calculator"}))
        self.race.acknowledge(True)
        self.race.jev.evaluate.return_value = ({"launched": {"noul": .95}}, None, None)
        self.race.jev.completion.return_value = .1  # open, but nothing worked out yet
        self.assertEqual(self.race.next_actions(b"img", self.ctx(goal)).reasoning, "planned")
        self.race.planner.forget_plan.assert_called_once()

    def test_a_stage_held_after_a_launch_asks(self):
        goal = "open mail and send john six"
        with patch("brain.agent.race.triage", return_value=Triage("launch", None, False, app="mail")):
            self.race.next_actions(b"img", self.ctx(goal))
        self.race.acknowledge(True)
        self.race.jev.evaluate.return_value = ({"launched": {"noul": .95}}, None, None)
        self.race.jev.completion.return_value = .1
        self.race.planner.next_actions.return_value = Plan("held", [], [], held="click Send button")
        self.assertEqual(self.race.next_actions(b"mail", self.ctx(goal)).actions[0].type, "ask_user")

    def test_a_launch_that_landed_elsewhere_goes_to_the_planner(self):
        with patch("brain.agent.race.triage", return_value=Triage("launch", None, False, app="calculadora")):
            self.race.next_actions(b"img", self.ctx("abre la calculadora"))
        self.race.acknowledge(True)
        self.race.jev.evaluate.return_value = ({"launched": {"noul": .1}}, None, None)  # a web search instead
        self.race.jev.completion.return_value = .9
        self.assertEqual(self.race.next_actions(b"img", self.ctx("abre la calculadora")).reasoning, "planned")

    def test_a_scroll_with_no_effect_goes_to_the_planner(self):
        with patch("brain.agent.race.triage", return_value=Triage("scroll", None, False, scroll=5)):
            self.race.next_actions(b"img", self.ctx("scroll down"))
        self.race.acknowledge(True)
        result = self.race.next_actions(b"img", self.ctx("scroll down", note="Screen unchanged; try something different."))
        self.assertEqual(result.reasoning, "planned")

if __name__ == "__main__":
    unittest.main()
