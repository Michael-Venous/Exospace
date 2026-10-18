"""Triage and the race executor: routing, zoom refinement, irreversible approval."""

import re
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from brain.agent.executor import Plan, StepContext
from brain.agent.jev_only import JevOnlyPlanner, spans as goal_spans
from brain.agent.memory import Memory
from brain.agent.race import RaceExecutor
from brain.agent.triage import ZOOM_WINDOW, Triage, refine, spans, triage, zoom_window
from brain.cancel import CancelToken
from brain.types import Action, DeviceProfile, Rect

BOX = {"box": {"left": .2, "right": .4, "top": .2, "bottom": .4}, "confidence": .9}


def jpeg(width=400, height=300) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((height, width, 3), 200, np.uint8))
    return buf.tobytes()


class TriageTests(unittest.TestCase):
    def setUp(self):
        self.jev = Mock(min_presence=.85)
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

    def test_zoom_moves_the_click_only_when_its_box_misses_the_first_look(self):
        def zoom(box):
            self.jev.evaluate.return_value = ({"present": {"noul": .9}, "target": {"box": box}}, *self.size)
            return refine(self.jev, jpeg(), {}, (.5, .5))
        # The whole control boxed around the first point: its center (lower down) may be a gap, so keep the point.
        self.assertEqual(zoom({"left": .1, "right": .9, "top": .4, "bottom": .9}), (.5, .5))
        x, y = zoom({"left": .1, "right": .3, "top": .1, "bottom": .3})  # the first look was off the control
        self.assertTrue(x < .5 and y < .5)

    def test_an_unsure_zoom_does_not_move_the_click(self):
        self.jev.evaluate.return_value = ({"present": {"noul": .8}, "target": {"box": {
            "left": .1, "right": .3, "top": .1, "bottom": .3}}}, *self.size)
        self.assertEqual(refine(self.jev, jpeg(), {}, (.5, .5)), (.5, .5))

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
        race._scroll, race._launched, race._tries, race._aim = 0, "", 0, (0.0, 0.0)
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

    def test_after_a_quick_command_the_planner_checks_it_instead_of_triage_redoing_it(self):
        ctx = self.ctx("open up a new tab")
        ctx.history.append("hotkey(CTRL+T)")
        with patch("brain.agent.race.triage", return_value=Triage("act", (.3, .4), False)):
            result = self.race.next_actions(b"img", ctx)
        self.assertEqual(result.reasoning, "planned")  # not triage's click

    def test_after_a_quick_command_an_irreversible_rest_still_asks_first(self):
        ctx = self.ctx("open discord and text peter you smell bad")
        ctx.history.append("skill(open_app {'name': 'discord'})")
        with patch("brain.agent.race.triage", return_value=Triage("plan", None, True)):
            result = self.race.next_actions(b"img", ctx)
        self.assertEqual(result.actions[0].type, "ask_user")

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


    def test_a_borderline_completion_after_a_click_is_left_to_the_planner(self):
        with patch("brain.agent.race.triage", return_value=Triage("act", (.98, .01), False)):
            self.race.next_actions(b"img", self.ctx("close the calculator"))
        self.race.acknowledge(True)
        self.race.jev.completion.return_value = .54  # measured: a corner click scored this, calculator still open
        self.assertEqual(self.race.next_actions(b"img", self.ctx("close the calculator")).reasoning, "planned")

    def test_a_small_target_is_hovered_and_confirmed_before_the_click(self):
        with patch("brain.agent.race.triage", return_value=Triage("act", (.98, .01), False, small=True)):
            first = self.race.next_actions(b"img", self.ctx("close the calculator", step=1))
        self.assertEqual((first.actions[0].name, first.actions[0].args), ("hover", {"x": .98, "y": .01}))
        self.race.acknowledge(True)
        with patch("brain.agent.race.pointed", return_value=True):
            click = self.race.next_actions(b"img", self.ctx("close the calculator", step=2))
        self.assertEqual((click.actions[0].type, click.actions[0].x), ("click", .98))

    def test_a_hover_on_the_wrong_neighbour_tries_the_neighbours(self):
        with patch("brain.agent.race.triage", return_value=Triage("act", (.98, .01), False, small=True)):
            self.race.next_actions(b"img", self.ctx("close the calculator", step=1))
        with patch("brain.agent.race.pointed", side_effect=[False, True]):
            again = self.race.next_actions(b"img", self.ctx("close the calculator", step=2))
            self.assertAlmostEqual(again.actions[0].args["x"], .99)  # the next button over
            click = self.race.next_actions(b"img", self.ctx("close the calculator", step=3))
        self.assertEqual((click.actions[0].type, round(click.actions[0].x, 2)), ("click", .99))

    def test_a_small_target_no_hover_confirms_is_left_to_the_planner_with_a_note(self):
        with patch("brain.agent.race.triage", return_value=Triage("act", (.5, .5), False, small=True)):
            self.race.next_actions(b"img", self.ctx("close the calculator", step=1))
        with patch("brain.agent.race.pointed", return_value=False):
            for step in range(2, 8):
                self.assertEqual(self.race.next_actions(b"img", self.ctx("close the calculator", step=step)).reasoning,
                                 "Jev hover check")
            last = self.race.next_actions(b"img", self.ctx("close the calculator", step=8))
        self.assertEqual(last.reasoning, "planned")  # never a click Jev couldn't confirm
        self.assertIn("another way", self.race.planner.next_actions.call_args.args[1].note)

    def test_touch_targets_are_tapped_without_a_hover(self):
        with patch("brain.agent.race.triage", return_value=Triage("act", (.98, .01), False, small=True)):
            ctx = self.ctx("close it")
            ctx = StepContext(ctx.goal, DeviceProfile("android", Rect(0, 0, 400, 300), "touch", 1), self.memory,
                              [], 1, 10, run_id="a")
            self.assertEqual(self.race.next_actions(b"img", ctx).actions[0].type, "click")

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

    def test_a_launch_that_landed_elsewhere_goes_to_the_planner(self):
        with patch("brain.agent.race.triage", return_value=Triage("launch", None, False, app="calculadora")):
            self.race.next_actions(b"img", self.ctx("abre la calculadora"))
        self.race.acknowledge(True)
        self.race.jev.evaluate.return_value = ({"launched": {"noul": .1}}, None, None)  # a web search instead
        self.race.jev.completion.return_value = .9
        self.assertEqual(self.race.next_actions(b"img", self.ctx("abre la calculadora")).reasoning, "planned")
        self.assertIn("did not open", self.race.planner.next_actions.call_args.args[1].note)

    def test_a_scroll_with_no_effect_goes_to_the_planner(self):
        with patch("brain.agent.race.triage", return_value=Triage("scroll", None, False, scroll=5)):
            self.race.next_actions(b"img", self.ctx("scroll down"))
        self.race.acknowledge(True)
        result = self.race.next_actions(b"img", self.ctx("scroll down", note="Screen unchanged; try something different."))
        self.assertEqual(result.reasoning, "planned")

if __name__ == "__main__":
    unittest.main()


class JevOnlyTests(unittest.TestCase):
    def setUp(self):
        self.planner = JevOnlyPlanner(Mock(min_presence=.85))

    def test_it_has_every_method_the_race_calls_on_its_planner(self):
        race = open(Path(__file__).parent.parent / "brain/agent/race.py").read()
        for name in set(re.findall(r"self\.planner\.(\w+)\(", race)):
            self.assertTrue(callable(getattr(self.planner, name, None)), name)

    def plan(self, goal, complete=.1, next="click", present=.95, text="none", replace=.1, note="",
             fits=.95, app_present=.1, app_name="none", launcher=.1, history=(), os="linux", zoom_present=.9,
             zooms=None, question=.05, composed=.05, loading=.05, done=0, exactly=()):
        step = {"complete": {"noul": complete}, "question": {"noul": question}, "composed": {"noul": composed}, "next": {"choice": next, "probabilities": {"done": done}}, "present": {"noul": present},
                "target": BOX, "text": {"choice": text}, "replace": {"noul": replace}, "loading": {"noul": loading}, "fits": {"noul": fits},
                "app_present": {"noul": app_present}, "app_name": {"choice": app_name}, "launcher": {"noul": launcher},
                "app": {"box": {"left": .1, "right": .5, "top": .6, "bottom": .7}}}
        # The zoomed second look boxes the middle of its crop: the click stays where the first box was.
        zoom = {"present": {"noul": zoom_present}, "target": {"box": {"left": .4, "right": .6, "top": .4, "bottom": .6}}}
        zooms = iter(zooms or [zoom, zoom]).__next__  # `next` is the step choice here
        # The text to type less its first i words: how sure Jev is that's exactly the text (`exactly`, else 0.1).
        rests = lambda questions: {k: {"noul": dict(enumerate(exactly)).get(i, .1)} for i, k in enumerate(questions)}
        self.planner.jev.evaluate.side_effect = lambda image, questions, **kw: (
            step if "next" in questions else rests(questions) if "rest0" in questions else zooms(), (400, 300),
            (400, 300))
        profile = DeviceProfile(os, Rect(0, 0, 400, 300), "mouse", 1)
        ctx = StepContext(goal, profile, Mock(), list(history), 2, 10, note=note, run_id="r")
        return self.planner.next_actions(jpeg(), ctx).actions

    def step(self, goal, **answers):
        return self.plan(goal, **answers)[0]

    def test_a_zoom_takes_in_a_control_wider_than_its_window(self):
        bar = {"box": {"left": .29, "right": .97, "top": .48, "bottom": .5}}  # Discord's message bar
        self.assertAlmostEqual(zoom_window(bar, (2880, 3600), (2880, 3600)), 1.2 * .68 * 2880 / 3600)
        self.assertEqual(zoom_window(BOX, (400, 300), (400, 300)), ZOOM_WINDOW)

    def test_an_app_a_quick_command_opened_is_not_routed_again(self):
        opened = ["skill(open_app {'name': 'display'})"]
        name = str(goal_spans("open up display").index("display"))
        self.assertEqual(self.step("open up display", fits=.35, app_name=name, history=opened).type, "click")
        self.assertEqual(self.step("open up display", fits=.35, app_name=name).brief(), "hotkey(GUI)")  # control

    def test_routing_waits_for_the_app_chosen_in_the_launcher(self):
        goal, typed = "open up my display settings", ["hotkey(GUI)", "type_text('display settings')"]
        name = str(goal_spans(goal).index("display settings"))
        self.assertEqual(self.step(goal, next="enter", fits=.55, launcher=.99, app_name=name, history=typed).brief(),
                         "hotkey(ENTER)")
        # Its window isn't drawn yet: wait for it instead of failing.
        self.assertEqual(self.step(goal, next="stuck", fits=.1, app_name=name,
                                   history=typed + ["hotkey(ENTER)"]).brief(), "skill(wait {})")

    def test_goal_jev_sees_done_is_done(self):
        self.assertEqual(self.step("open calculator", complete=.9).type, "done")

    def test_a_page_still_loading_is_waited_for_a_few_times(self):
        wait = "skill(wait {})"
        self.assertEqual(self.step("search youtube for cats", loading=.9).name, "wait")
        self.assertEqual(self.step("search youtube for cats", loading=.9, history=[wait] * 2).name, "wait")
        self.assertEqual(self.step("search youtube for cats", loading=.9, history=[wait] * 3).type, "click")

    def test_jev_choosing_no_next_step_is_done(self):
        goal = "open Discord, click Peter, and text you smell"
        self.assertEqual(self.step(goal, complete=.3, next="done", done=.9).type, "done")
        self.assertEqual(self.step(goal, complete=.3, next="done", done=.6).type, "ask_user")  # not confirmed: asked

    def test_unfinished_goals_are_not_done(self):
        self.assertEqual(self.step("go to wikipedia", complete=.63).type, "click")
        typed = self.plan("search for cats", complete=.9, next="enter", history=["type_text('cats')"])
        self.assertEqual([a.brief() for a in typed], ["hotkey(ENTER)"])
        self.assertEqual(self.step("search for cats", complete=.9, next="enter", history=["hotkey(ENTER)"]).type, "done")

    def test_next_click_lands_on_the_target(self):
        action = self.step("open calculator, then click 7")
        self.assertEqual((action.type, round(action.x, 2), round(action.y, 2)), ("click", .3, .3))

    def test_unclear_target_fails_instead_of_guessing(self):
        self.assertEqual(self.step("click the other one", present=.4).type, "skill")  # it may still be drawing
        self.assertEqual(self.step("click the other one", history=["skill(wait {})"] * 3, present=.4).type, "fail")
        self.assertEqual(self.step("click the other one", zoom_present=.2, history=["skill(wait {})"] * 3).type, "fail")  # lost on the second look

    def test_a_zoom_that_strays_is_outvoted(self):
        centered = {"present": {"noul": .9}, "target": {"box": {"left": .4, "right": .6, "top": .4, "bottom": .6}}}
        stray = {"present": {"noul": .9}, "target": {"box": {"left": .4, "right": .6, "top": .9, "bottom": 1}}}
        for zooms in ([centered, stray], [stray, centered]):
            action = self.step("open calculator, then click 7", zooms=zooms)
            self.assertEqual((round(action.x, 2), round(action.y, 2)), (.3, .3))

    def test_an_unsure_target_is_clicked_only_where_both_zooms_confirm_it(self):
        centered = {"present": {"noul": .9}, "target": {"box": {"left": .4, "right": .6, "top": .4, "bottom": .6}}}
        stray = {"present": {"noul": .9}, "target": {"box": {"left": .4, "right": .6, "top": .9, "bottom": 1}}}
        self.assertEqual(self.step("search for horse videos", present=.8).type, "click")
        for zooms in ([centered, stray], [stray, centered], [centered, {"present": {"noul": .2}, "target": {}}]):
            self.assertEqual(self.step("search for horse videos", present=.8, zooms=zooms, history=["skill(wait {})"] * 3).type, "fail")
        self.assertEqual(self.step("search for horse videos", present=.69).type, "click")  # both zooms confirm it
        self.assertEqual(self.step("search for horse videos", present=.45).type, "skill")  # it may still be drawing
        self.assertEqual(self.step("search for horse videos", history=["skill(wait {})"] * 3, present=.45).type, "fail")  # not even leaning seen

    def test_two_zooms_that_agree_move_a_nearly_sure_click(self):
        moved = {"present": {"noul": .9}, "target": {"box": {"left": .4, "right": .6, "top": .9, "bottom": 1}}}
        action = self.step("open a new tab and search cats", present=.8, zooms=[moved, moved])
        self.assertEqual(action.type, "click")
        self.assertGreater(action.y, .31)  # off the first box, where both zooms put it

    def test_two_unsure_zooms_that_agree_elsewhere_move_the_click(self):
        # Discord: the first box was the user panel, both zooms (0.72 sure) found the message bar.
        moved = {"present": {"noul": .7}, "target": {"box": {"left": .4, "right": .6, "top": .9, "bottom": 1}}}
        action = self.step("open a new tab and search cats", zooms=[moved, moved])
        self.assertGreater(action.y, .31)
        apart = {"present": {"noul": .7}, "target": {"box": {"left": .0, "right": .1, "top": .9, "bottom": 1}}}
        action = self.step("open a new tab and search cats", zooms=[moved, apart])
        self.assertEqual((round(action.x, 2), round(action.y, 2)), (.3, .3))  # no agreement: the first box stands
        # Zooms that each boxed elsewhere never confirm a nearly sure first box.
        self.assertEqual(self.step("search cats", present=.8, zooms=[moved, apart], history=["skill(wait {})"] * 3).type, "fail")

    def test_leading_words_jev_reads_as_the_instruction_are_not_typed(self):
        goal = "go to youtube, search for embedded systems videos and click the third reel"
        chosen = str(goal_spans(goal).index("search for embedded systems videos"))
        actions = self.plan(goal, next="fill", text=chosen, exactly=(.64, .6, .79, .34, .05))
        self.assertEqual(actions[-1].text, "embedded systems videos")
        actions = self.plan(goal, next="fill", text=chosen, exactly=(.5, .5, .5, .3, .1))
        self.assertEqual(actions[-1].text, "search for embedded systems videos")  # a tie keeps the words

    def test_typed_text_is_a_run_of_the_goals_words(self):
        goal = 'type "hello world", then press enter'
        action = self.step(goal, next="type", text=str(goal_spans(goal).index("hello world")), history=["click(1,1)"])
        self.assertEqual((action.type, action.text), ("type_text", "hello world"))
        self.assertEqual(self.step("write a poem", next="type").type, "fail")
        self.assertEqual(self.step("type hi", next="type", text="99").type, "fail")

    def test_filled_box_is_selected_before_typing(self):
        actions = self.plan("search for cats", next="type", text=str(goal_spans("search for cats").index("cats")),
                            replace=.9, history=["click(1,1)"])
        self.assertEqual([a.brief() for a in actions], ["hotkey(CTRL+A)", "type_text('cats')", "hotkey(DELETE)"])

    def test_typing_without_a_click_before_it_clicks_the_target_first(self):
        goal = "go to youtube, search for cats"
        text = str(goal_spans(goal).index("cats"))
        for history in ([], ["hotkey(ENTER)"], ["click(1,1)", "hotkey(ENTER)"]):  # focus a screen can't show
            self.assertEqual([a.type for a in self.plan(goal, next="type", text=text, history=history)],
                             ["click", "type_text"])
        actions = self.plan(goal, next="type", text=text, history=["click(1,1)", "skill(wait {})"])
        self.assertEqual([a.type for a in actions], ["type_text"])

    def test_fill_clicks_the_field_then_types(self):
        goal = 'enter "Morgan" in Name'
        actions = self.plan(goal, next="fill", text=str(goal_spans(goal).index("Morgan")))
        self.assertEqual([a.type for a in actions], ["click", "type_text"])
        self.assertEqual(self.step("enter it in Name", next="fill", present=.3).type, "fail")

    def test_spans_are_unique_and_unquoted(self):
        self.assertEqual(goal_spans('enter "a b" here.'), ["enter", "a", "b", "here", 'enter "a', 'a b', 'b" here',
                                                      'enter "a b', 'a b" here'])

    def test_spans_stay_inside_a_line_and_never_offer_a_whole_request(self):
        texts = goal_spans("Just did: Text swamp tech.\nGo to Discord and say hi.\nUser correction: Go ahead.")
        self.assertIn("swamp tech", texts)
        self.assertIn("hi", texts)
        for whole in ("Text swamp tech", "Go to Discord and say hi", "Go ahead", "tech.\nGo"):
            self.assertNotIn(whole, texts)
        self.assertEqual(goal_spans("Discord"), ["Discord"])  # one word is no command plus text

    def test_a_question_or_nonsense_is_left_to_flash(self):
        self.assertEqual(self.step("what is this page about?", complete=.9, question=.9).type, "fail")
        self.assertEqual(self.step("asdkjh qwe zxcv", fits=.1, next="stuck").type, "fail")  # no launcher

    def test_text_the_request_does_not_give_word_for_word_is_left_to_flash(self):
        self.assertEqual(self.step("work out 12 times 9", composed=.9).type, "fail")

    def test_step_with_no_visible_effect_fails(self):
        self.assertEqual(self.step("open it", note="Screen unchanged; try something different.").type, "fail")

    def test_reasoning_needs_the_language_model(self):
        self.assertIn("EXECUTOR=race", self.step("summarize this page", next="stuck").reason)

    def test_app_in_front_that_fits_takes_the_step_and_ends_routing(self):
        self.assertEqual(self.step("find a video about horses").type, "click")
        self.assertEqual(self.step("find a video about horses", fits=.1).type, "click")  # routed once per run

    def test_visible_suited_app_is_switched_to(self):
        goal = "find a video about horses"
        action = self.step(goal, fits=.1, app_present=.95, app_name=str(goal_spans(goal).index("video")))
        self.assertEqual((action.type, round(action.x, 2), round(action.y, 2)), ("click", .3, .65))

    def test_no_suited_app_opens_the_launcher_waits_once_for_the_app_then_fails(self):
        goal = "find a video about horses"
        named = str(goal_spans(goal).index("video"))
        self.assertEqual(self.step(goal, fits=.1, app_name=named).brief(), "hotkey(GUI)")
        self.assertEqual(self.step(goal, fits=.1, app_name=named, os="macos").brief(), "hotkey(GUI+SPACE)")
        self.assertEqual(self.step(goal, fits=.1, app_name=named, history=["hotkey(GUI)"]).brief(), "skill(wait {})")  # menu opening
        self.assertEqual(self.step(goal, fits=.1, app_name=named, history=["hotkey(GUI)", "skill(wait {})"]).type, "fail")
        opening = ["hotkey(GUI)", "click(120,260)"]
        self.assertEqual(self.step(goal, fits=.1, app_name=named, history=opening).brief(), "skill(wait {})")
        self.assertEqual(self.step(goal, fits=.1, app_name=named, history=opening + ["skill(wait {})"]).type, "fail")

    def test_launcher_is_searched_for_the_app_the_goal_names_before_an_icon_is_clicked(self):
        goal = "open up camera and take a picture"
        search = dict(fits=.1, app_name=str(goal_spans(goal).index("camera")), launcher=.9, app_present=.95,
                      history=["hotkey(GUI)"])
        self.assertEqual(self.step(goal, **search).brief(), "type_text('camera')")  # not the camera-shaped icon
        self.assertEqual(self.step(goal, **search | {"fits": .9}).brief(), "type_text('camera')")  # nor the goal
        self.assertEqual(self.step(goal, **search | {"app_present": .1, "launcher": .4}).brief(),
                         "skill(wait {})")  # no launcher seen yet
        typed = ["hotkey(GUI)", "type_text('camera')"]
        self.assertEqual(self.step(goal, **search | {"history": typed}).type, "click")  # then the result
        self.assertEqual(self.step(goal, **search | {"history": typed, "app_present": .1}).brief(), "skill(wait {})")
        self.assertEqual(self.step(goal, **search | {"history": []}).type, "click")  # a visible app needs no launcher

    def test_os_comes_from_a_jev_choice(self):
        self.planner.jev.evaluate.return_value = ({"os": {"choice": "android"}}, None, None)
        self.assertEqual(self.planner.classify_os(b"img"), "android")
        self.planner.jev.evaluate.return_value = ({"os": {"choice": "beos"}}, None, None)
        with self.assertRaises(ValueError):
            self.planner.classify_os(b"img")
