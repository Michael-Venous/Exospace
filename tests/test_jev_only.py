"""Jev's own next step (ported from origin/jev-only) and the Jev-first planner that falls back to Flash."""

import unittest
from unittest.mock import Mock, patch

from brain.agent.executor import Plan, StepContext
from brain.agent.jev_first import JevFirstPlanner
from brain.agent.jev_only import FAILED, JevOnlyPlanner
from brain.agent.triage import spans
from brain.types import Action, DeviceProfile, Rect

BOX = {"box": {"left": .2, "right": .4, "top": .2, "bottom": .4}, "confidence": .9}
ON, STRAY = (.3, .3), (.3, .95)  # a zoomed look on the first box's center, and one a row off


class JevOnlyTests(unittest.TestCase):
    def setUp(self):
        self.planner = JevOnlyPlanner(Mock(min_presence=.85))

    def plan(self, goal, complete=.1, next="click", present=.95, text="none", replace=.1, note="",
             fits=.95, app_present=.1, app_name="none", launcher=.1, history=(), os="linux", looks=(ON, ON),
             question=.05, composed=.05):
        step = {"complete": {"noul": complete}, "question": {"noul": question}, "composed": {"noul": composed}, "next": {"choice": next}, "present": {"noul": present},
                "target": BOX, "text": {"choice": text}, "replace": {"noul": replace}, "fits": {"noul": fits},
                "app_present": {"noul": app_present}, "app_name": {"choice": app_name}, "launcher": {"noul": launcher},
                "app": {"box": {"left": .1, "right": .5, "top": .6, "bottom": .7}}}
        self.planner.jev.evaluate.return_value = (step, (400, 300), (400, 300))
        profile = DeviceProfile(os, Rect(0, 0, 400, 300), "mouse", 1)
        ctx = StepContext(goal, profile, Mock(), list(history), 2, 10, note=note, run_id="r")
        with patch("brain.agent.jev_only.refine", side_effect=list(looks)):
            return self.planner.next_actions(b"img", ctx).actions

    def step(self, goal, **answers):
        return self.plan(goal, **answers)[0]

    def test_goal_jev_sees_done_is_done(self):
        self.assertEqual(self.step("open calculator", complete=.9).type, "done")

    def test_unfinished_goals_are_not_done(self):
        self.assertEqual(self.step("go to wikipedia", complete=.63).type, "click")
        typed = self.plan("search for cats", complete=.9, next="enter", history=["type_text('cats')"])
        self.assertEqual([a.brief() for a in typed], ["hotkey(ENTER)"])
        self.assertEqual(self.step("search for cats", complete=.9, next="enter", history=["hotkey(ENTER)"]).type, "done")

    def test_next_click_lands_on_the_target(self):
        action = self.step("open calculator, then click 7")
        self.assertEqual((action.type, round(action.x, 2), round(action.y, 2)), ("click", .3, .3))

    def test_unclear_target_fails_instead_of_guessing(self):
        self.assertEqual(self.step("click the other one", present=.4).type, "fail")
        self.assertEqual(self.step("click the other one", looks=(None, None)).type, "fail")  # lost when zoomed

    def test_a_target_not_visible_after_a_step_is_looked_for_again_twice(self):
        goal = "search youtube for horses"
        self.assertEqual(self.step(goal, present=.4, history=["skill(open_app {'name': 'firefox'})"]).brief(),
                         "skill(wait {})")
        waited = ["skill(open_app {'name': 'firefox'})", "skill(wait {})", "skill(wait {})"]
        self.assertEqual(self.step(goal, present=.4, history=waited).type, "fail")

    def test_a_zoom_that_strays_is_outvoted(self):
        for looks in ((ON, STRAY), (STRAY, ON)):
            action = self.step("open calculator, then click 7", looks=looks)
            self.assertEqual((round(action.x, 2), round(action.y, 2)), (.3, .3))

    def test_a_nearly_sure_target_is_clicked_only_where_both_zooms_confirm_it(self):
        self.assertEqual(self.step("search for horse videos", present=.8).type, "click")
        for looks in ((ON, STRAY), (STRAY, ON), (ON, None)):
            self.assertEqual(self.step("search for horse videos", present=.8, looks=looks).type, "fail")
        self.assertEqual(self.step("search for horse videos", present=.7).type, "fail")

    def test_two_zooms_that_agree_move_a_nearly_sure_click(self):
        action = self.step("open a new tab and search cats", present=.8, looks=(STRAY, STRAY))
        self.assertEqual((action.type, action.y), ("click", .95))

    def test_typed_text_is_a_run_of_the_goals_words(self):
        goal = 'type "hello world", then press enter'
        action = self.step(goal, next="type", text=str(spans(goal, 250).index("hello world")),
                           history=["click(300,300)"])
        self.assertEqual((action.type, action.text), ("type_text", "hello world"))
        self.assertEqual(self.step("write a poem", next="type").type, "fail")
        self.assertEqual(self.step("type hi", next="type", text="99").type, "fail")

    def test_typing_not_right_after_a_click_clicks_the_field_first(self):
        goal = "search for cats"
        actions = self.plan(goal, next="type", text=str(spans(goal, 250).index("cats")), history=["hotkey(GUI)"])
        self.assertEqual([a.type for a in actions], ["click", "type_text"])

    def test_filled_box_is_selected_before_typing(self):
        actions = self.plan("search for cats", next="type", text=str(spans("search for cats").index("cats")), replace=.9,
                            history=["click(300,300)"])
        self.assertEqual([a.brief() for a in actions], ["hotkey(CTRL+A)", "type_text('cats')"])

    def test_fill_clicks_the_field_then_types(self):
        goal = 'enter "Morgan" in Name'
        actions = self.plan(goal, next="fill", text=str(spans(goal).index("Morgan")))
        self.assertEqual([a.type for a in actions], ["click", "type_text"])
        self.assertEqual(self.step("enter it in Name", next="fill", present=.3).type, "fail")

    def test_step_with_no_visible_effect_fails(self):
        self.assertEqual(self.step("open it", note="Screen unchanged; try something different.").type, "fail")

    def test_a_question_or_nonsense_is_left_to_flash(self):
        self.assertEqual(self.step("what is this page about?", complete=.9, question=.9).type, "fail")
        self.assertEqual(self.step("asdkjh qwe zxcv", fits=.1, next="stuck").type, "fail")  # no launcher

    def test_the_same_input_is_never_repeated_right_after_it_ran(self):
        self.assertEqual(self.step("open calculator, then click 7", history=["click(300,300)"]).type, "fail")
        self.assertEqual(self.step("open calculator, then click 7", history=["click(600,300)"]).type, "click")
        typed = self.plan("search for cats", next="enter", history=["type_text('cats')", "hotkey(ENTER)"])
        self.assertEqual(typed[0].type, "fail")

    def test_the_request_itself_is_never_typed(self):
        goal = "write a short note about my day"
        self.assertEqual(self.step(goal, next="type", text=str(spans(goal, 250).index(goal))).type, "fail")

    def test_a_scroll_first_moves_the_pointer_over_the_page(self):
        actions = self.plan("scroll down to the comments", next="scroll_down")
        self.assertEqual([a.brief() for a in actions], ["skill(hover {'x': 0.5, 'y': 0.5})", "scroll(5)"])

    def test_text_the_request_does_not_give_word_for_word_is_left_to_flash(self):
        self.assertEqual(self.step("work out 12 times 9", composed=.9).type, "fail")

    def test_reasoning_fails_over(self):
        self.assertEqual(self.step("summarize this page", next="stuck").type, "fail")

    def test_app_in_front_that_fits_takes_the_step_and_ends_routing(self):
        self.assertEqual(self.step("find a video about horses").type, "click")
        self.assertEqual(self.step("find a video about horses", fits=.1).type, "click")  # routed once per run

    def test_visible_suited_app_is_switched_to(self):
        goal = "find a video about horses"
        action = self.step(goal, fits=.1, app_present=.95, app_name=str(spans(goal, 250).index("video")))
        self.assertEqual((action.type, round(action.x, 2), round(action.y, 2)), ("click", .3, .65))

    def test_no_suited_app_opens_the_launcher_waits_once_for_the_app_then_fails(self):
        goal = "find a video about horses"
        named = str(spans(goal, 250).index("video"))
        self.assertEqual(self.step(goal, fits=.1, app_name=named).brief(), "hotkey(GUI)")
        self.assertEqual(self.step(goal, fits=.1, app_name=named, history=["hotkey(GUI)"]).brief(), "skill(wait {})")
        self.assertEqual(self.step(goal, fits=.1, app_name=named, history=["hotkey(GUI)", "skill(wait {})"]).type,
                         "fail")

    def test_launcher_is_searched_for_the_app_the_goal_names_before_an_icon_is_clicked(self):
        goal = "open up camera and take a picture"
        search = dict(fits=.1, app_name=str(spans(goal, 250).index("camera")), launcher=.9, app_present=.95,
                      history=["hotkey(GUI)"])
        self.assertEqual(self.step(goal, **search).brief(), "type_text('camera')")  # not the camera-shaped icon
        typed = ["hotkey(GUI)", "type_text('camera')"]
        self.assertEqual(self.step(goal, **search | {"history": typed}).type, "click")  # then the result


class JevOnlyGuardTests(unittest.TestCase):
    def next_step(self, approved):
        planner = JevOnlyPlanner(Mock(min_presence=.85), approved)
        answers = {"complete": {"noul": .1}, "question": {"noul": .05}, "composed": {"noul": .05},
                   "next": {"choice": "enter"},
                   "present": {"noul": .95}, "target": BOX, "text": {"choice": "none"}, "replace": {"noul": .1},
                   "fits": {"noul": .95}, "app_present": {"noul": .1}, "app_name": {"choice": "none"},
                   "launcher": {"noul": .1}, "app": BOX, "composed": {"noul": .05}, "irreversible": {"noul": .95}}
        planner.jev.evaluate.return_value = (answers, (400, 300), (400, 300))
        ctx = StepContext("reply saying six and send it", DeviceProfile("linux", Rect(0, 0, 400, 300), "mouse", 1),
                          Mock(), ["type_text('six')"], 2, 10, run_id="r")
        return planner.next_actions(b"img", ctx), planner.jev.evaluate.call_args.args[1]

    def test_an_unapproved_irreversible_next_step_is_held(self):
        plan, questions = self.next_step(lambda ctx: False)
        self.assertEqual((plan.actions, plan.held), ([], 'the next step of "reply saying six and send it"'))
        self.assertIn("irreversible", questions)

    def test_an_approved_run_takes_the_irreversible_next_step(self):
        plan, questions = self.next_step(lambda ctx: True)
        self.assertEqual(([a.brief() for a in plan.actions], plan.held), (["hotkey(ENTER)"], ""))
        self.assertNotIn("irreversible", questions)

    def test_without_a_policy_no_irreversible_question_is_asked(self):
        plan, questions = self.next_step(None)
        self.assertEqual([a.brief() for a in plan.actions], ["hotkey(ENTER)"])
        self.assertNotIn("irreversible", questions)

    def test_jev_first_gives_both_planners_the_policy(self):
        policy = Mock()
        with patch("brain.agent.jev_first.JevFlashExecutor") as flash, \
             patch("brain.agent.jev_first.JevOnlyPlanner") as steps:
            JevFirstPlanner(Mock(), Mock(), policy)
        self.assertIs(flash.call_args.args[2], policy)
        self.assertIs(steps.call_args.args[1], policy)


class PlannerInterfaceTests(unittest.TestCase):
    def test_both_race_planners_answer_everything_race_calls(self):
        # EXECUTOR=jev crashed on a missing forget_plan after a launch (nrebolloso, origin/live-speech).
        from brain.agent.jev_flash import JevFlashExecutor
        for planner in (JevFirstPlanner, JevFlashExecutor):
            for name in ("next_actions", "acknowledge", "forget_plan", "classify_os", "close", "image_side"):
                self.assertTrue(hasattr(planner, name), f"{planner.__name__}.{name}")


class JevFirstTests(unittest.TestCase):
    def setUp(self):
        self.planner = object.__new__(JevFirstPlanner)
        self.planner.flash, self.planner.steps, self.planner._flash_run = Mock(), Mock(), None
        self.planner.handoff = True
        self.planner.flash.next_actions.return_value = Plan("Stage 1/1: planned", [Action("hotkey", keys=("TAB",))], [])

    def ctx(self, run):
        return StepContext("summarize this page", DeviceProfile("linux", Rect(0, 0, 4, 3), "mouse", 1), Mock(), [], 1,
                           10, run_id=run)

    def test_without_handoff_a_step_jev_cant_take_ends_the_request_at_once(self):
        self.planner.handoff = False
        self.planner.steps.next_actions.return_value = Plan(FAILED, [Action("fail", reason="needs new text")], [])
        note = StepContext("open calculadora", DeviceProfile("linux", Rect(0, 0, 4, 3), "mouse", 1), Mock(), [], 2,
                           10, note="the launch landed elsewhere", run_id="a")
        for ctx in (self.ctx("a"), note):
            self.assertEqual(self.planner.next_actions(b"img", ctx).actions[0].type, "fail")
        self.planner.flash.next_actions.assert_not_called()

    def test_a_recovery_note_hands_the_run_to_flash(self):
        ctx = StepContext("open calculadora", DeviceProfile("linux", Rect(0, 0, 4, 3), "mouse", 1), Mock(), [], 2,
                          10, note="the launch landed elsewhere", run_id="a")
        self.assertTrue(self.planner.next_actions(b"img", ctx).reasoning.startswith("Stage 1/1: planned [Jev handed off"))
        self.planner.steps.next_actions.assert_not_called()

    def test_jev_steps_need_no_flash_call(self):
        self.planner.steps.next_actions.return_value = Plan("Jev next step: click", [Action("click", x=.1, y=.1)], [])
        self.assertEqual(self.planner.next_actions(b"img", self.ctx("a")).reasoning, "Jev next step: click")
        self.planner.flash.next_actions.assert_not_called()

    def test_a_step_jev_cant_take_hands_the_run_to_flash(self):
        self.planner.steps.next_actions.return_value = Plan(FAILED, [Action("fail", reason="stuck")], [])
        self.assertIn("[Jev handed off: stuck]", self.planner.next_actions(b"img", self.ctx("a")).reasoning)
        self.planner.next_actions(b"img", self.ctx("a"))
        self.assertEqual(self.planner.steps.next_actions.call_count, 1)  # the rest of that run stays on Flash
        self.planner.next_actions(b"img", self.ctx("b"))
        self.assertEqual(self.planner.steps.next_actions.call_count, 2)  # a new request starts with Jev again


if __name__ == "__main__":
    unittest.main()
