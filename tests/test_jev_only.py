"""Jev's own next step (ported from origin/jev-only) and the Jev-first planner that falls back to Flash."""

import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from brain.agent.executor import Plan, StepContext
from brain.agent.jev_first import JevFirstPlanner
from brain.agent.jev_only import FAILED, LOOK_AGAIN, JevOnlyPlanner
from brain.agent.triage import spans
from brain.types import Action, DeviceProfile, Rect

BOX = {"box": {"left": .28, "right": .32, "top": .28, "bottom": .32}, "confidence": .9}  # small: looked at closer
BIG = {"box": {"left": .2, "right": .4, "top": .2, "bottom": .4}, "confidence": .9}
ON, STRAY = (.3, .3), (.3, .95)  # a zoomed look on the first box's center, and one a row off


class JevOnlyTests(unittest.TestCase):
    def setUp(self):
        self.planner = JevOnlyPlanner(Mock(min_presence=.85))

    def plan(self, goal, complete=.1, next="click", present=.95, text="none", replace=.1, note="",
             fits=.95, app_present=.1, app_name="none", launcher=.1, history=(), os="linux", looks=(ON, ON),
             question=.05, composed=.05, site=.05, submit=.05, left=.9, window=.9, loading=.05, focused=.95,
             target=BOX, run="r", drawing=.05, browser_front=.05, opened_front=.95, input=.05, dotcom=.05, commit=.05, app=(.1, .5, .55, .75), typed=.95,
             lookup=.05, pending=.05, copied=BIG, zoomed=.9, boxes=(None, None), image=b"img", device="mouse", joined="two"):
        step = {"complete": {"noul": complete}, "question": {"noul": question}, "composed": {"noul": composed}, "next": {"choice": next}, "present": {"noul": present},
                "target": target, "text": {"choice": text}, "replace": {"noul": replace}, "fits": {"noul": fits},
                "app_present": {"noul": app_present}, "app_name": {"choice": app_name}, "launcher": {"noul": launcher},
                "app": {"box": dict(zip(("left", "right", "top", "bottom"), app))}, "site": {"noul": site},
                "submit": {"noul": submit}, "browser_named": {"noul": .05}, "browser_front": {"noul": browser_front}, "opened_front": {"noul": opened_front}, "left": {"noul": left},
                "window": {"noul": window}, "loading": {"noul": loading}, "focused": {"noul": focused},
                "drawing": {"noul": drawing}, "input": {"noul": input},
                "dotcom": {"noul": dotcom}, "commit": {"noul": commit}, "typed": {"noul": typed},
                "lookup": {"noul": lookup}, "pending": {"noul": pending}, "copy_present": {"noul": .5}, "copied": copied, "joined": {"choice": joined}}
        self.planner.jev.evaluate.return_value = (step, (400, 300), (400, 300))
        profile = DeviceProfile(os, Rect(0, 0, 400, 300), device, 1)
        ctx = StepContext(goal, profile, Mock(), list(history), 2, 10, note=note, run_id=run)
        with patch("brain.agent.jev_only.refine", side_effect=list(looks)) as self.refine, \
                patch("brain.agent.jev_only.zoom", side_effect=[(p, zoomed, b) for p, b in zip(looks, boxes)]) as self.zoom:
            self.last = self.planner.next_actions(image, ctx)
            return self.last.actions

    def step(self, goal, **answers):
        return self.plan(goal, **answers)[0]

    def test_goal_jev_sees_done_is_done(self):
        self.assertEqual(self.step("open calculator", left=.1).type, "done")
        self.assertEqual(self.step("open calculator", next="stuck", complete=.9).type, "done")  # nothing left to do

    def test_a_line_already_done_when_reached_takes_no_step(self):
        self.assertEqual(self.step("scroll down to the wikipedia link", complete=.88).type, "done")
        self.assertEqual(self.step("close google", complete=.46, next="close_tab", run="s").brief(), "hotkey(CTRL+W)")
        clicked = self.step("click the link", complete=.88, history=["click(300,300)"], run="t", left=.9)
        self.assertNotEqual(clicked.type, "done")  # after its own step only the line-done check ends it
        # Brave in front of "open up chrome": a doubtful line-done (0.58) ends no line before its first step
        self.assertNotEqual(self.step("open up chrome", left=.42, next="click", run="u").type, "done")

    def test_unfinished_goals_are_not_done(self):
        self.assertEqual(self.step("go to wikipedia", complete=.63).type, "click")
        typed = self.plan("search for cats", left=.1, next="enter", history=["type_text('cats')"])
        self.assertEqual([a.brief() for a in typed], ["hotkey(ENTER)"])  # typed text isn't sent yet
        self.assertEqual(self.step("search for cats", left=.1, next="enter", history=["hotkey(ENTER)"]).type, "done")

    def test_text_typed_into_nothing_is_typed_again_into_the_field_clicked_first(self):
        typed = ["type_text('horses')", "hotkey(ENTER)", "skill(wait {'s': 1.0})"]
        self.planner.run_id, self.planner.typed = "r", typed[0]  # Jev typed it
        # sent with Enter: the box may be empty only while the results load
        self.assertEqual(self.step("search for horses", left=.1, typed=.01, history=typed).brief(), typed[2])
        typed.append(typed[2])
        again = self.plan("search for horses", left=.1, next="stuck", submit=.9, typed=.01, history=typed,
                          looks=((.4, .1), (.4, .1)))  # though Jev sees the line done
        self.assertEqual([a.brief() for a in again], ["click(400,100)", "type_text('horses')", "hotkey(ENTER)"])
        self.assertEqual(self.zoom.call_args.args[2]["executed_steps"], [])  # not told the text is in
        self.assertEqual(self.step("search for horses", left=.1, typed=.6, history=typed, run="s").type, "done")
        address = ["hotkey(CTRL+L)", "type_text('gmail')", "hotkey(CTRL+ENTER)"]  # the page it opened replaced it
        self.assertEqual(self.step("go to gmail", left=.1, typed=.01, history=address, run="s").type, "done")
        self.planner.jev.evaluate.reset_mock()
        flash = ["click(400,30)", "type_text('docs.new\\n')"]  # Flash's typing: it checked the field itself
        self.assertEqual(self.step("open a doc", left=.1, typed=.01, history=flash, run="t").type, "done")
        self.assertFalse(any("typed" in c.args[1] for c in self.planner.jev.evaluate.call_args_list))

    def test_a_done_line_hands_the_screen_to_the_next_one_and_a_continued_request_skips_it(self):
        goal = "open up discord\nclose discord"
        self.assertEqual(self.step(goal, left=.1, next="close_app").brief(), "hotkey(ALT+F4)")
        # both lines asked at once: no extra look; each line's questions in one call, all in its one state so far
        self.assertEqual(self.planner.jev.evaluate.call_count, 2)
        self.assertEqual(self.planner.line, 1)
        self.assertEqual(self.step(goal, left=.1, history=["hotkey(ALT+F4)"]).type, "done")
        more = goal + "\nopen a terminal"
        launch = self.step(more, run="s", fits=.1, app_name=str(spans("open a terminal").index("terminal")))
        self.assertEqual(launch.brief(), "skill(open_app {'name': 'terminal'})")  # not discord's lines again

    def test_a_failed_run_continued_in_a_new_run_starts_at_the_line_it_failed_on(self):
        goal = "open up google\nclose google\nsearch up genghis khan"
        self.plan(goal, left=.1, next="close_tab")
        self.assertEqual(self.step(goal, left=.1, complete=.9, question=.9, history=["hotkey(CTRL+W)"]).type, "fail")
        self.plan(goal + "\nUser correction: scroll down", run="s")  # its history starts empty
        self.assertEqual(self.planner.line, 2)  # not "close google" again

    def test_question_sets_on_one_image_in_one_state_share_a_call(self):
        self.planner.jev.evaluate.side_effect = lambda image, questions, **_: ({k: image for k in questions}, 1, 2)
        ctx = StepContext("g", DeviceProfile("linux", Rect(0, 0, 400, 300), "mouse", 1), Mock(), [], 1, 10)
        screen, marked, one, two = b"screen", b"marked", {"goal": "a"}, {"goal": "b"}
        replies = self.planner._evaluate_all(ctx, [
            (screen, "jev", [({"a": {}}, one), ({"b": {}}, two)]), (screen, "typed", [({"c": {}}, one)]),
            (marked, "line-done", [({"a": {}}, one)]), (screen, "other", [({"a": {"x": 1}}, one)])])
        self.assertEqual(replies, [({"a": screen, "b": screen}, 1, 2), ({"c": screen}, 1, 2), ({"a": marked}, 1, 2),
                                   ({"a": screen}, 1, 2)])
        calls = sorted(c.kwargs["label"] for c in self.planner.jev.evaluate.call_args_list)
        self.assertEqual(calls, ["jev", "jev+typed", "line-done", "other"])  # a different question: a call of its own

    def test_next_click_lands_on_the_target(self):
        action = self.step("open calculator, then click 7")
        self.assertEqual((action.type, round(action.x, 2), round(action.y, 2)), ("click", .3, .3))

    def test_the_current_line_is_the_one_jev_is_on(self):
        self.step("open up youtube\nUser correction: search for horses", left=.1, next="click", present=.4)
        self.assertEqual(self.planner.current_line, "search for horses")
        ctx = StepContext(self.planner.goal, DeviceProfile("linux", Rect(0, 0, 400, 300), "mouse", 1), Mock(), [], 2, 10,
                          run_id="r")
        self.assertEqual(self.planner.line_done(b"img", ctx).actions[0].type, "done")  # Flash finished its last line

    def test_the_rest_of_a_sentence_said_after_a_pause_joins_its_line(self):
        goal = "Close this browser\nUser correction: browser."
        self.assertEqual(self.step(goal, joined="one", left=.9, next="close_app").brief(), "hotkey(ALT+F4)")
        self.assertEqual(self.planner.current_line, "Close this browser browser.")
        # once its line is done the request is: the rest is no request of its own
        self.assertEqual(self.step(goal, left=.3, next="click", history=["hotkey(ALT+F4)"]).type, "done")
        rests = [c for c in self.planner.jev.evaluate.call_args_list if "joined" in c.args[1]]
        self.assertEqual(len(rests), 1)  # asked once
        self.assertEqual(rests[0].kwargs["state"], {"goal": "Close this browser\nbrowser."})
        # a request of its own is a line of its own
        self.step("open up youtube\nUser correction: search for horses", left=.1, next="click", run="s")
        self.assertEqual(self.planner.current_line, "search for horses")

    def test_a_zoom_boxing_a_column_of_controls_clicks_the_row_of_the_first_look(self):
        # LockDown Browser above Chrome on a desktop: both zooms boxed the two icons (2026-10-20)
        first = {"box": {"left": .02, "right": .07, "top": .68, "bottom": .76}, "confidence": .2}
        column = (.06, .5658, .0989, .7354)
        action = self.step("open up a browser", next="double_click", target=first, looks=((.0794, .6506), (.0794, .6506)),
                           boxes=(column, column))
        self.assertEqual((round(action.x, 3), round(action.y, 3)), (.079, .708))  # Chrome's, not the gap above it

    def test_unclear_target_fails_instead_of_guessing(self):
        self.assertEqual(self.step("click the other one", present=.4).type, "fail")
        self.assertEqual(self.step("click the other one", looks=(None, None)).type, "fail")  # lost when zoomed
        self.assertTrue(self.last.needs_planner)  # Flash takes that one step

    def test_a_target_not_visible_after_a_step_is_looked_for_again_twice(self):
        goal = "search youtube for horses"
        self.assertEqual(self.step(goal, present=.4, history=["skill(open_app {'name': 'firefox'})"]).brief(),
                         "skill(wait {'s': 1.0})")
        waited = ["skill(open_app {'name': 'firefox'})", "skill(wait {'s': 1.0})", "skill(wait {'s': 1.0})"]
        self.assertEqual(self.step(goal, present=.4, history=waited).type, "fail")
        page = cv2.imencode(".jpg", np.full((300, 400, 3), 200, np.uint8))[1].tobytes()
        clicked = ["click(300,300)"]
        self.assertEqual(self.step(goal, present=.4, history=clicked, run="d", image=page).brief(), waited[1])
        # the page didn't change while it was looked at again: it is drawn, and another look won't find it
        self.assertEqual(self.step(goal, present=.4, history=clicked + waited[1:2], run="d", image=page).type, "fail")

    def test_a_paste_whose_place_jev_cant_see_goes_where_the_keys_go_once_looked_for(self):
        goal, opened = "paste it in", ["click(684,713)", LOOK_AGAIN.brief()]
        self.assertEqual(self.step(goal, next="paste", present=.39, focused=.2, history=opened).brief(), opened[1])
        pasted = self.plan(goal, next="paste", present=.39, focused=.2, history=opened + opened[1:])
        self.assertEqual([a.brief() for a in pasted], ["hotkey(CTRL+V)"])

    def test_a_zoom_that_strays_is_outvoted(self):
        for looks in ((ON, STRAY), (STRAY, ON)):
            action = self.step("open calculator, then click 7", looks=looks)
            self.assertEqual((round(action.x, 2), round(action.y, 2)), (.3, .3))

    def test_a_nearly_sure_target_is_clicked_only_where_a_zoom_lands_in_its_box(self):
        for looks in ((ON, STRAY), (STRAY, ON), (ON, None)):
            action = self.step("search for horse videos", present=.7, looks=looks)
            self.assertEqual((action.type, round(action.y, 2)), ("click", .3))
        self.assertEqual(self.step("search for horse videos", present=.7, looks=(STRAY, STRAY)).type, "fail")

    def test_a_zoom_boxing_the_region_around_the_other_is_dropped(self):
        logo, field = (.52, .25), (.52, .41)  # Chrome's new tab: the logo and search box, and the search box alone
        for looks, boxes in (((logo, field), ((.24, .05, .81, .45), (.24, .37, .8, .45))),
                             ((field, logo), ((.24, .37, .8, .45), (.24, .05, .81, .45)))):
            bar = {"box": {"left": .12, "right": .87, "top": .07, "bottom": .11}, "confidence": .9}  # first look's box
            action = self.step("search up genghis khan", target=bar, looks=looks, boxes=boxes)
            self.assertEqual(round(action.y, 2), .41)
        self.assertEqual(round(self.step("click it", looks=(ON, STRAY), boxes=((.2, .2, .4, .4), None)).y, 2), .3)

    def test_a_zoom_beside_a_doubted_box_is_taken_once_a_look_again_saw_it_too(self):
        note, box = (.345, .3), (.34, .29, .35, .31)  # Wikipedia's [b], just right of the birth date Jev boxed
        wait = self.step("click the date of birth reference", present=.7, looks=(note, note), boxes=(box, box),
                         history=["click(1,1)"])
        self.assertEqual(wait.brief(), "skill(wait {'s': 1.0})")
        click = self.step("click the date of birth reference", present=.7, looks=(note, note), boxes=(box, box),
                          history=["click(1,1)", wait.brief()], run="s")
        self.assertEqual((round(click.x, 3), round(click.y, 2)), note)
        far = (.29, .9, .31, .99)
        self.assertEqual(self.step("click it", present=.7, looks=(STRAY, STRAY), boxes=(far, far),
                                   history=["click(1,1)", wait.brief()], run="t").type, "skill")

    def test_two_zooms_that_agree_move_a_sure_click(self):
        action = self.step("open a new tab and search cats", looks=(STRAY, STRAY))
        self.assertEqual((action.type, action.y), ("click", .95))

    def test_a_wide_target_is_zoomed_whole(self):
        wide = {"box": {"left": .05, "right": .55, "top": .28, "bottom": .32}, "confidence": .9}  # a search box
        self.step("search for cats", target=wide, looks=((.3, .3), (.3, .3)))
        self.assertGreater(self.zoom.call_args.args[5], .5)

    def test_a_click_is_judged_done_on_the_screen_it_was_planned_on(self):
        jev = self.planner.jev

        def evaluate(image, questions, **_):  # Jev sees the line done only with the click marked on its screen
            answers, size, padded = jev.evaluate.return_value
            return answers | ({"left": {"noul": .2}} if image == b"marked" else {}), size, padded

        goal = "click the third video"
        with patch("brain.agent.jev_only.mark", return_value=b"marked") as marked:
            jev.evaluate.side_effect = evaluate
            click = self.step(goal, target=BIG).brief()
            self.assertEqual(self.step(goal, target=BIG, history=[click]).type, "done")
            self.assertEqual([round(v, 2) for v in marked.call_args.args[1]], [.3, .3])
            self.step(goal, target=BIG, run="w")  # judged so after a look again too: the page may still be drawing
            self.assertEqual(self.step(goal, target=BIG, history=[click, LOOK_AGAIN.brief()], run="w").type, "done")
            for later in ({"history": [click, "hotkey(ENTER)"]},  # not the step just taken
                          {"history": [click], "note": "Screen unchanged; try something different."}):
                self.step(goal, target=BIG, run=str(later))
                self.assertNotEqual(self.step(goal, target=BIG, run=str(later), **later).type, "done")
            self.assertEqual(marked.call_count, 2)
            # Flash's click, on the screen Jev handed it
            self.assertEqual(self.step(goal, question=.9, complete=.9, history=["hotkey(CTRL+T)"], run="f").type,
                             "fail")
            self.assertEqual(self.step(goal, history=["hotkey(CTRL+T)", "click(500,500)"], run="f").type, "done")
            self.assertEqual(marked.call_args.args[1], (.5, .5))

    def test_a_click_ends_its_line_unless_jev_is_fairly_sure_something_is_left(self):
        goal = "click the wikipedia link"
        with patch("brain.agent.jev_only.mark", return_value=b"marked"):
            click = self.step(goal).brief()
            self.assertEqual(self.step(goal, history=[click], left=.5).type, "done")
            self.step(goal, run="s")
            self.assertNotEqual(self.step(goal, history=[click], left=.85, run="s").type, "done")
            self.step(goal, run="g")
            self.assertNotEqual(self.step(goal, history=[click], left=.6, run="g").type, "done")  # Docs' gallery
            menu = self.step("copy the title", next="right_click", run="r").brief()
            self.assertNotEqual(self.step("copy the title", history=[menu], left=.55, run="r").type, "done")
        self.assertNotEqual(self.step(goal, history=[click], left=.55, run="t").type, "done")  # no click of its own

    def test_a_doubtful_click_is_judged_only_on_the_screen_it_led_to(self):
        goal = "click the date of birth reference"
        with patch("brain.agent.jev_only.mark", return_value=b"marked") as marked:
            click = self.step(goal, zoomed=.52).brief()
            self.assertNotEqual(self.step(goal, history=[click], left=.55).type, "done")
            self.assertFalse(marked.called)  # no look at the screen it was planned on
        # read done on the screen it led to, yet Jev is sure of another click for the line: that click, else done
        click = self.step(goal, zoomed=.52, run="s").brief()
        again = self.step(goal, history=[click], left=.1, run="s", looks=((.6, .6), (.6, .6)))
        self.assertEqual(again.brief(), "click(600,600)")
        click = self.step(goal, zoomed=.52, run="t").brief()
        self.assertEqual(self.step(goal, history=[click], left=.1, run="t", zoomed=.52).type, "done")
        # a click sure enough to place but not to settle its line is held open too; a sure one isn't
        with patch("brain.agent.jev_only.mark", return_value=b"marked"):
            click = self.step(goal, zoomed=.65, run="w").brief()
            again = self.step(goal, history=[click], left=.1, run="w", looks=((.6, .6), (.6, .6)))
            self.assertEqual(again.brief(), "click(600,600)")
            click = self.step(goal, run="x").brief()
            self.assertEqual(self.step(goal, history=[click], left=.1, run="x").type, "done")
        # zooms that half see it: a look again first, then the click, doubted
        wait = self.step(goal, zoomed=.45, history=["click(1,1)"], run="u").brief()
        self.assertEqual(wait, "skill(wait {'s': 1.0})")
        click = self.step(goal, zoomed=.45, history=["click(1,1)", wait], run="u")
        self.assertEqual(click.brief(), "click(300,300)")
        self.assertEqual(self.planner.doubted, click.brief())
        self.assertEqual(self.step(goal, zoomed=.2, history=["click(1,1)", wait], run="v").type, "skill")

    def test_a_big_target_is_clicked_without_a_closer_look(self):
        action = self.step("click the third video", target=BIG, looks=(STRAY, STRAY))
        self.assertEqual((action.type, round(action.x, 2), round(action.y, 2)), ("click", .3, .3))

    def test_typed_text_is_a_run_of_the_goals_words(self):
        goal = 'type "hello world", then press enter'
        action = self.step(goal, next="type", text=str(spans(goal, 250).index("hello world")),
                           history=["click(300,300)"])
        self.assertEqual((action.type, action.text), ("type_text", "hello world"))
        self.assertEqual(self.step("write a poem", next="type").type, "fail")
        self.assertTrue(self.last.needs_planner)  # Flash writes it
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

    def test_step_with_no_visible_effect_is_tried_once_more_then_fails(self):
        unchanged = "Screen unchanged; try something different."
        wait = LOOK_AGAIN.brief()
        self.assertEqual(self.step("open it", note=unchanged, history=["click(300,300)"]).brief(), wait)
        self.assertEqual(self.step("open it", history=["click(300,300)", wait]).brief(), "click(300,300)")  # the retry
        self.assertEqual(self.step("open it", note=unchanged, history=["click(300,300)", wait, "click(300,300)"]).type,
                         "fail")
        # nor does it finish its line, however done the screen reads
        self.assertEqual(self.step("open it", note=unchanged, history=["click(600,300)"], left=.1, run="d").brief(),
                         wait)
        # a paste that changed nothing clicks its place first, at once
        pasted = self.plan("paste it in", next="paste", note=unchanged, history=["hotkey(CTRL+V)"], left=.1, run="e")
        self.assertEqual([a.brief() for a in pasted], ["click(300,300)", "hotkey(CTRL+V)"])
        self.assertEqual(self.step("paste it in", next="paste", note=unchanged, history=["hotkey(CTRL+V)", *[
            a.brief() for a in pasted]], run="e").type, "fail")
        # a copy that changed nothing selected nothing: it copies again at once, though Jev reads it done
        copied = ["skill(select {'x': 0.405, 'y': 0.3})", "hotkey(CTRL+C)"]
        again = self.plan("copy the header\npaste it in", next="paste", note=unchanged, history=copied, left=.1, run="f")
        self.assertEqual([a.brief() for a in again], copied)
        self.assertEqual(self.step("copy the header\npaste it in", next="paste", note=unchanged, history=copied * 2,
                                   run="f").type, "fail")

    def test_the_look_after_a_step_with_no_visible_effect_finishes_nothing_while_the_screen_stays(self):
        page = np.full((300, 400, 3), 200, np.uint8)
        still, moved = (cv2.imencode(".jpg", screen)[1].tobytes() for screen in (page, cv2.rectangle(
            page.copy(), (0, 0), (400, 150), (40, 40, 40), -1)))
        unchanged, wait = "Screen unchanged; try something different.", LOOK_AGAIN.brief()
        for run, after, step in (("a", still, "click(300,300)"), ("b", moved, "done")):
            self.assertEqual(self.step("open it", note=unchanged, history=["click(300,300)"], run=run,
                                       image=still).brief(), wait)
            self.assertEqual(self.step("open it", history=["click(300,300)", wait], left=.1, run=run,
                                       image=after).brief(), step)

    def test_a_question_or_nonsense_is_left_to_flash(self):
        self.assertEqual(self.step("what is this page about?", complete=.9, question=.9).type, "fail")
        self.assertEqual(self.step("asdkjh qwe zxcv", fits=.1, next="stuck").type, "fail")  # no launcher
        self.assertEqual(self.step("search up genghis khan", question=.6, run="s").type, "click")  # a step to take

    def test_the_same_input_is_never_repeated_right_after_it_ran(self):
        waited = [LOOK_AGAIN.brief()] * 2  # it looks again first: the screen may not show the step yet
        self.assertEqual(self.step("open calculator, then click 7", history=["click(300,300)"]).type, "skill")
        self.assertEqual(self.step("open calculator, then click 7", history=["click(300,300)", *waited]).type, "fail")
        self.assertEqual(self.step("open calculator, then click 7", history=["click(600,300)"]).type, "click")
        typed = self.plan("search for cats", next="enter", history=["type_text('cats')", "hotkey(ENTER)", *waited])
        self.assertEqual(typed[0].type, "fail")
        sent = self.plan("open a template", next="enter", history=["hotkey(CTRL+L)", "type_text('docs.new\\n')"])
        self.assertEqual(sent[0].brief(), LOOK_AGAIN.brief())
        sent = self.plan("open a template", next="enter", history=["type_text('docs.new\\n')", *waited])
        self.assertEqual(sent[0].type, "fail")

    def test_a_new_lines_click_where_the_last_line_clicked_is_no_repeat(self):
        self.assertEqual(self.step("go to google docs\ntype in hello", left=.1, history=["click(300,300)"]).type, "click")

    def test_the_request_itself_is_never_typed(self):
        goal = "write a short note about my day"
        self.assertEqual(self.step(goal, next="type", text=str(spans(goal, 250).index(goal))).type, "fail")
        name = "Type Nicholas Rebelloza."  # the command's dictated words, one word short of the line
        typed = self.plan(name, next="type", text=str(spans(name, 250).index("Nicholas Rebelloza")), run="s")
        self.assertEqual(typed[-1].brief(), "type_text('Nicholas Rebelloza')")

    def test_copy_selects_the_boxed_text_and_paste_goes_where_the_focus_is(self):
        wide = {"box": {"left": .2, "right": .6, "top": .28, "bottom": .32}}
        boxes = iter([BIG, wide])  # two looks at what is copied: the select goes just past the further box's end

        def evaluate(image, questions, **_):
            answers, size, padded = self.planner.jev.evaluate.return_value
            return answers | ({"copied": next(boxes)} if "copied" in questions else {}), size, padded

        self.planner.jev.evaluate.side_effect = evaluate
        copied = self.plan("copy the header\npaste it in", next="copy", target=BOX, present=.25, looks=(STRAY, STRAY))  # doubted
        self.assertEqual([a.brief() for a in copied], ["skill(select {'x': 0.605, 'y': 0.3})", "hotkey(CTRL+C)"])
        asked = [c.kwargs["state"]["goal"] for c in self.planner.jev.evaluate.call_args_list if c.kwargs["label"] == "copy"]
        self.assertEqual(asked, ["copy the header\npaste it in"] * 2)  # with the later line that names it
        boxes = iter([BIG, wide])
        rowed = self.plan("copy the header\npaste it in", next="copy", looks=((.5, .31), ON), run="z")
        self.assertEqual(rowed[0].brief(), "skill(select {'x': 0.605, 'y': 0.31})")  # the closer look's row, in the box
        self.planner.jev.evaluate.side_effect = None
        pasted = self.plan("paste it in", next="paste", run="s")
        self.assertEqual([a.brief() for a in pasted], ["hotkey(CTRL+V)"])
        unfocused = self.plan("paste it in", next="paste", focused=.2, target=BIG, run="t")
        self.assertEqual([a.brief() for a in unfocused], ["click(300,300)", "hotkey(CTRL+V)"])
        # The clipboard doesn't show: the line isn't done until its copy or paste is taken
        self.assertEqual(self.step("copy the header", next="copy", left=.1, history=["click(53,179)"], run="v").type,
                         "skill")
        self.assertEqual(self.step("copy the header", next="paste", left=.1, history=["hotkey(Control_L+c)"],
                                   run="w").type, "done")
        opening = self.step("paste it in", next="paste", focused=.2, history=["click(500,300)"], run="u")
        self.assertEqual(opening.brief(), "skill(wait {'s': 1.0})")  # the click's document may still be opening

    def test_a_line_begun_on_enter_looks_again_before_it_scrolls(self):
        goal = "search for horses\nUser correction: click the third video"
        self.planner.run_id, self.planner.line, self.planner.since = "r", 1, 1
        self.assertEqual(self.step(goal, next="scroll_down", history=["hotkey(ENTER)"]).brief(), LOOK_AGAIN.brief())
        self.planner.line, self.planner.since = 1, 1
        scrolled = self.plan(goal, next="scroll_down", history=["hotkey(ENTER)", "scroll(5)"])
        self.assertEqual(scrolled[-1].brief(), "scroll(5)")  # its own steps came after the Enter

    def test_a_scroll_first_moves_the_pointer_over_what_it_scrolls(self):
        actions = self.plan("scroll down to the comments", next="scroll_down")
        self.assertEqual([a.brief() for a in actions], ["skill(hover {'x': 0.3, 'y': 0.3})", "scroll(5)"])
        unsure = self.plan("scroll down to the comments", next="scroll_down", present=.2, run="s")
        self.assertEqual(unsure[0].brief(), "skill(hover {'x': 0.5, 'y': 0.5})")  # no box to go by: the middle

    def test_text_the_request_does_not_give_word_for_word_is_left_to_flash(self):
        self.assertEqual(self.step("work out 12 times 9", composed=.9, next="type").type, "fail")
        self.assertTrue(self.last.needs_planner)  # text to write goes to Flash even without hand-off
        self.assertEqual(self.step("open camera", composed=.9).type, "click")  # a tap needs no text

    def test_a_page_the_browser_is_still_going_to_is_waited_for_but_not_its_keys(self):
        self.assertEqual(self.step("search for horses", pending=.7, history=["click(400,200)"]).brief(),
                         LOOK_AGAIN.brief())
        self.assertEqual(self.step("close this tab", run="k", pending=.7, next="close_tab").brief(), "hotkey(CTRL+W)")

    def test_a_kind_of_app_read_as_a_website_opens_its_app_when_the_browser_route_dead_ends(self):
        goal, kind = "open up a spreadsheet", str(spans("open up a spreadsheet").index("spreadsheet"))
        routed = dict(site=.92, browser_front=1, app_name=kind)
        self.step(goal, run="s", **routed, next="new_tab")
        unseen = self.step(goal, run="s", next="stuck", question=.9, complete=.9, history=["hotkey(CTRL+T)"])
        self.assertEqual(unseen.brief(), "skill(open_app {'name': 'spreadsheet'})")  # whatever dead end
        self.assertEqual(self.step(goal, **routed, next="new_tab").brief(), "hotkey(CTRL+T)")
        typed = dict(composed=.6, next="type", history=["hotkey(CTRL+T)"])
        self.assertEqual(self.step(goal, **typed).brief(), "skill(open_app {'name': 'spreadsheet'})")
        self.assertEqual(self.step(goal, **typed | {"history": ["hotkey(CTRL+T)", self.last.actions[0].brief()]}).type,
                         "fail")  # once
        # a site's own name is no app to search the launcher for
        self.step("go to youtube", run="y", site=.92, browser_front=1, dotcom=.9, next="new_tab",
                  app_name=str(spans("go to youtube").index("youtube")))
        self.assertEqual(self.step("go to youtube", run="y", **typed).type, "fail")

    def test_reasoning_fails_over(self):
        self.assertEqual(self.step("summarize this page", next="stuck").type, "fail")

    def test_a_stuck_step_mid_request_looks_again_before_it_fails(self):
        clicked = ["click(574,977)"]
        self.assertEqual(self.step("open up youtube", next="stuck", history=clicked).brief(), "skill(wait {'s': 1.0})")
        looked = clicked + ["skill(wait {'s': 1.0})"] * 2
        self.assertEqual(self.step("open up youtube", next="stuck", history=looked).type, "fail")

    def test_app_in_front_that_fits_takes_the_step_and_ends_routing(self):
        self.assertEqual(self.step("find a video about horses").type, "click")
        self.assertEqual(self.step("find a video about horses", fits=.1).type, "click")  # routed once per run

    def test_visible_suited_app_is_switched_to(self):
        goal = "find a video about horses"
        action = self.step(goal, fits=.1, app_present=.95, app_name=str(spans(goal, 250).index("video")))
        self.assertEqual((action.type, round(action.x, 2), round(action.y, 2)), ("click", .3, .65))

    def test_a_small_app_icon_is_looked_at_closer_before_it_is_clicked(self):
        goal = "open a terminal"
        icon = dict(fits=.1, app_present=.95, app_name=str(spans(goal).index("terminal")), app=(.44, .46, .96, .99))
        action = self.step(goal, **icon, looks=((.42, .98),))
        self.assertEqual((action.type, action.x, action.y), ("click", .42, .98))  # the zoom's icon, not the box's
        self.assertEqual(self.step(goal, **icon, looks=(None,)).brief(), "skill(open_app {'name': 'terminal'})")

    def test_a_line_seen_not_done_is_not_done_without_input_of_its_own(self):
        goal = "open up discord\nclose discord"
        self.planner.line, self.planner.since, self.planner.run_id = 1, 1, "r"  # its look so far: a wait
        self.planner.seen = True
        waited = ["click(475,977)", "skill(wait {'s': 1.0})"]
        self.assertEqual(self.step(goal, left=.1, next="close_app", history=waited).brief(), "hotkey(ALT+F4)")
        self.assertEqual(self.step(goal, left=.1, history=waited + ["hotkey(ALT+F4)"]).type, "done")
        self.planner.since = 2  # its one plan so far discarded unrun
        self.assertEqual(self.step(goal, left=.1, next="close_app", history=waited).brief(), "hotkey(ALT+F4)")

    def test_words_said_mid_run_are_shown_to_jev_as_requests(self):
        self.step("open up libre office\nUser correction: add hello to b1", left=.1)
        shown = [c.kwargs["state"]["goal"] for c in self.planner.jev.evaluate.call_args_list]
        self.assertIn("add hello to b1", shown)
        self.assertFalse(any("correction" in goal for goal in shown))

    def test_jev_sees_which_line_took_the_steps_before_this_one(self):
        self.planner.line, self.planner.since, self.planner.run_id, self.planner.seen = 1, 2, "r", True
        goal, history = "open up youtube\nsearch for horses\nclick the third video", ["skill(wait {'s': 1.0})", "click(471,196)"]
        self.step(goal, next="fill", text="1", input=.99, history=history + ["click(400,100)"])
        seen = {c.kwargs["state"]["goal"]: c.kwargs["state"]["executed_steps"]
                for c in self.planner.jev.evaluate.call_args_list if "next" in c.args[1]}
        self.assertEqual(seen["search for horses"], ["skill(wait {'s': 1.0}) for \"open up youtube\"",
                                                     "click(471,196) for \"open up youtube\"", "click(400,100)"])
        self.assertEqual(seen["click the third video"], ["click(400,100) for \"search for horses\""])

    def test_a_line_reached_already_done_is_done(self):
        self.assertEqual(self.step("open up discord", left=.1, next="stuck").type, "done")

    def test_the_app_icon_just_clicked_is_waited_for_not_clicked_again(self):
        goal = "close discord"
        # boxed a little off it: the closer look finds the icon just clicked
        icon = dict(fits=.1, app_present=.95, app_name=str(spans(goal).index("discord")), app=(.42, .45, .96, .99))
        clicked = ["click(475,977)"]
        self.assertEqual(self.step(goal, **icon, looks=((.478, .968),), history=clicked).brief(),
                         "skill(wait {'s': 1.0})")
        looked = clicked + ["skill(wait {'s': 1.0})"] * 6
        self.assertEqual(self.step(goal, **icon, looks=((.478, .968),), history=looked).type, "fail")
        on = dict(icon, app=(.46, .49, .96, .99))  # boxed right on the click: no closer look, which found none
        self.assertEqual(self.step(goal, **on, looks=(None,), history=clicked).brief(), "skill(wait {'s': 1.0})")

    def test_the_suited_app_in_front_is_not_switched_to(self):
        goal = "click the third video"
        front = dict(fits=.1, app_present=.95, site=.9, app=(.01, .99, .04, .98),  # the browser, its results loading
                     history=["hotkey(ENTER)"])
        self.assertEqual(self.step(goal, **front, present=.1).brief(), "skill(wait {'s': 1.0})")
        action = self.step(goal, **front)
        self.assertEqual((action.type, round(action.x, 2), round(action.y, 2)), ("click", .3, .3))  # the step's own

    def test_no_suited_app_is_opened_then_looked_for_in_the_launcher_waits_once_then_fails(self):
        goal = "find a video about horses"
        named = str(spans(goal, 250).index("video"))
        self.assertEqual(self.step(goal, fits=.1, app_name=named).brief(), "skill(open_app {'name': 'video'})")
        self.assertTrue(self.last.blind)  # launched by name whatever plays behind it
        opened = ["skill(open_app {'name': 'video'})"]
        # what it launched may still be coming: LibreOffice drew after a look at the bare desktop
        self.assertEqual(self.step(goal, fits=.1, app_name=named, history=opened).brief(), LOOK_AGAIN.brief())
        opened += [LOOK_AGAIN.brief()] * 2
        self.assertEqual(self.step(goal, fits=.1, app_name=named, history=opened).brief(), "hotkey(GUI)")
        self.assertEqual(self.step(goal, fits=.1, app_name=named, history=opened + ["hotkey(GUI)"]).brief(),
                         "skill(wait {'s': 3.0})")
        self.assertEqual(self.step(goal, fits=.1, app_name=named,
                                   history=opened + ["hotkey(GUI)", "skill(wait {'s': 3.0})"]).type, "fail")

    def test_a_launch_whose_launcher_never_showed_is_not_tried_again(self):
        goal = "find a video about horses"
        named = str(spans(goal, 250).index("video"))
        self.assertEqual(self.step(goal, fits=.1, app_name=named).brief(), "skill(open_app {'name': 'video'})")
        self.planner.acknowledge(False)  # the skill raised: no launcher showed, so it isn't in the history
        self.assertEqual(self.step(goal, fits=.1, app_name=named).brief(), "skill(wait {'s': 3.0})")

    def test_launcher_is_searched_for_the_app_the_goal_names_before_an_icon_is_clicked(self):
        goal = "open up camera and take a picture"
        search = dict(fits=.1, app_name=str(spans(goal, 250).index("camera")), launcher=.9, app_present=.95,
                      history=["hotkey(GUI)"])
        self.assertEqual(self.step(goal, **search).brief(), "type_text('camera')")  # not the camera-shaped icon
        typed = ["hotkey(GUI)", "type_text('camera')"]
        self.assertEqual(self.step(goal, **search | {"history": typed}).type, "click")  # then the result

    def test_a_website_goes_to_a_visible_browser_or_the_launchers_list_never_its_search(self):
        goal = "open up youtube"
        site = dict(fits=.1, site=.9, app_name=str(spans(goal).index("youtube")))
        action = self.step(goal, **site, app_present=.95)
        self.assertEqual((action.type, round(action.x, 2), round(action.y, 2)), ("click", .3, .65))
        self.assertEqual(self.step(goal, **site).brief(), "skill(open_app {'name': 'web browser'})")  # not 'youtube'
        opened = ["skill(open_app {'name': 'web browser'})", "hotkey(GUI)"]
        self.assertEqual(self.step(goal, **site, launcher=.9, history=opened).brief(), "type_text('web browser')")
        self.assertEqual(self.step(goal, **site, browser_front=.99, run="s").type, "click")  # in the browser in front
        phone = dict(site, device="touch", app_present=.95)
        self.assertEqual(self.step(goal, **phone, browser_front=.99, run="t").type, "click")
        self.assertEqual(self.step(goal, **phone, run="u").brief(), "skill(open_app {'name': 'web browser'})")  # keys

    def test_an_open_launcher_listing_the_app_leaves_its_line_to_click(self):
        goal, opened = "open the calculator", ["skill(open_app {'name': 'calculator'})"]
        self.assertEqual(self.step(goal, left=.28, launcher=.99, history=opened).type, "click")
        self.assertEqual(self.step(goal, left=.28, history=opened, run="s").type, "done")

    def test_each_line_routes_again(self):
        first = "open up discord"
        self.plan(first, app_name=str(spans(first).index("discord")))  # fits: routing ends
        goal = first + "\nopen a terminal"
        opened = ["skill(open_app {'name': 'discord'})"]
        self.assertEqual(self.step(goal, left=.1, fits=.1, app_name=str(spans("open a terminal").index("terminal")),
                                   history=opened).brief(), "skill(open_app {'name': 'terminal'})")

    def test_a_line_naming_no_app_reopens_the_last_one_only_with_no_window_open(self):
        goal = "open up chrome and close out the tab\nmake a new tab"
        closed = ["skill(open_app {'name': 'chrome'})", "hotkey(CTRL+W)"]
        self.assertEqual(self.step(goal, left=.1, fits=.1, window=.1, next="new_tab", history=closed).brief(),
                         "skill(open_app {'name': 'chrome'})")
        self.setUp()  # a planner that hasn't seen the first line done
        self.assertEqual(self.step(goal, left=.1, fits=.1, next="new_tab", history=closed, run="s",
                                   browser_front=.99).brief(), "hotkey(CTRL+T)")  # another browser window in front

    def test_a_line_naming_no_app_with_another_app_in_front_reopens_the_last_app(self):
        goal = "open up chrome and close out the tab\nmake a new tab"
        closed = ["skill(open_app {'name': 'chrome'})", "hotkey(CTRL+W)"]
        # the spreadsheet left in front fits the line (its sheet tabs), and its add-sheet button is the step
        plan = self.step(goal, left=.1, fits=.97, next="click", opened_front=.01, history=closed)
        self.assertEqual(plan.brief(), "skill(open_app {'name': 'chrome'})")
        self.assertTrue(self.last.blind)

    def test_a_failed_request_continued_in_a_new_run_reopens_the_app_it_opened(self):
        goal = "open up chrome and close out the tab"
        self.plan(goal, history=["skill(open_app {'name': 'chrome'})", "hotkey(CTRL+W)"])
        closed = dict(fits=.1, window=.1, next="new_tab")
        self.assertEqual(self.step(goal + "\nmake a new tab", run="s", left=.1, **closed).brief(),
                         "skill(open_app {'name': 'chrome'})")  # its history starts empty
        self.assertEqual(self.step("make a new tab", run="t", **closed).brief(), "hotkey(CTRL+T)")  # a new request

    def test_address_goes_to_the_site_in_the_browser_in_front(self):
        goal = "open chrome and go to google docs"
        actions = self.plan(goal, next="address", text=str(spans(goal).index("google docs")))
        self.assertEqual([a.brief() for a in actions], ["hotkey(CTRL+L)", "type_text('google docs')", "hotkey(ENTER)"])
        self.assertTrue(self.last.blind)  # the address bar takes it whatever the page draws meanwhile
        self.assertEqual(self.step("go to email", next="address").type, "click")  # no words: the page's link
        goal = "open up a google doc template"  # describes a page: its words, typed or clicked, would search the web
        for text in (str(spans(goal).index("google doc template")), "none"):
            self.assertEqual(self.step(goal, next="address", text=text, lookup=.65).type, "fail")
            self.assertTrue(self.last.needs_planner)  # Flash finds the page
        goal = "search up genghis khan"  # a search the line asks for is typed there
        self.assertEqual(self.plan(goal, next="address", text=str(spans(goal).index("genghis khan")), lookup=.6,
                                   submit=.9)[-1].brief(), "hotkey(ENTER)")

    def test_a_site_named_by_its_dotcom_word_is_opened_without_a_search(self):
        goal = "open up youtube"
        youtube = str(spans(goal).index("youtube"))
        self.assertEqual(self.plan(goal, next="address", text=youtube, dotcom=.9)[-1].brief(), "hotkey(CTRL+ENTER)")
        for other in ({"dotcom": .1}, {"os": "macos"}):
            self.assertEqual(self.plan(goal, **{"next": "address", "text": youtube, "dotcom": .9} | other)[-1].brief(),
                             "hotkey(ENTER)")
        goal = "go to google docs"  # no web address has a space
        self.assertEqual(self.plan(goal, next="address", text=str(spans(goal).index("google docs")), dotcom=.9)[-1]
                         .brief(), "hotkey(ENTER)")

    def test_a_search_is_typed_and_sent_in_one_step(self):
        goal = "search for horses"
        actions = self.plan(goal, next="type", text=str(spans(goal).index("horses")), submit=.9,
                            history=["click(300,300)"])
        self.assertEqual([a.brief() for a in actions], ["type_text('horses')", "hotkey(ENTER)"])

    def test_a_click_on_a_box_the_line_types_into_types_in_the_same_step(self):
        goal = "search for horses"
        horses = str(spans(goal).index("horses"))
        actions = self.plan(goal, next="click", text=horses, input=.95, submit=.9)
        self.assertEqual([a.type for a in actions], ["click", "type_text", "hotkey"])
        for unsure in ({"input": .7}, {"text": "none"}):  # a tile that opens a document, or nothing to type
            self.assertEqual([a.type for a in self.plan(goal, **{"next": "click", "text": horses} | unsure)], ["click"])

    def test_text_typed_into_a_cell_is_entered_in_the_same_step(self):
        goal = "add hello to b1"
        hello = str(spans(goal).index("hello"))
        actions = self.plan(goal, next="click", text=hello, input=.95, commit=.9)
        self.assertEqual([a.type for a in actions], ["click", "type_text", "hotkey"])
        self.assertEqual([a.type for a in self.plan(goal, next="click", text=hello, input=.95, commit=.6)],
                         ["click", "type_text"])  # a terminal in front of a spreadsheet: Enter would run it

    def test_typed_words_are_erased_word_by_word(self):
        goal = "type in hello world, delete hello world"
        actions = self.plan(goal, next="erase", text=str(spans(goal).index("hello world")),
                            history=["type_text('hello world')"])
        self.assertEqual([a.brief() for a in actions], ["hotkey(CTRL+BACKSPACE)"] * 2)
        self.assertEqual(self.step(goal, next="erase", os="macos").type, "fail")  # no text: nothing to count
        erased = self.plan("delete hello", next="erase", history=["type_text('hello')"])
        self.assertEqual([a.brief() for a in erased], ["hotkey(CTRL+BACKSPACE)"])  # the text last typed

    def test_tabs_and_apps_close_by_keyboard(self):
        self.assertEqual(self.step("make a new tab", next="new_tab").brief(), "hotkey(CTRL+T)")
        self.assertEqual(self.step("close out the tab", next="close_tab", os="macos").brief(), "hotkey(GUI+W)")
        self.assertEqual(self.step("close discord", next="close_app").brief(), "hotkey(ALT+F4)")
        self.assertEqual(self.step("close discord", next="close_app", os="macos").brief(), "hotkey(GUI+Q)")

    def test_an_app_still_opening_is_waited_for_even_on_the_first_screen(self):
        self.assertEqual(self.step("open up discord", next="stuck", loading=.9).brief(), "skill(wait {'s': 1.0})")
        waited = ["skill(wait {'s': 1.0})"] * 6
        self.assertEqual(self.step("open up discord", next="stuck", loading=.9, history=waited).type, "fail")

    def test_an_app_just_opened_is_waited_for_while_it_draws(self):
        evaluate = self.planner.jev.evaluate

        def asked():
            return any("drawing" in call.args[1] for call in evaluate.call_args_list)

        opened = ["skill(open_app {'name': 'web browser'})", "skill(wait {'s': 1.0})"]
        for still in ({"drawing": .9}, {"loading": .9}):
            self.assertEqual(self.step("search for horses", history=opened, **still).brief(), "skill(wait {'s': 1.0})")
        self.assertTrue(asked())
        self.assertEqual(self.step("search for horses", history=opened).type, "click")
        self.assertEqual(self.step("search for horses", drawing=.9, history=opened + opened[1:]).type, "click")
        self.assertEqual(self.step("close google", next="close_tab", drawing=.9, history=opened).brief(),
                         "hotkey(CTRL+W)")  # a key needs nothing drawn
        # after other steps the question isn't asked, and a page still loading is no reason to wait
        evaluate.reset_mock()
        self.assertEqual(self.step("search for horses", drawing=.9, loading=.9, history=["click(1,1)"]).type, "click")
        self.assertFalse(asked())

    def test_typing_where_nothing_takes_text_clicks_first(self):
        goal = "type in hello"
        action = self.step(goal, next="type", text=str(spans(goal).index("hello")), focused=.1,
                           history=["click(600,300)"])
        self.assertEqual(action.type, "click")


class JevOnlyGuardTests(unittest.TestCase):
    def next_step(self, approved):
        planner = JevOnlyPlanner(Mock(min_presence=.85), approved)
        answers = {"complete": {"noul": .1}, "question": {"noul": .05}, "composed": {"noul": .05},
                   "next": {"choice": "enter"},
                   "present": {"noul": .95}, "target": BOX, "text": {"choice": "none"}, "replace": {"noul": .1},
                   "fits": {"noul": .95}, "app_present": {"noul": .1}, "app_name": {"choice": "none"},
                   "launcher": {"noul": .1}, "app": BOX, "composed": {"noul": .05}, "irreversible": {"noul": .95},
                   "site": {"noul": .05}, "submit": {"noul": .05}, "browser_named": {"noul": .05},
                   "left": {"noul": .9}, "window": {"noul": .9}, "loading": {"noul": .05}, "focused": {"noul": .95},
                   "typed": {"noul": .95}, "pending": {"noul": .05}}
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
        self.planner.handoff, self.planner.uses_flash = True, True
        self.planner.flash.next_actions.return_value = Plan("Stage 1/1: planned", [Action("hotkey", keys=("TAB",))], [])

    def ctx(self, run):
        return StepContext("summarize this page", DeviceProfile("linux", Rect(0, 0, 4, 3), "mouse", 1), Mock(), [], 1,
                           10, run_id=run)

    def test_jev_tells_the_os_when_flash_cant(self):
        self.planner.flash.classify_os.side_effect = RuntimeError("401 UNAUTHENTICATED")
        self.planner.jev = Mock()
        self.planner.jev.evaluate.return_value = ({"os": {"choice": "linux", "confidence": .9}}, (4, 3), (0, 0))
        self.assertEqual(self.planner.classify_os(b"jpeg"), "linux")

    def test_without_handoff_a_step_jev_cant_take_ends_the_request_at_once(self):
        self.planner.handoff = False
        self.planner.steps.next_actions.return_value = Plan(FAILED, [Action("fail", reason="needs new text")], [])
        note = StepContext("open calculadora", DeviceProfile("linux", Rect(0, 0, 4, 3), "mouse", 1), Mock(), [], 2,
                           10, note="the launch landed elsewhere", run_id="a")
        for ctx in (self.ctx("a"), note):
            self.assertEqual(self.planner.next_actions(b"img", ctx).actions[0].type, "fail")
        self.planner.flash.next_actions.assert_not_called()

    def test_jev_only_never_calls_flash(self):
        self.planner.handoff, self.planner.uses_flash = False, False
        self.planner.steps.next_actions.return_value = Plan(FAILED, [Action("fail", reason="composed")], [],
                                                            needs_planner=True)
        self.assertEqual(self.planner.next_actions(b"img", self.ctx("a")).actions[0].type, "fail")
        self.planner.jev = Mock()
        self.planner.jev.evaluate.return_value = ({"os": {"choice": "linux", "confidence": .9}}, (4, 3), (0, 0))
        self.assertEqual(self.planner.classify_os(b"jpeg"), "linux")
        self.planner.flash.next_actions.assert_not_called()
        self.planner.flash.classify_os.assert_not_called()

    def test_without_handoff_text_jev_cant_write_still_goes_to_flash(self):
        self.planner.handoff = False
        self.planner.steps.next_actions.return_value = Plan(FAILED, [Action("fail", reason="composed")], [],
                                                            needs_planner=True)
        self.planner.steps.current_line = "work out 12 times 9"
        self.assertIn("[Jev handed off: composed]", self.planner.next_actions(b"img", self.ctx("a")).reasoning)
        self.assertEqual(self.planner.flash.next_actions.call_args.args[1].goal, "work out 12 times 9")  # its line only
        self.planner.next_actions(b"img", self.ctx("a"))
        self.assertEqual(self.planner.steps.next_actions.call_count, 2)  # Jev takes the next step again
        self.planner.flash.next_actions.return_value = Plan("verified", [Action("done", summary="Done.")], [])
        self.planner.steps.line_done.return_value = Plan("Jev next step: click", [Action("click", x=.1, y=.1)], [])
        self.assertEqual(self.planner.next_actions(b"img", self.ctx("a")).actions[0].type, "click")
        self.assertEqual(self.planner.steps.line_done.call_args.args[1].goal, "summarize this page")  # its line done
        failed = Plan(FAILED, [Action("fail", reason="no sure spot")], [], needs_planner=True)
        self.planner.steps.line_done.return_value, self.planner.steps.current_line = failed, "paste it in"
        self.planner.flash.next_actions.side_effect = [self.planner.flash.next_actions.return_value,
                                                       Plan("Stage 1/1: pasted", [Action("hotkey", keys=("V",))], [])]
        self.assertEqual(self.planner.next_actions(b"img", self.ctx("a")).actions[0].brief(), "hotkey(V)")
        self.assertEqual(self.planner.flash.next_actions.call_args.args[1].goal, "paste it in")  # the next line's step

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
