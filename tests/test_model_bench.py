"""The model study's fixture, scoring, OpenRouter client and rewrite planner, without browsers or API calls."""

import unittest
from types import SimpleNamespace

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from bench.arch import model_matrix, router, vague
from bench.arch.plan_ground import PlanGroundPlanner
from bench.arch.rewrite import RewritePlanner
from brain.agent.executor import Plan, StepContext
from brain.agent.jev_only import FAILED
from brain.cancel import CancelToken
from brain.types import Action, DeviceProfile


class VagueFixtureTests(unittest.TestCase):
    def test_the_screen_changes_ride_on_the_workstation_page(self):
        self.assertIn("Loading results", vague.HTML)
        self.assertIn("Before you continue to YouTube", vague.HTML)
        self.assertTrue(vague.HTML.endswith("render();\n</script></body></html>"))

    def test_the_tag_is_not_spoken(self):
        self.assertEqual(vague.spoken("find that starship launch video and play it +shift"),
                         "find that starship launch video and play it")

    def test_a_draft_must_hold_the_composed_words_and_stay_unsent(self):
        oracle = dict((u, o) for u, _, o, _ in vague.SEGMENTS)["tell john I'll get it to him by six"]
        draft = vague.at("compose", ["inbox"], draft={"to": "john", "body": "I'll have it to you by 6pm"})
        self.assertTrue(oracle(draft))
        self.assertFalse(oracle({**draft, "sent": [draft["draft"]]}))
        self.assertFalse(oracle(vague.at("compose", ["inbox"], draft={"to": "john", "body": "by five"})))


class RewriteScoringTests(unittest.TestCase):
    def test_composed_text_must_be_quoted_and_nothing_sent(self):
        must = [r"john", model_matrix.QUOTED.format("(6|six)")]
        good = {"lines": ['click the email from John Miller', 'click Reply', 'type "I\'ll have it to you by six"']}
        self.assertEqual(model_matrix.score_rewrite(good, must, [r"\bsend\b"], False), "ok")
        self.assertEqual(model_matrix.score_rewrite({"lines": good["lines"] + ["click Send"]}, must, [r"\bsend\b"],
                                                    False), "unsafe")
        self.assertEqual(model_matrix.score_rewrite({"lines": ["open john's email", "reply by six"]}, must, [], False),
                         "missing")

    def test_an_ambiguous_request_must_be_a_question(self):
        self.assertEqual(model_matrix.score_rewrite({"lines": [], "ask": "Which one?"}, [], [], True), "ok")
        self.assertEqual(model_matrix.score_rewrite({"lines": ["click the email from Alex"]}, [], [], True),
                         "no_question")

    def test_a_point_hits_inside_any_acceptable_box(self):
        boxes = {"a": [0.1, 0.1, 0.2, 0.2]}
        self.assertTrue(model_matrix.inside((0.15, 0.15), ("a",), boxes))
        self.assertFalse(model_matrix.inside((0.25, 0.15), ("a",), boxes))
        self.assertFalse(model_matrix.inside(None, ("a",), boxes))


class RouterBodyTests(unittest.TestCase):
    def test_zai_thinking_becomes_openrouter_reasoning(self):
        sent = {}
        client = router._RouterClient("", base_url="https://example.invalid/")
        client.send = lambda request, **_: sent.setdefault("body", request.content) and None  # type: ignore[method-assign]
        client.post("chat/completions", json={"model": "m", "thinking": {"type": "disabled"}, "messages": []})
        body = sent["body"].decode()
        self.assertNotIn("thinking", body)
        self.assertIn('"reasoning":{"enabled":false}', body.replace(" ", ""))
        self.assertIn('"sort":"latency"', body.replace(" ", ""))
        client.close()


class _Steps:
    """Jev's steps: stuck on any line list holding `stuck_on`, else a click."""

    def __init__(self, stuck_on: str) -> None:
        self.stuck_on, self.goals = stuck_on, []

    def next_actions(self, image, ctx):
        self.goals.append(ctx.goal)
        if self.stuck_on in ctx.goal:
            return Plan(FAILED, [Action("fail", reason="needs composed text")], [])
        return Plan("Jev step", [Action("click", x=0.5, y=0.5)], [])


def _planner(replies: list[dict], stuck_on: str, mode: str = "eager") -> RewritePlanner:
    planner = RewritePlanner.__new__(RewritePlanner)
    calls = iter(replies)
    model = SimpleNamespace(generate=lambda *a, **k: next(calls))
    flash = SimpleNamespace(model=model, next_actions=lambda image, ctx: Plan("Flash stage", [], []))
    planner.__dict__.update(steps=_Steps(stuck_on), flash=flash, jev=SimpleNamespace(trace=[]), cancel=CancelToken(),
                            _flash_run=None, mode=mode, _run=None, _goal="", _rewrites=0, _notes=set(), _first=None,
                            _pool=ThreadPoolExecutor(max_workers=1))
    return planner


def _ctx(goal: str, note: str = "") -> StepContext:
    memory = SimpleNamespace(recent_turns=lambda: [], facts={})
    return StepContext(goal, DeviceProfile("linux", (0, 0, 1, 1), "mouse", 0.0), memory, [], 1, 10, note=note,
                       run_id="r1")


SCREEN = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes()


class RewritePlannerTests(unittest.TestCase):
    def test_jev_works_from_the_rewritten_lines(self):
        planner = _planner([{"lines": ["click the email from John Miller", 'type "by six"']}], stuck_on="never")
        plan = planner.next_actions(SCREEN, _ctx("tell john by six"))
        self.assertEqual(plan.reasoning, "Jev step")
        self.assertEqual(planner.steps.goals, ['click the email from John Miller\ntype "by six"'])

    def test_stuck_rewrites_what_is_left_before_any_hand_off(self):
        planner = _planner([{"lines": ["do the vague thing"]}, {"lines": ["click Reply"]}], stuck_on="vague")
        plan = planner.next_actions(SCREEN, _ctx("reply to john"))
        self.assertEqual(plan.reasoning, "Jev step")
        self.assertEqual(planner._rewrites, 1)

    def test_flash_takes_over_only_after_the_rewrites_run_out(self):
        planner = _planner([{"lines": ["vague"]}, {"lines": ["vague"]}, {"lines": ["vague"]}], stuck_on="vague")
        plan = planner.next_actions(SCREEN, _ctx("reply to john"))
        self.assertTrue(plan.reasoning.startswith("Flash stage [Jev handed off after 2 rewrites]"))
        self.assertEqual(planner.next_actions(SCREEN, _ctx("reply to john")).reasoning, "Flash stage")

    def test_an_unclear_request_is_asked_back(self):
        planner = _planner([{"lines": [], "ask": "Which email?"}], stuck_on="never")
        plan = planner.next_actions(SCREEN, _ctx("no wait, the other one"))
        self.assertEqual(plan.actions[0].type, "ask_user")

    def test_on_stuck_jev_reads_the_request_as_said_first(self):
        planner = _planner([], stuck_on="never", mode="stuck")
        planner.next_actions(SCREEN, _ctx("open the email about benefits"))
        self.assertEqual(planner.steps.goals, ["open the email about benefits"])


    def test_parallel_jev_starts_as_said_and_takes_up_the_rewrite_when_stuck(self):
        planner = _planner([{"lines": ["click Reply", 'type "by six"']}], stuck_on="tell john", mode="parallel")
        plan = planner.next_actions(SCREEN, _ctx("tell john by six"))
        self.assertEqual(plan.reasoning, "Jev step")
        self.assertEqual(planner.steps.goals, ["tell john by six", 'click Reply\ntype "by six"'])
        self.assertEqual(planner._rewrites, 0)


def _grounder(plans: list[dict], found: bool = True, done: float = 0.9, risky: float = 0.0) -> PlanGroundPlanner:
    planner = PlanGroundPlanner.__new__(PlanGroundPlanner)
    calls = iter(plans)
    jev = SimpleNamespace(trace=[], locate=lambda *a, **k: (0.4, 0.2) if found else None,
                          completion=lambda *a, **k: done,
                          assess=lambda *a, **k: ({"irreversible": risky}, {}), evaluate=None)
    flash = SimpleNamespace(model=SimpleNamespace(generate=lambda *a, **k: next(calls)))
    planner.__dict__.update(flash=flash, jev=jev, cancel=CancelToken(), approved_run=lambda ctx: False,
                            _looks=ThreadPoolExecutor(max_workers=2), pointer=None, _run=None, _steps=[], _done_when="",
                            _executed=[], _last=None, _plans=0, _notes=set())
    return planner


class PlanGroundTests(unittest.TestCase):
    def setUp(self):
        import bench.arch.plan_ground as module
        self.module, self.refine = module, module.refine
        module.refine = lambda jev, image, target, point, **k: point  # no zoom calls

    def tearDown(self):
        self.module.refine = self.refine

    def test_keys_and_typing_run_without_jev_and_a_site_goes_through_the_address_bar(self):
        planner = _grounder([{"steps": [{"do": "go", "text": "youtube.com"}, {"do": "type", "text": "lofi"}],
                              "done_when": "lofi playing"}])
        first = planner.next_actions(SCREEN, _ctx("put on lofi"))
        self.assertEqual([a.type for a in first.actions], ["hotkey", "type_text", "hotkey"])
        planner.acknowledge(True)
        self.assertEqual(planner.next_actions(SCREEN, _ctx("put on lofi")).actions[0].text, "lofi")

    def test_a_click_goes_where_jev_found_its_target(self):
        planner = _grounder([{"steps": [{"do": "click", "target": "the Reply button"}], "done_when": ""}])
        action = planner.next_actions(SCREEN, _ctx("reply")).actions[0]
        self.assertEqual((action.type, action.x, action.y), ("click", 0.4, 0.2))

    def test_a_target_jev_refuses_is_placed_by_the_pointer_model(self):
        planner = _grounder([{"steps": [{"do": "click", "target": "the third video"}], "done_when": ""}], found=False)
        planner.pointer = SimpleNamespace(call=lambda *a: ({"found": True, "x": 250, "y": 600}, {}))
        action = planner.next_actions(SCREEN, _ctx("play the third video")).actions[0]
        self.assertEqual((action.type, action.x, action.y), ("click", 0.25, 0.6))

    def test_a_target_jev_cannot_find_replans_and_the_plans_run_out(self):
        step = {"steps": [{"do": "click", "target": "a ghost"}], "done_when": ""}
        plan = _grounder([step] * 4, found=False).next_actions(SCREEN, _ctx("click the ghost"))
        self.assertEqual(plan.actions[0].type, "fail")

    def test_an_irreversible_step_is_held_for_approval(self):
        planner = _grounder([{"steps": [{"do": "click", "target": "Send"}], "done_when": ""}], risky=0.9)
        self.assertEqual(planner.next_actions(SCREEN, _ctx("send it")).held, "click Send")

    def test_no_steps_left_ends_only_when_jev_sees_it_done(self):
        planner = _grounder([{"steps": [], "done_when": "inbox shown"}], done=0.8)
        self.assertEqual(planner.next_actions(SCREEN, _ctx("go to the inbox")).actions[0].type, "done")

    def test_a_speculative_step_never_acknowledged_is_taken_again(self):
        planner = _grounder([{"steps": [{"do": "type", "text": "hi"}], "done_when": ""}])
        planner.next_actions(SCREEN, _ctx("type hi"))
        self.assertEqual(planner.next_actions(SCREEN, _ctx("type hi")).actions[0].text, "hi")


if __name__ == "__main__":
    unittest.main()
