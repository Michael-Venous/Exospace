"""The model study's fixture, scoring, OpenRouter client and rewrite planner, without browsers or API calls."""

import unittest
from types import SimpleNamespace

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from bench.arch import model_matrix, router, vague
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


if __name__ == "__main__":
    unittest.main()
