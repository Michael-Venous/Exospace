"""Experimental triage-first executor: Jev decides act / plan / none per segment.

act  -> one Jev-grounded (zoom-refined) click, no LLM call
none -> remembered as conversation context, no input
plan -> delegated unchanged to the runtime JevFlashExecutor
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace

from brain.agent.executor import Plan, StepContext
from brain.agent.jev_flash import JevFlashExecutor
from brain.cancel import CancelToken
from brain.config import Config
from brain.types import Action, OSName

from bench.arch.triage_study import INTENT_CRITERIA, box_center, jev_raw
from bench.arch.zoom_study import refine

GATE = 0.5  # presence/uniqueness/zoom-presence floor (zero unsafe actions in the triage study)


class TriageExecutor:
    image_grid = False

    def __init__(self, cfg: Config, cancel: CancelToken) -> None:
        self.inner = JevFlashExecutor(cfg, cancel)
        self.jev = self.inner.jev
        self.model = self.inner.model
        self.cancel = cancel
        self._key: tuple | None = None
        self.mode = ""
        self.point: tuple[float, float] | None = None
        self._executed = False
        self.triage_log: list[dict] = []

    @property
    def verifies_stages(self) -> bool:
        return self.mode == "plan"

    def acknowledge(self, executed: bool) -> None:
        if self.mode == "plan":
            self.inner.acknowledge(executed)
        else:
            self._executed = executed

    def _triage(self, image: bytes, utterance: str) -> None:
        questions = {
            "intent": {"type": "choice", "criteria": INTENT_CRITERIA, "instructions":
                       "The user is talking to an AI assistant that operates the device shown in `screen` "
                       "with a mouse/touch and keyboard. Classify the user's latest `utterance`."},
            "present": {"type": "noul", "instructions":
                        "Is there a control visible in `screen` that the `utterance` asks to click, tap or open right now?"},
            "unique": {"type": "noul", "instructions":
                       "Does the `utterance` identify exactly one visible control in `screen`, with no ambiguity?"},
            "target": {"type": "box", "instructions":
                       "The single visible control in `screen` that the `utterance` asks to click, tap or open right now. "
                       "For an item in a list or grid, locate the whole item."},
        }
        answers, size, padded, _ = self.cancel.run(lambda: jev_raw(self.jev, image, utterance, questions))
        choice = answers["intent"].get("choice")
        point = box_center(answers["target"], size, padded)
        gated = min(answers["present"].get("noul", 0), answers["unique"].get("noul", 0)) >= GATE
        self.mode, self.point = "plan", None
        if choice == "no_action":
            self.mode = "none"
        elif choice == "act_now" and gated and point:
            zoom = self.cancel.run(lambda: refine(self.jev, image, utterance, point))
            if zoom["point"] and (zoom["present"] or 0) >= GATE:
                self.mode, self.point = "act", zoom["point"]
        self.triage_log.append({"utterance": utterance, "choice": choice, "mode": self.mode})

    def next_actions(self, image: bytes, ctx: StepContext) -> Plan:
        key = (ctx.run_id, ctx.goal)
        if key != self._key:
            self._key, self._executed = key, False
            self._triage(image, ctx.goal.splitlines()[-1])
        if self.mode == "none":
            return Plan("Context noted", [Action("remember", key=f"context_{ctx.run_id}", value=ctx.goal),
                                          Action("done", summary="Noted.")], [])
        if self.mode == "act":
            if not self._executed:
                return Plan("Jev direct action", [Action("click", x=self.point[0], y=self.point[1])], [])
            if "unchanged" not in (ctx.note or ""):
                return Plan("Jev direct action done", [Action("done", summary="Done.")], [])
            self.mode = "plan"  # the click had no visible effect: let the planner recover
        return self.inner.next_actions(image, ctx)

    def classify_os(self, image: bytes) -> OSName:
        return self.inner.classify_os(image)

    def close(self) -> None:
        self.inner.close()


COMPLETION_GATE = 0.5  # concrete-state checks: wrong states scored 0.00 in the verification probe


class TriageLedgerExecutor(TriageExecutor):
    """Triage-first, and the plan path's final check verifies the planner's concrete
    completion state instead of re-judging the original (history-dependent) wording.
    History is covered by the stages that already passed their own checks."""

    def __init__(self, cfg: Config, cancel: CancelToken) -> None:
        super().__init__(cfg, cancel)
        inner, assess = self.inner, self.jev.assess
        self._raw_assess = assess

        def concrete_assess(image, checks, targets, **kwargs):
            checks = dict(checks)
            if "complete" in checks:
                checks["complete"] = f"The screen shows this state: {inner.completion}"
            values, points = assess(image, checks, targets, **kwargs)
            if "complete" in values and values["complete"] >= COMPLETION_GATE:
                values["complete"] = 1.0
            return values, points

        self.jev.assess = concrete_assess


class RaceExecutor(TriageLedgerExecutor):
    """Triage and the planner start together on every new segment. A confident Jev
    act/none wins immediately and the planner's result is discarded; otherwise the
    planner's already-running result is used, so a Jev miss costs no extra time."""

    def __init__(self, cfg: Config, cancel: CancelToken) -> None:
        super().__init__(cfg, cancel)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="race-planner")
        self._planner: Future[Plan] | None = None

    def next_actions(self, image: bytes, ctx: StepContext) -> Plan:
        key = (ctx.run_id, ctx.goal)
        if key != self._key:
            self._planner = self._pool.submit(self.inner.next_actions, image, ctx)
            self._key, self._executed = key, False
            self._triage(image, ctx.goal.splitlines()[-1])
            if self.mode == "plan":
                return self.cancel.run(self._planner.result)
        if self.mode == "plan" and self._planner is not None and not self._planner.done():
            self.cancel.run(self._planner.result)  # an abandoned speculative plan must settle first
        return super().next_actions(image, ctx)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        super().close()


RELAXED_GATE = 0.5  # presence/uniqueness/check floor; zoom refinement protects precision


class RaceRelaxedExecutor(RaceExecutor):
    """Race + concrete verification, with the planner's Jev gate relaxed to 0.5 and
    every grounded planner target zoom-refined before it is used."""

    def __init__(self, cfg: Config, cancel: CancelToken) -> None:
        super().__init__(replace(cfg, jev_min_presence=RELAXED_GATE, jev_min_confidence=0.0), cancel)
        assess = self.jev.assess

        def refined_assess(image, checks, targets, **kwargs):
            values, points = assess(image, checks, targets, **kwargs)
            for key, point in list(points.items()):
                if point is not None:
                    zoom = self.cancel.run(lambda p=point, k=key: refine(self.jev, image, targets[k], p))
                    points[key] = zoom["point"] if zoom["point"] and (zoom["present"] or 0) >= RELAXED_GATE else None
            return values, points

        self.jev.assess = refined_assess


LEDGER_GATE = 0.3  # finished tasks scored >= 0.55, traps <= 0.04 in the verifier study


class RaceLedgerExecutor(RaceExecutor):
    """Race, with completion judged from the goal + the record of executed stages + the
    screen (verifier study: 12/12 finished accepted, 0/18 traps accepted)."""

    def __init__(self, cfg: Config, cancel: CancelToken) -> None:
        super().__init__(cfg, cancel)
        self.executed: list[str] = []
        inner, raw_assess = self.inner, self._raw_assess
        acknowledge = inner.acknowledge

        def recording_acknowledge(executed: bool) -> None:
            if executed and inner.stages and inner.cursor < len(inner.stages):
                self.executed.extend(_describe(a) for a in inner.stages[inner.cursor]["actions"])
            acknowledge(executed)

        def ledger_assess(image, checks, targets, **kwargs):
            if "complete" not in checks:
                return raw_assess(image, checks, targets, **kwargs)
            # Final verification: the steps-aware score alone decides. It subsumes the last
            # stage's own `previous` check, and the routing vote is meaningless here.
            score = self.cancel.run(lambda: ledger_score(self.jev, image, kwargs.get("request") or self._goal, self.executed))
            passed = 1.0 if score >= LEDGER_GATE else score
            return {key: passed for key in checks} | {"route_execute": 1.0 if score >= LEDGER_GATE else 0.0}, {}

        inner.acknowledge = recording_acknowledge
        self.jev.assess = ledger_assess
        self._goal = ""

    def next_actions(self, image: bytes, ctx: StepContext) -> Plan:
        if (ctx.run_id, ctx.goal) != self._key:
            self.executed, self._goal = [], ctx.goal
        return super().next_actions(image, ctx)


def _describe(action: dict) -> str:
    kind = action.get("type")
    detail = action.get("target") or action.get("text") or "+".join(action.get("keys") or []) or ""
    return f"{kind} {detail}".strip()


def ledger_score(jev, image: bytes, goal: str, executed: list[str]) -> float:
    from bench.arch.verify_study import ask

    score, _ = ask(jev, image, {"goal": goal, "executed_steps": executed},
                   "Considering the `executed_steps` (what was actually done) together with the current `screen`, "
                   "has the user's ENTIRE `goal` been accomplished? Every part of the goal must be covered by an "
                   "executed step or be visibly true on the screen. A visible link, button or preview is not evidence "
                   "that it was used; text typed but not submitted is not a completed search or send.")
    return score
