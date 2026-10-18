"""Deep mode: when race is stuck, a stronger model takes over the rest of the request.

Race handles almost everything fast (Jev triage + Jev-checked Flash stages). Long, open-ended
work in dense apps ("make me a house in Blender") stalls it: its stage checks veto steps whose
effect is hard to see. Once DEEP_AFTER of race's last WINDOW steps were unproductive, the rest
of that request goes to a one-call-per-step agent on DEEP_MODEL (Argon, lowest thinking), free
to take any route; it chose Blender's Python console and built the house where race failed
11/12 times. Strategy-only Argon with race's gated steps, and ungrounded Flash steps, both
stalled. Every deep action batch passes Jev's side-effect question and, unless the user approved
the run, the irreversible one: a request that only drafts a reply must still ask before Send.
Requests race finishes are untouched.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from brain.agent.executor import ModelExecutor, Plan, StepContext
from brain.agent.jev import SIDE_EFFECT_GATE, SIDE_EFFECTS, probability
from brain.agent.triage import GATE, IRREVERSIBLE_KINDS
from brain.cancel import CancelToken, Cancelled
from brain.config import Config
from brain.imaging import grid_copy
from brain.types import OSName

UNPRODUCTIVE = ("unchanged", "Rejected action", "Recovery required")
WINDOW = 6  # race steps looked at; deep mode once DEEP_AFTER of them were unproductive
STEP_IRREVERSIBLE = "Would carrying out the proposed `actions` on this `screen` " + IRREVERSIBLE_KINDS
DEEP_STALL_S = 60.0  # one Argon step can take 20 s+; race keeps the 20 s no-progress rule


class DeepFallback:
    """Executor wrapper: `inner` (race) until it is stuck, then the deep agent for that request."""

    def __init__(self, inner: Any, cfg: Config, cancel: CancelToken, on_status=lambda message: None) -> None:
        self.inner, self.cancel, self.on_status = inner, cancel, on_status
        self.after = cfg.deep_after
        self.deep = ModelExecutor(replace(cfg, gemini_model=cfg.deep_model, gemini_thinking=cfg.deep_thinking,
                                          fallback_model="", model_timeout_s=cfg.deep_timeout_s, zai_api_key=""),
                                  cancel, routing=False)
        self._run: tuple[str, str] | None = None
        self._recent: list[bool] = []  # unproductive flags of the last WINDOW race steps
        self.active = False
        self._race_step = False  # this step went to race because the deep model failed

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    @property
    def verifies_stages(self) -> bool:
        return False if self.active else getattr(self.inner, "verifies_stages", False) is True

    @property
    def stall_s(self) -> float | None:
        return DEEP_STALL_S if self.active else None

    def next_actions(self, image_jpeg: bytes, ctx: StepContext) -> Plan:
        if self._run != (ctx.run_id, ctx.goal):
            self._run, self._recent, self.active = (ctx.run_id, ctx.goal), [], False
        if not self.active:
            plan = self.inner.next_actions(image_jpeg, ctx)
            unproductive = (not plan.actions and bool(plan.rejected)) or any(w in (ctx.note or "") for w in UNPRODUCTIVE)
            # Count over a window: race alternating a step and a veto (fail, ok, fail, ok...) never stalls
            # outright but ground through 18 steps of hand modelling in Blender.
            self._recent = (self._recent + [unproductive])[-WINDOW:]
            if sum(self._recent) < self.after or any(a.type in ("done", "ask_user") for a in plan.actions):
                return plan
            self.active = True
            self.on_status("This one needs more thought; switching to deep mode…")
        try:
            plan = self.deep.next_actions(grid_copy(image_jpeg), replace(ctx, note=ctx.note or "Deep mode: the fast "
                                                                         "planner was stuck; take any route that works."))
        except Cancelled:
            raise
        except Exception:  # a 504 or timeout ended a whole house run: race takes this one step instead
            self._race_step = True
            return self.inner.next_actions(image_jpeg, ctx)
        self._race_step = False
        return self._guard(image_jpeg, plan, ctx)

    def acknowledge(self, executed: bool) -> None:
        if not self.active or self._race_step:
            self.inner.acknowledge(executed)

    def _guard(self, image_jpeg: bytes, plan: Plan, ctx: StepContext) -> Plan:
        """Deep actions that would close or discard something nobody asked for never run; irreversible
        ones (a send, a delete) wait for the user's yes."""
        acting = [a for a in plan.actions if a.type not in ("done", "ask_user", "fail", "remember")]
        if not acting:
            return plan
        approved = self.inner.approved_run()
        questions = {"side_effects": {"type": "noul", "instructions": SIDE_EFFECTS}}
        if not approved:
            questions["irreversible"] = {"type": "noul", "instructions": STEP_IRREVERSIBLE}
        briefs = [a.brief() for a in acting]
        answers, _, _ = self.inner.jev.evaluate(image_jpeg, questions, run_id=ctx.run_id, step=ctx.step,
                                                state={"request": ctx.goal, "actions": json.dumps(briefs)})
        if probability(answers["side_effects"].get("noul")) >= SIDE_EFFECT_GATE:
            return replace(plan, actions=[], rejected=["Blocked: those actions would close or discard something "
                                                       "the request did not ask for. Find another way."])
        if not approved and probability(answers["irreversible"].get("noul")) >= GATE:
            return self.inner.hold(ctx, "; ".join(briefs))
        return plan

    def classify_os(self, image_jpeg: bytes) -> OSName:
        return self.inner.classify_os(image_jpeg)

    def close(self) -> None:
        self.deep.close()
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()
