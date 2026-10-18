"""Deep mode: when race is stuck, a stronger model takes over the rest of the request.

Race handles almost everything fast (Jev triage + Jev-checked Flash stages). Long, open-ended
work in dense apps ("make me a house in Blender") stalls it: its stage checks veto steps whose
effect is hard to see. After DEEP_AFTER unproductive steps in a row the request switches to a
one-call-per-step agent on DEEP_MODEL (direct executor, aiming grid, its own done check), free
to take any route. Requests race finishes are untouched. Every deep action batch still passes
Jev's side-effect question, and the request-level irreversible gate already ran in race.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from brain.agent.executor import ModelExecutor, Plan, StepContext
from brain.agent.jev import SIDE_EFFECTS, probability
from brain.cancel import CancelToken
from brain.config import Config
from brain.imaging import grid_copy
from brain.types import OSName

UNPRODUCTIVE = ("unchanged", "Rejected action", "Recovery required")


class DeepFallback:
    """Executor wrapper: `inner` (race) until it is stuck, then the deep agent for that request."""

    def __init__(self, inner: Any, cfg: Config, cancel: CancelToken, on_status=lambda message: None) -> None:
        self.inner, self.cancel, self.on_status = inner, cancel, on_status
        self.after = cfg.deep_after
        self.deep = ModelExecutor(replace(cfg, gemini_model=cfg.deep_model, gemini_thinking=cfg.deep_thinking,
                                          fallback_model="", model_timeout_s=cfg.deep_timeout_s, zai_api_key=""),
                                  cancel, routing=False)
        self._run: tuple[str, str] | None = None
        self._stuck = 0
        self.active = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    @property
    def verifies_stages(self) -> bool:
        return False if self.active else getattr(self.inner, "verifies_stages", False) is True

    def next_actions(self, image_jpeg: bytes, ctx: StepContext) -> Plan:
        if self._run != (ctx.run_id, ctx.goal):
            self._run, self._stuck, self.active = (ctx.run_id, ctx.goal), 0, False
        if not self.active:
            plan = self.inner.next_actions(image_jpeg, ctx)
            unproductive = (not plan.actions and bool(plan.rejected)) or any(w in (ctx.note or "") for w in UNPRODUCTIVE)
            self._stuck = self._stuck + 1 if unproductive else 0
            if self._stuck < self.after or any(a.type in ("done", "ask_user") for a in plan.actions):
                return plan
            self.active = True
            self.on_status("This one needs more thought; switching to deep mode…")
        plan = self.deep.next_actions(grid_copy(image_jpeg), replace(ctx, note=ctx.note or "Deep mode: the fast "
                                                                     "planner was stuck; take any route that works."))
        return self._guard(image_jpeg, plan, ctx)

    def acknowledge(self, executed: bool) -> None:
        if not self.active:
            acknowledge = getattr(self.inner, "acknowledge", None)
            if callable(acknowledge):
                acknowledge(executed)

    def _guard(self, image_jpeg: bytes, plan: Plan, ctx: StepContext) -> Plan:
        """Deep actions that would close or discard something nobody asked for never run."""
        acting = [a for a in plan.actions if a.type not in ("done", "ask_user", "fail", "remember")]
        if not acting:
            return plan
        answers, _, _ = self.inner.jev.evaluate(
            image_jpeg, {"side_effects": {"type": "noul", "instructions": SIDE_EFFECTS}}, run_id=ctx.run_id,
            step=ctx.step, state={"request": ctx.goal, "actions": json.dumps([a.brief() for a in acting])})
        if probability(answers["side_effects"].get("noul")) < 0.5:
            return plan
        return replace(plan, actions=[], rejected=["Blocked: those actions would close or discard something the "
                                                   "request did not ask for. Find another way."])

    def classify_os(self, image_jpeg: bytes) -> OSName:
        return self.inner.classify_os(image_jpeg)

    def close(self) -> None:
        self.deep.close()
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()
