"""Deep mode: when race is stuck, a stronger model sets the strategy and a fast one carries it out.

Race handles almost everything fast (Jev triage + Jev-checked Flash stages). Long, open-ended
work in dense apps ("make me a house in Blender") stalls it: its stage checks veto steps whose
effect is hard to see. After DEEP_AFTER unproductive steps in a row, one DEEP_MODEL call (Argon)
writes a short strategy for the rest of the request, and race's own Jev-grounded planner follows
it as guidance (precise, checked clicks). The strategy is rewritten when execution stalls again.
Measured on "make me a house in blender": Argon at every step built it in 507 s (10-68 s per
step); Flash alone circled for 39 steps modelling by hand; Argon's strategy with ungrounded
Flash steps chose the right route (the script console) but kept missing small controls. Requests race finishes are untouched. Every deep action batch still passes
Jev's side-effect question, and the request-level irreversible gate already ran in race.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from brain.agent.executor import Plan, StepContext, context_prompt
from brain.agent.jev import SIDE_EFFECT_GATE, SIDE_EFFECTS, probability
from brain.agent.model import language_model
from brain.cancel import CancelToken
from brain.config import Config
from brain.imaging import grid_copy
from brain.types import OSName

UNPRODUCTIVE = ("unchanged", "Rejected action", "Recovery required")
WINDOW = 6  # race steps looked at; deep mode once DEEP_AFTER of them were unproductive
RESTRATEGIZE = 3  # unproductive deep steps in a row before the strategy is rewritten
STRATEGY_PROMPT = """You set the strategy for an agent that operates a device through its screen, keyboard and
mouse. A fast step-by-step agent got stuck on the user's goal. From the screenshot, the goal and what was
tried, write the most reliable route to finish it: prefer routes whose every step has a clearly visible
result, such as keyboard shortcuts, menus, typed values, or an app's own command line or script console
when it has one, over long chains of precise mouse work. Name the concrete steps. At most 80 words."""
STRATEGY_SCHEMA = {"type": "object", "properties": {"strategy": {"type": "string"}}, "required": ["strategy"]}


class DeepFallback:
    """Executor wrapper: `inner` (race) until it is stuck, then the deep agent for that request."""

    def __init__(self, inner: Any, cfg: Config, cancel: CancelToken, on_status=lambda message: None) -> None:
        self.inner, self.cancel, self.on_status = inner, cancel, on_status
        self.after = cfg.deep_after
        self.deep = inner.planner  # race's Jev + Flash planner: grounded, zoom-refined, checked stages
        self.strategist = language_model(replace(cfg, gemini_model=cfg.deep_model, gemini_thinking=cfg.deep_thinking,
                                                 fallback_model="", model_timeout_s=cfg.deep_timeout_s,
                                                 zai_api_key=""), cancel)
        self.strategy = ""
        self._deep_stuck = 0
        self._run: tuple[str, str] | None = None
        self._recent: list[bool] = []  # unproductive flags of the last WINDOW race steps
        self.active = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    @property
    def verifies_stages(self) -> bool:
        return True if self.active else getattr(self.inner, "verifies_stages", False) is True

    def next_actions(self, image_jpeg: bytes, ctx: StepContext) -> Plan:
        if self._run != (ctx.run_id, ctx.goal):
            self._run, self._recent, self.active, self.strategy = (ctx.run_id, ctx.goal), [], False, ""
        if not self.active:
            plan = self.inner.next_actions(image_jpeg, ctx)
            unproductive = (not plan.actions and bool(plan.rejected)) or any(w in (ctx.note or "") for w in UNPRODUCTIVE)
            # Count over a window: race alternating a step and a veto (fail, ok, fail, ok...) never stalls
            # outright but ground through 18 steps of hand modelling in Blender.
            self._recent = (self._recent + [unproductive])[-WINDOW:]
            if sum(self._recent) < self.after or any(a.type in ("done", "ask_user") for a in plan.actions):
                return plan
            self.active = True
            self.deep._key = None  # plan afresh, with the strategy
            self.on_status("This one needs more thought; switching to deep mode…")
        stalled = any(w in (ctx.note or "") for w in UNPRODUCTIVE)
        self._deep_stuck = self._deep_stuck + 1 if stalled else 0
        if not self.strategy or self._deep_stuck >= RESTRATEGIZE:
            self.strategy, self._deep_stuck = self._strategize(image_jpeg, ctx), 0
        plan = self.deep.next_actions(image_jpeg, replace(ctx, guidance=self.strategy))
        return self._guard(image_jpeg, plan, ctx)

    def _strategize(self, image_jpeg: bytes, ctx: StepContext) -> str:
        previous = f"\nThe previous strategy stalled: {self.strategy}" if self.strategy else ""
        data = self.strategist.generate(grid_copy(image_jpeg), context_prompt(ctx) + previous, STRATEGY_PROMPT,
                                        STRATEGY_SCHEMA, role="strategy", run_id=ctx.run_id, step=ctx.step)
        return str(data.get("strategy", ""))

    def acknowledge(self, executed: bool) -> None:
        (self.deep if self.active else self.inner).acknowledge(executed)

    def _guard(self, image_jpeg: bytes, plan: Plan, ctx: StepContext) -> Plan:
        """Deep actions that would close or discard something nobody asked for never run."""
        acting = [a for a in plan.actions if a.type not in ("done", "ask_user", "fail", "remember")]
        if not acting:
            return plan
        answers, _, _ = self.inner.jev.evaluate(
            image_jpeg, {"side_effects": {"type": "noul", "instructions": SIDE_EFFECTS}}, run_id=ctx.run_id,
            step=ctx.step, state={"request": ctx.goal, "actions": json.dumps([a.brief() for a in acting])})
        if probability(answers["side_effects"].get("noul")) < SIDE_EFFECT_GATE:
            return plan
        return replace(plan, actions=[], rejected=["Blocked: those actions would close or discard something the "
                                                   "request did not ask for. Find another way."])

    def classify_os(self, image_jpeg: bytes) -> OSName:
        return self.inner.classify_os(image_jpeg)

    def close(self) -> None:
        self.strategist.close()
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()
