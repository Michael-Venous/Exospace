"""Candidates from study 10, each the runtime race with one change (bench only; the runtime code is in
brain/agent/jev_first.py and jev_flash.py, Agent owner).

  race+continue   Flash planned one stage per plan in 21 of 21 vague-speech requests, and each finished
                  stage that left the goal undone was a Flash replan (33 of race's 61 plans, 103 s), all
                  correctly not done. Here Jev's own step (~1 s) is tried first; Flash replans only if Jev
                  can't take it. Jev's steps join Flash's executed steps, so the steps-aware check sees them.
  race-gate0.6    the planner's completion gate at 0.6, not 0.5: on 129 labelled calls replayed 3 times it
                  passed 1-3 of 65 unfinished screens (5-7 at 0.5) and rejected none of 36 finished ones.
"""

from __future__ import annotations

from dataclasses import replace

from brain.agent import jev_flash
from brain.agent.executor import Plan, StepContext
from brain.agent.jev_first import JevFirstPlanner
from brain.agent.jev_flash import JevFlashExecutor
from brain.agent.jev_only import FAILED
from brain.agent.race import RaceExecutor
from brain.cancel import CancelToken
from brain.config import Config

NOT_DONE = "Completion not verified"  # the reason JevFlashExecutor replans with when its stages are done
# Jev's steps in a row for one Flash plan. Uncapped, Jev's step behind the consent popup (a click it had no
# effect with, or a look again) repeated until the run ended in 4 of the 6 runs of 36 it lost (2026-10-20).
MAX_CONTINUE = 2


class _HandBack(Exception):
    """Flash's stages are done and the goal isn't: Jev may take the next step."""


class ContinueFlash(JevFlashExecutor):
    hand_back = True

    def _new_plan(self, image, ctx, reason="", visual_fallback=False):
        if self.hand_back and reason.startswith(NOT_DONE):
            raise _HandBack
        return super()._new_plan(image, ctx, reason, visual_fallback)


class ContinuePlanner(JevFirstPlanner):
    flash: ContinueFlash
    jev_step: list[str] | None = None  # the step Jev took for Flash's request, until it is acknowledged
    streak = 0  # Jev's steps since Flash last planned

    def next_actions(self, image: bytes, ctx: StepContext) -> Plan:
        if self._flash_run != ctx.run_id:
            return super().next_actions(image, ctx)
        try:
            return self.flash.next_actions(image, ctx)
        except _HandBack:
            # A note says the last step had no effect or was discarded: Flash replans from this screen.
            plan = self.steps.next_actions(image, ctx) if self.streak < MAX_CONTINUE and not ctx.note else None
            if plan is not None and plan.reasoning != FAILED:
                self.streak += 1
                self.jev_step = [a.brief() for a in plan.actions]
                return replace(plan, reasoning=f"{plan.reasoning} [Jev continued Flash's request]")
        self.streak = 0
        self.flash.hand_back = False  # Jev can't: Flash replans now, as the runtime does
        try:
            return self.flash.next_actions(image, ctx)
        finally:
            self.flash.hand_back = True

    def acknowledge(self, executed: bool) -> None:
        if self.jev_step is None:
            super().acknowledge(executed)
            return
        self.steps.acknowledge(executed)
        if executed:  # Flash's stage was acknowledged already: only Jev's step is new
            self.flash.executed.extend(self.jev_step)
        self.jev_step = None


class RaceContinueExecutor(RaceExecutor):
    def __init__(self, cfg: Config, cancel: CancelToken) -> None:
        super().__init__(replace(cfg, jev_first=True), cancel)
        # The runtime builds both inside the race; they keep their state and gain the hand-back.
        self.planner.__class__ = ContinuePlanner
        self.planner.flash.__class__ = ContinueFlash


class RaceGateExecutor(RaceExecutor):
    """One run at a time in a study, so the module's gate is set for this executor's life only."""

    gate = 0.6

    def __init__(self, cfg: Config, cancel: CancelToken) -> None:
        super().__init__(cfg, cancel)
        self._runtime_gate, jev_flash.COMPLETION_GATE = jev_flash.COMPLETION_GATE, self.gate

    def close(self) -> None:
        jev_flash.COMPLETION_GATE = self._runtime_gate
        super().close()
