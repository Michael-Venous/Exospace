"""Race's planner: Jev's own next step first; Flash takes over a request once Jev can't.

Most planned steps are a click, a typed phrase from the request, Enter or a scroll, which one
batched Jev call picks (brain/agent/jev_only.py) in about a second. A step that needs reasoning,
new text or a key Jev doesn't offer fails there, and that request continues on the Flash
planner for the rest of its run, so Flash is called only when Jev is stuck. With handoff off
(EXECUTOR=jev) the request ends there instead: every step stays at Jev's speed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from brain.agent.executor import Plan, StepContext
from brain.agent.jev_flash import JevFlashExecutor
from brain.agent.jev_only import FAILED, JevOnlyPlanner
from brain.cancel import CancelToken
from brain.config import Config
from brain.types import OSName


class JevFirstPlanner:
    image_side = JevFlashExecutor.image_side

    def __init__(self, cfg: Config, cancel: CancelToken,
                 approved_run: Callable[[StepContext], bool] | None = None, handoff: bool = True) -> None:
        self.flash = JevFlashExecutor(cfg, cancel, approved_run)
        self.handoff = handoff
        self.jev = self.flash.jev
        self.steps = JevOnlyPlanner(self.jev, approved_run)
        self._flash_run: str | None = None  # the run Jev handed to Flash

    def next_actions(self, image: bytes, ctx: StepContext) -> Plan:
        why = ""
        if ctx.note and self.handoff and self._flash_run != ctx.run_id:  # a recovery note: Jev doesn't read it
            self._flash_run, why = ctx.run_id, f"recovery note: {ctx.note[:80]}"
        if self._flash_run != ctx.run_id:
            plan = self.steps.next_actions(image, ctx)
            if plan.reasoning != FAILED or not self.handoff:
                return plan
            self._flash_run, why = ctx.run_id, plan.actions[0].reason if plan.actions else "failed"
        plan = self.flash.next_actions(image, ctx)
        # The hand-off reason goes in the step's log line: why a request left Jev's ~1 s steps for Flash.
        return replace(plan, reasoning=f"{plan.reasoning} [Jev handed off: {why}]") if why else plan

    def acknowledge(self, executed: bool) -> None:
        self.flash.acknowledge(executed)

    def forget_plan(self) -> None:
        self.flash.forget_plan()

    def classify_os(self, image: bytes) -> OSName:
        return self.flash.classify_os(image)

    def close(self) -> None:
        self.flash.close()
