"""Race's planner: Jev's own next step first; Flash takes over a request once Jev can't.

Most planned steps are a click, a typed phrase from the request, Enter or a scroll, which one
batched Jev call picks (brain/agent/jev_only.py) in about a second. A step that needs reasoning,
new text or a key Jev doesn't offer fails there, and that request continues on the Flash
planner for the rest of its run, so Flash is called only when Jev is stuck. With handoff off
(EXECUTOR=jev) the request ends there instead, unless the step needs text written for it or
has a target Jev can't make out: Flash takes that one step.
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
        why, request = "", ctx
        if ctx.note and self.handoff and self._flash_run != ctx.run_id:  # a recovery note: Jev doesn't read it
            self._flash_run, why = ctx.run_id, f"recovery note: {ctx.note[:80]}"
        if self._flash_run != ctx.run_id:
            plan = self.steps.next_actions(image, ctx)
            if plan.reasoning != FAILED or not (self.handoff or plan.needs_planner):
                return plan
            why = plan.actions[0].reason if plan.actions else "failed"
            if not self.handoff:
                # Without hand-off Flash writes this step only. "open up a google doc template": Jev read it as
                # needing composed text (0.46-0.62, 9/9) and ended a nine-clause request; Flash typed docs.new
                # there, then kept the run and wandered the template gallery for 70 s (2026-10-19).
                # Only for the line Jev is on: given the whole request, Flash took "copy the header" as its cue to
                # open docs.new for the line after it, then went back to the article (2026-10-19).
                self.flash.forget_plan()
                ctx = replace(ctx, goal=self.steps.current_line)
            else:
                self._flash_run = ctx.run_id
        plan = self.flash.next_actions(image, ctx)
        while not self.handoff and any(a.type == "done" for a in plan.actions):
            # Flash verified the line it was given, not the request: "open up a google doc template" so ended a
            # request with "paste the date of birth reference in" still to do (2026-10-19).
            plan = self.steps.line_done(image, request)
            if plan.reasoning != FAILED or not plan.needs_planner:
                break
            # The next line's step is Flash's too: a paste into the template just opened, where Jev saw no sure
            # spot (0.47), ended the request (2026-10-19).
            self.flash.forget_plan()
            plan = self.flash.next_actions(image, replace(request, goal=self.steps.current_line))
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
