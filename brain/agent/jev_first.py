"""Race's planner: Jev's own next step first; Flash takes over a request once Jev can't.

Most planned steps are a click, a typed phrase from the request, Enter or a scroll, which one
batched Jev call picks (brain/agent/jev_only.py) in about a second. A step that needs reasoning,
new text or a key Jev doesn't offer fails there, and that request continues on the Flash
planner for the rest of its run, so Flash is called only when Jev is stuck.
"""

from __future__ import annotations

from brain.agent.executor import Plan, StepContext
from brain.agent.jev_flash import JevFlashExecutor
from brain.agent.jev_only import FAILED, JevOnlyPlanner
from brain.cancel import CancelToken
from brain.config import Config
from brain.types import OSName


class JevFirstPlanner:
    image_side = JevFlashExecutor.image_side

    def __init__(self, cfg: Config, cancel: CancelToken) -> None:
        self.flash = JevFlashExecutor(cfg, cancel)
        self.jev = self.flash.jev
        self.steps = JevOnlyPlanner(self.jev)
        self._flash_run: str | None = None  # the run Jev handed to Flash

    def next_actions(self, image: bytes, ctx: StepContext) -> Plan:
        if self._flash_run != ctx.run_id:
            plan = self.steps.next_actions(image, ctx)
            if plan.reasoning != FAILED:
                return plan
            self._flash_run = ctx.run_id
        return self.flash.next_actions(image, ctx)

    def acknowledge(self, executed: bool) -> None:
        self.flash.acknowledge(executed)

    def forget_plan(self) -> None:
        self.flash.forget_plan()

    def classify_os(self, image: bytes) -> OSName:
        return self.flash.classify_os(image)

    def close(self) -> None:
        self.flash.close()
