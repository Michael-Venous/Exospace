"""Candidate: the runtime race, with a pointing model where Jev refuses a planned stage's target.

In the race a Flash stage whose target Jev can't place (locate + zoom below its gates) is planned
again by Flash with visual-fallback coordinates, a ~3 s call. Here Perceptron Mk1.5 places that target
instead (bench/arch/pointer.py); a stage whose target it doesn't see either replans as before.
Everything else is the runtime's own code. Bench only: the runtime change would be in
brain/agent/jev_flash.py (Agent owner).
"""

from __future__ import annotations

from brain.agent.jev_flash import JevFlashExecutor
from brain.agent.race import RaceExecutor
from brain.cancel import CancelToken
from brain.config import Config

from bench.arch.pointer import Pointer

POINTED = ("target", "destination")  # a stage's pointer targets; a typing field or window button is left to Jev


class PointerFlash(JevFlashExecutor):
    pointer: Pointer

    def _zoom(self, image, targets, points, ctx):
        out = super()._zoom(image, targets, points, ctx)
        for key in POINTED:
            if key in out and out[key] is None:
                out[key] = self.pointer.point(image, targets[key])
        return out


class RacePointerExecutor(RaceExecutor):
    def __init__(self, cfg: Config, cancel: CancelToken) -> None:
        super().__init__(cfg, cancel)
        flash = getattr(self.planner, "flash", self.planner)  # JEV_FIRST=0 plans with Flash's executor itself
        # The runtime builds its Flash stage executor inside the planner; it keeps its state, gains the pointer.
        flash.__class__ = PointerFlash
        flash.pointer = Pointer(cancel, flash.jev)

    def close(self) -> None:
        getattr(self.planner, "flash", self.planner).pointer.close()
        super().close()
