"""Race executor: Jev triage and the Flash planner start together on every segment.

  act   confident Jev target -> one zoom-refined click, no LLM wait; the plan is discarded
  none  a remark or context -> remembered, no input
  plan  the planner's already-running result is used, so a Jev miss costs no extra time
Irreversible requests never execute until the user approves them in the next turn.
Measured in bench/arch: 89% end-to-end vs 53% for the planner alone, 0 unsafe actions.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

from brain.agent.executor import Plan, StepContext
from brain.agent.jev_flash import JevFlashExecutor
from brain.agent.triage import approves, triage
from brain.cancel import Cancelled, CancelToken
from brain.config import Config
from brain.types import Action, OSName


class RaceExecutor:
    image_grid = False

    def __init__(self, cfg: Config, cancel: CancelToken) -> None:
        self.planner = JevFlashExecutor(cfg, cancel)
        self.jev = self.planner.jev
        self.cancel = cancel
        # One worker: a speculative plan for an earlier segment always finishes before
        # the next one starts, so the planner's state is never touched concurrently.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="race-planner")
        self._plan: Future[Plan] | None = None
        self._key: tuple | None = None
        self.mode = ""
        self._point: tuple[float, float] | None = None
        self._gesture = "click"
        self._clicked = False
        self._pending: str | None = None  # irreversible request awaiting the user's approval

    @property
    def verifies_stages(self) -> bool:
        return self.mode == "plan"

    def acknowledge(self, executed: bool) -> None:
        if self.mode == "plan":
            self.planner.acknowledge(executed)
        else:
            self._clicked = executed

    def next_actions(self, image: bytes, ctx: StepContext) -> Plan:
        key = (ctx.run_id, ctx.goal)
        if key != self._key:
            self._key, self._clicked = key, False
            return self._start(image, ctx)
        if self.mode == "act":
            if not self._clicked:
                return self._direct()
            if "unchanged" not in (ctx.note or ""):
                return Plan("Jev direct action done", [Action("done", summary="Done.")], [])
            self.mode = "plan"  # the click had no visible effect: let the planner recover
        self._settle_speculation()
        return self.planner.next_actions(image, ctx)

    def _start(self, image: bytes, ctx: StepContext) -> Plan:
        utterance = ctx.goal.splitlines()[-1]
        confirmed = False
        if self._pending is not None:
            pending, self._pending = self._pending, None
            if not self.cancel.run(lambda: approves(self.jev, image, pending, utterance, run_id=ctx.run_id)):
                self.mode = "none"
                return Plan("Irreversible request declined", [Action("done", summary="Okay, I won't do that.")], [])
            utterance, confirmed = pending, True
        # The single planner worker queues this behind any discarded speculation, so triage
        # (the fast path) never waits for an old plan to finish.
        self._plan = self._pool.submit(self.planner.next_actions, image, ctx)
        result = self.cancel.run(lambda: triage(self.jev, image, utterance, self.cancel, run_id=ctx.run_id))
        if result.irreversible and not confirmed:
            self.mode, self._pending = "none", utterance
            return Plan("Irreversible request needs approval",
                        [Action("ask_user", question=f'This can\'t be undone: "{utterance}". Should I go ahead?')], [])
        self.mode, self._point, self._gesture = result.mode, result.point, result.gesture
        if self.mode == "none":
            note = f"note_{len(ctx.memory.facts) + 1}"
            return Plan("Context noted", [Action("remember", key=note, value=utterance),
                                          Action("done", summary="Noted.")], [])
        if self.mode == "act":
            return self._direct()
        return self.cancel.run(self._plan.result)

    def _direct(self) -> Plan:
        return Plan("Jev direct action", [Action(self._gesture, x=self._point[0], y=self._point[1])], [])

    def _settle_speculation(self) -> None:
        """Wait out a discarded speculative plan before the planner is used again."""
        if self._plan is not None and not self._plan.done():
            try:
                self.cancel.run(self._plan.result)
            except Cancelled:
                raise
            except Exception:  # noqa: BLE001 - a discarded plan's own failure is irrelevant
                pass

    def classify_os(self, image: bytes) -> OSName:
        return self.planner.classify_os(image)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        self.planner.close()
