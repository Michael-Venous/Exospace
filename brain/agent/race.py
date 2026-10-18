"""Race executor: Jev triage and the Flash planner start together on every segment.

  act   confident Jev target -> one zoom-refined click, no LLM wait; the plan is discarded
  none  a remark or context -> remembered, no input
  plan  the planner's already-running result is used, so a Jev miss costs no extra time
Irreversible requests never execute until the user approves them in the next turn.
Measured in bench/arch: 89% end-to-end vs 53% for the planner alone, 0 unsafe actions.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace

from brain.agent.executor import Plan, StepContext
from brain.agent.jev_flash import COMPLETION_GATE, JevFlashExecutor
from brain.agent.triage import answer_to_pending, triage
from brain.cancel import Cancelled, CancelToken
from brain.config import Config
from brain.types import Action, OSName


@dataclass(frozen=True)
class _Pending:
    """An irreversible request waiting for the user's answer, bound to the device that asked."""

    utterance: str
    goal: str
    connected_at: float


class RaceExecutor:
    image_grid = False
    image_side = JevFlashExecutor.image_side

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
        self._scroll = 0
        self._clicked = False
        self._utterance = ""
        self._triage_step = 0
        self._pending: _Pending | None = None
        self._approved_goal: str | None = None  # an approved run plans only what was approved

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
            self._key, self._clicked, self._approved_goal = key, False, None
            return self._start(image, ctx)
        if self.mode == "scroll":
            # "scroll down to the comments" also triages as a scroll: done only if the goal is met.
            if "unchanged" not in (ctx.note or "") and self._goal_met(image, ctx):
                return Plan("Scrolled", [Action("done", summary="Scrolled.")], [])
            self.mode = "plan"  # nothing to scroll here (or not focused): the planner recovers
        if self.mode == "act":
            if not self._clicked and ctx.step != self._triage_step:
                # The loop discarded the click (the screen moved): aim again on this screen.
                return self._route(image, ctx, self._utterance, confirmed=True)
            if not self._clicked:
                return self._direct()
            if "unchanged" not in (ctx.note or "") and self._goal_met(image, ctx):
                return Plan("Jev direct action done", [Action("done", summary="Done.")], [])
            self.mode = "plan"  # no visible effect, or not the goal (a neighbouring button): the planner recovers
        self._settle_speculation()
        return self.planner.next_actions(image, self._planner_ctx(ctx))

    def _start(self, image: bytes, ctx: StepContext) -> Plan:
        utterance = ctx.goal.splitlines()[-1]
        pending, self._pending = self._pending, None
        if pending is not None and pending.connected_at == ctx.profile.connected_at:
            answer = self.cancel.run(lambda: answer_to_pending(self.jev, image, pending.utterance, utterance,
                                                               run_id=ctx.run_id))
            if answer == "decline":
                # "No" alone is just a refusal; "no, open John's email instead" also asks for something.
                self._approved_goal = None
                return self._route(image, ctx, utterance, confirmed=False, declined=True)
            if answer == "approve":
                self._approved_goal = pending.goal + "\nThe user approved this request."
                return self._route(image, ctx, pending.utterance, confirmed=True)
            # Neither: a new command or a correction. Handle it as its own request.
        self._approved_goal = None
        return self._route(image, ctx, utterance, confirmed=False)

    def _route(self, image: bytes, ctx: StepContext, utterance: str, *, confirmed: bool,
               declined: bool = False) -> Plan:
        self._utterance, self._triage_step = utterance, ctx.step
        # The single planner worker queues this behind any discarded speculation, so triage
        # (the fast path) never waits for an old plan to finish.
        self._plan = self._pool.submit(self.planner.next_actions, image, self._planner_ctx(ctx))
        result = self.cancel.run(lambda: triage(self.jev, image, utterance, self.cancel, run_id=ctx.run_id))
        if result.irreversible and not confirmed:
            return self.hold(ctx, f'"{utterance}"')
        self.mode, self._point, self._gesture, self._scroll = result.mode, result.point, result.gesture, result.scroll
        if self.mode == "none" and declined:
            return Plan("Irreversible request declined", [Action("done", summary="Okay, I won't do that.")], [])
        if self.mode == "none":
            note = f"note_{len(ctx.memory.facts) + 1}"
            return Plan("Context noted", [Action("remember", key=note, value=utterance),
                                          Action("done", summary="Noted.")], [])
        if self.mode == "act":
            return self._direct()
        if self.mode == "scroll":  # no target and no plan needed (it took 10.8 s through the planner)
            return Plan("Jev scroll", [Action("scroll", dy=result.scroll)], [])
        return self.cancel.run(self._plan.result)

    def approved_run(self) -> bool:
        return self._approved_goal is not None

    def hold(self, ctx: StepContext, what: str) -> Plan:
        """Ask before an irreversible request or step; the user's next reply answers it."""
        self.mode = "none"
        self._pending = _Pending(ctx.goal.splitlines()[-1], ctx.goal, ctx.profile.connected_at)
        return Plan("Irreversible request needs approval",
                    [Action("ask_user", question=f"This can't be undone: {what}. Should I go ahead?")], [])

    def _goal_met(self, image: bytes, ctx: StepContext) -> bool:
        """One Jev completion call on the new screen: a changed screen alone is no proof
        (measured: "close the calculator" minimized it and reported done)."""
        executed = ([f"scroll {'down' if self._scroll > 0 else 'up'} {abs(self._scroll)} notches"]
                    if self.mode == "scroll" else [f"{self._gesture} on the control for: {self._utterance}"])
        score = self.cancel.run(lambda: self.jev.completion(image, ctx.goal, executed, run_id=ctx.run_id,
                                                            step=ctx.step))
        return score >= COMPLETION_GATE

    def _planner_ctx(self, ctx: StepContext) -> StepContext:
        return replace(ctx, goal=self._approved_goal) if self._approved_goal else ctx

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
