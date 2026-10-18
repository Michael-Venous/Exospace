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
from brain.agent.jev import probability
from brain.agent.jev_first import JevFirstPlanner
from brain.agent.jev_flash import JevFlashExecutor
from brain.agent.triage import GATE, answer_to_pending, pointed, triage
from brain.cancel import Cancelled, CancelToken
from brain.skills import OPEN_APP_KEYS
from brain.config import Config
from brain.types import Action, OSName


DIRECT_GATE = 0.55  # completion after a direct click, scroll or launch
# Where to hover next when the pointer rests on the wrong control: Jev re-aimed onto the same wrong one, but
# its check of the hovered control is sure (close 0.97, neighbours <= 0.21), so the neighbours are tried.
# 0.01 of the width is about one window button on a 2880 px screen.
NEIGHBOURS = ((.01, 0), (-.01, 0), (0, .015), (0, -.015), (.02, 0), (-.02, 0))
LAUNCHED = ("Is the app the `utterance` asks to open now open in front as its own app window: not a web search "
            "page about it, and not a different app with a similar name?")


@dataclass(frozen=True)
class _Pending:
    """An irreversible request waiting for the user's answer, bound to the device that asked."""

    utterance: str
    goal: str
    connected_at: float
    step: str = ""


class RaceExecutor:
    image_grid = False
    image_side = JevFlashExecutor.image_side

    def __init__(self, cfg: Config, cancel: CancelToken) -> None:
        if cfg.executor == "jev":  # speed first: Jev's steps only, a step Jev can't take ends the request
            self.planner = JevFirstPlanner(cfg, cancel, approved_run=self.approved_run, handoff=False)
        else:
            planner = JevFirstPlanner if cfg.jev_first else JevFlashExecutor
            self.planner = planner(cfg, cancel, approved_run=self.approved_run)
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
        self._launched = ""
        self._tries = 0
        self._aim: tuple[float, float] = (0.0, 0.0)  # Jev's point for a small target; hovers try around it
        self._clicked = False
        self._utterance = ""
        self._earlier = ""
        self._triage_step = 0
        self._pending: _Pending | None = None
        self._approved_goal: str | None = None  # an approved run plans only what was approved
        self._approved_connected_at = 0.0

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
        if self.mode == "hover":
            return self._after_hover(image, ctx)
        if self.mode == "launch":
            launched = "unchanged" not in (ctx.note or "") and self._launched_app(image, ctx)
            if launched and self._goal_met(image, ctx):
                return Plan("Jev launch done", [Action("done", summary="Done.")], [])
            self.mode = "plan"  # more to do in the app ("... and work out 12 times 9"): plan on its screen
            self._settle_speculation()
            self.planner.forget_plan()  # the speculative plan was made before the launch
            if not launched:
                ctx = replace(ctx, note=ctx.note or f'Opening "{self._launched}" from the launcher did not open '
                                                    "that app; find it another way.")
            return self._plan_now(image, ctx)
        if self.mode == "act":
            if not self._clicked and ctx.step != self._triage_step:
                # The loop discarded the click (the screen moved): aim again on this screen.
                return self._route(image, ctx, self._utterance, self._earlier, confirmed=self.approved_run(ctx))
            if not self._clicked:
                return self._direct()
            if "unchanged" not in (ctx.note or "") and self._goal_met(image, ctx):
                return Plan("Jev direct action done", [Action("done", summary="Done.")], [])
            self.mode = "plan"  # no visible effect, or not the goal (a neighbouring button): the planner recovers
        self._settle_speculation()
        return self._plan_now(image, ctx)

    def _start(self, image: bytes, ctx: StepContext) -> Plan:
        *lines, utterance = ctx.goal.splitlines()
        earlier = "\n".join(lines)
        pending, self._pending = self._pending, None
        if pending is not None and pending.connected_at == ctx.profile.connected_at:
            answer = self.cancel.run(lambda: answer_to_pending(self.jev, image, pending.utterance, utterance,
                                                               held=pending.step, run_id=ctx.run_id))
            if answer == "decline":
                # "No" alone is just a refusal; "no, open John's email instead" also asks for something.
                self._approved_goal = None
                return self._route(image, ctx, utterance, earlier="", confirmed=False, declined=True)
            if answer == "approve":
                self._approved_goal = pending.goal + "\nThe user approved this request."
                self._approved_connected_at = pending.connected_at
                if pending.step:
                    self.mode = "plan"
                    self._settle_speculation()
                    return self._plan_now(image, ctx)
                return self._route(image, ctx, pending.utterance, confirmed=True)
            # Neither: a new command or a correction. Handle it as its own request.
        self._approved_goal = None
        return self._route(image, ctx, utterance, earlier, confirmed=False)

    def _route(self, image: bytes, ctx: StepContext, utterance: str, earlier: str = "", *, confirmed: bool,
               declined: bool = False) -> Plan:
        self._utterance, self._earlier, self._triage_step = utterance, earlier, ctx.step
        # The single planner worker queues this behind any discarded speculation, so triage
        # (the fast path) never waits for an old plan to finish.
        self._plan = self._pool.submit(self.planner.next_actions, image, self._planner_ctx(ctx))
        result = self.cancel.run(lambda: triage(self.jev, image, utterance, self.cancel, earlier=earlier,
                                                run_id=ctx.run_id))
        if result.irreversible and not confirmed:
            return self.hold(ctx)
        self.mode, self._point, self._gesture, self._scroll = result.mode, result.point, result.gesture, result.scroll
        if self.mode == "act" and result.app and ctx.profile.input == "touch" and ctx.profile.os in OPEN_APP_KEYS:
            # Tapping an app's icon did nothing on the demo phone (its touch input doesn't register); the
            # keyboard launcher opens the app the request names.
            self.mode = "launch"
            result = replace(result, mode="launch")
        if self.mode == "launch" and ctx.profile.os not in OPEN_APP_KEYS:
            # A phone has no launcher key: the launch was rejected and its note sent the whole request to
            # Flash (6 s a step on Android); Jev's own steps tap the app's icon instead.
            self.mode = "plan"
        if self.mode == "none" and declined:
            return Plan("Irreversible request declined", [Action("done", summary="Okay, I won't do that.")], [])
        if self.mode == "none":
            note = f"note_{len(ctx.memory.facts) + 1}"
            return Plan("Context noted", [Action("remember", key=note, value=utterance),
                                          Action("done", summary="Noted.")], [])
        if self.mode == "act" and result.small and ctx.profile.input == "mouse":
            self._aim = result.point
            return self._hover(0)
        if self.mode == "act":
            return self._direct()
        if self.mode == "scroll":  # no target and no plan needed (it took 10.8 s through the planner)
            return Plan("Jev scroll", [Action("scroll", dy=result.scroll)], [])
        if self.mode == "launch":
            self._launched = result.app
            return Plan("Jev launch", [Action("skill", name="open_app", args={"name": result.app})], [])
        return self._planned(self.cancel.run(self._plan.result), ctx)

    def approved_run(self, ctx: StepContext) -> bool:
        return self._approved_goal is not None and ctx.profile.connected_at == self._approved_connected_at

    def hold(self, ctx: StepContext, step: str = "") -> Plan:
        """Ask before an irreversible request, or before one `step` of it; the user's next reply answers it."""
        self.mode = "none"
        utterance = ctx.goal.splitlines()[-1]
        self._pending = _Pending(utterance, ctx.goal, ctx.profile.connected_at, step)
        what = step or f'"{utterance}"'
        return Plan("Irreversible request needs approval",
                    [Action("ask_user", question=f"This can't be undone: {what}. Should I go ahead?")], [])

    def _plan_now(self, image: bytes, ctx: StepContext) -> Plan:
        return self._planned(self.planner.next_actions(image, self._planner_ctx(ctx)), ctx)

    def _planned(self, plan: Plan, ctx: StepContext) -> Plan:
        return self.hold(ctx, plan.held) if plan.held else plan

    def _goal_met(self, image: bytes, ctx: StepContext) -> bool:
        """One Jev completion call on the new screen: a changed screen alone is no proof
        (measured: "close the calculator" minimized it and reported done)."""
        executed = ([f"scroll {'down' if self._scroll > 0 else 'up'} {abs(self._scroll)} notches"]
                    if self.mode == "scroll" else [f"open the app {self._launched}"]
                    if self.mode == "launch" else [f"{self._gesture} on the control for: {self._utterance}"])
        score = self.cancel.run(lambda: self.jev.completion(image, ctx.goal, executed, run_id=ctx.run_id,
                                                            step=ctx.step))
        # Direct-click scores vs the real outcome: true 0.56-0.93 (two at 0.50 and 0.53), false 0.54 ("close the
        # calculator" clicked the screen's corner). Borderline goes to the planner rather than ending the run.
        return score >= DIRECT_GATE

    def _hover(self, tries: int) -> Plan:
        """Rest the pointer on a small target first: its highlight and tooltip confirm it next step."""
        self.mode, self._tries = "hover", tries
        x, y = self._aim
        if tries:
            dx, dy = NEIGHBOURS[tries - 1]
            x, y = min(max(x + dx, 0.0), 1.0), min(max(y + dy, 0.0), 1.0)
        self._point = (x, y)
        return Plan("Jev hover check", [Action("skill", name="hover", args={"x": x, "y": y})], [])

    def _after_hover(self, image: bytes, ctx: StepContext) -> Plan:
        if self.cancel.run(lambda: pointed(self.jev, image, self._utterance, self._point, run_id=ctx.run_id,
                                           step=ctx.step)):
            self.mode, self._clicked, self._triage_step = "act", False, ctx.step
            return self._direct()
        if self._tries < len(NEIGHBOURS):
            return self._hover(self._tries + 1)
        self.mode = "plan"  # never click a small control Jev could not confirm
        self._settle_speculation()
        note = f"No control near where Jev aimed does what was asked ({self._utterance}); find it another way."
        return self._plan_now(image, replace(ctx, note=ctx.note or note))

    def _launched_app(self, image: bytes, ctx: StepContext) -> bool:
        """The launcher can land elsewhere: "abre la calculadora" on an English desktop became a web
        search with a calculator widget, which the plain completion check passed (0.9)."""
        answers, _, _ = self.cancel.run(lambda: self.jev.evaluate(image, {"launched": {"type": "noul", "instructions": LAUNCHED}},
                                                                  run_id=ctx.run_id, step=ctx.step,
                                                                  state={"utterance": self._utterance}))
        return probability(answers["launched"].get("noul")) >= GATE

    def _planner_ctx(self, ctx: StepContext) -> StepContext:
        return replace(ctx, goal=self._approved_goal) if self.approved_run(ctx) else ctx

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
