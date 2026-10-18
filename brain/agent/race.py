"""Race executor: Jev triage and the Flash planner start together on every segment.

  act   confident Jev target -> one zoom-refined click, no LLM wait; the plan is discarded
  keys  (JEV_SOLO_SKILLS=1) easy tasks (open app/site, type, key, scroll, find), no LLM wait
With solo skills, a chained request ("open chrome and go to youtube") runs one instruction at
a time: after each works, the remaining words are triaged on the new screen, and the planner
only ever sees what's left. With JEV_LEARNED_CACHE=1, a decision that worked replays on the same request and screen
with no Jev call at all; a replay that changes nothing is forgotten (brain/agent/learned.py).
  none  a remark or context -> remembered, no input
  plan  the planner's already-running result is used, so a Jev miss costs no extra time
Irreversible requests never execute until the user approves them in the next turn.
Measured in bench/arch: 89% end-to-end vs 53% for the planner alone, 0 unsafe actions.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace

from brain.agent.executor import Plan, StepContext
from brain.agent.jev_flash import JevFlashExecutor
from brain.agent.learned import LearnedCache
from brain.agent.solo import solo_actions
from brain.agent.triage import Triage, answer_to_pending, triage
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
    solo = False  # JEV_SOLO_SKILLS: triage may route easy keyboard tasks without the planner
    learned: LearnedCache | None = None  # JEV_LEARNED_CACHE: replay decisions that worked
    _decided: tuple[str, str, bytes, Triage, bool] | None = None  # os, utterance, screen, decision, replayed
    _rest = ""  # words of a chained request still to do after the current instruction
    _chained = False  # the current instruction is a later part of a chained request

    def __init__(self, cfg: Config, cancel: CancelToken) -> None:
        self.planner = JevFlashExecutor(cfg, cancel)
        self.jev = self.planner.jev
        self.cancel = cancel
        self.solo = cfg.jev_solo_skills
        self.learned = LearnedCache(cfg.learned_path) if cfg.jev_learned_cache else None
        self._keys: list[Action] = []
        # One worker: a speculative plan for an earlier segment always finishes before
        # the next one starts, so the planner's state is never touched concurrently.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="race-planner")
        self._plan: Future[Plan] | None = None
        self._key: tuple | None = None
        self.mode = ""
        self._point: tuple[float, float] | None = None
        self._gesture = "click"
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
            self._key, self._clicked, self._approved_goal, self._chained = key, False, None, False
            return self._start(image, ctx)
        if self.mode in ("act", "keys"):
            if not self._clicked and ctx.step != self._triage_step:
                # The loop discarded the input (the screen moved): decide again on this screen.
                return self._route(image, ctx, self._utterance, confirmed=True)
            if not self._clicked:
                return self._direct() if self.mode == "act" else self._solo()
            if "unchanged" not in (ctx.note or ""):
                self._learn(worked=True)
                if self._rest:  # a chained request: triage what's left on this new screen
                    rest, self._clicked, self._chained = self._rest, False, True
                    return self._route(image, ctx, rest, confirmed=False)
                return Plan("Jev direct action done", [Action("done", summary="Done.")], [])
            self._learn(worked=False)
            self.mode = "plan"  # the click had no visible effect: let the planner recover
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
        replay = (self.learned.lookup(ctx.profile.os, utterance, image)
                  if self.learned is not None and not declined and self._approved_goal is None else None)
        if replay is not None:
            self._plan, result = None, replay  # a remembered decision: no Jev call, no planner
        else:
            # The single planner worker queues this behind any discarded speculation, so triage
            # (the fast path) never waits for an old plan to finish.
            self._plan = self._pool.submit(self.planner.next_actions, image, self._planner_ctx(ctx))
            result = self.cancel.run(lambda: triage(self.jev, image, utterance, self.cancel, solo=self.solo,
                                                    run_id=ctx.run_id))
        self._decided = (ctx.profile.os, utterance, image, result, replay is not None)
        label = "Remembered: " if replay is not None else ""
        if result.irreversible and not confirmed:
            self.mode = "none"
            self._pending = _Pending(utterance, ctx.goal, ctx.profile.connected_at)
            return Plan("Irreversible request needs approval",
                        [Action("ask_user", question=f'This can\'t be undone: "{utterance}". Should I go ahead?')], [])
        self.mode, self._point, self._gesture, self._rest = result.mode, result.point, result.gesture, result.rest
        if self.mode == "none" and declined:
            return Plan("Irreversible request declined", [Action("done", summary="Okay, I won't do that.")], [])
        if self.mode == "none":
            note = f"note_{len(ctx.memory.facts) + 1}"
            return Plan("Context noted", [Action("remember", key=note, value=utterance),
                                          Action("done", summary="Noted.")], [])
        if self.mode == "act":
            return self._direct(label)
        if self.mode == "keys":
            self._keys = solo_actions(result.skill, result.text, ctx.profile.os, result.point) or []
            if self._keys:
                return self._solo(f"{result.skill} {result.text!r}", label)
            self.mode = "plan"  # this OS has no keyboard route for it
        if self._plan is None:
            self._plan = self._pool.submit(self.planner.next_actions, image, self._planner_ctx(ctx))
        return self.cancel.run(self._plan.result)

    def _planner_ctx(self, ctx: StepContext) -> StepContext:
        if self._approved_goal:
            return replace(ctx, goal=self._approved_goal)
        if self._chained:  # earlier instructions are done: plan only the words still to do
            return replace(ctx, goal="\n".join([*ctx.goal.splitlines()[:-1], self._utterance]))
        return ctx

    def _direct(self, label: str = "") -> Plan:
        return Plan(f"{label}Jev direct action", [Action(self._gesture, x=self._point[0], y=self._point[1])], [])

    def _solo(self, what: str = "", label: str = "") -> Plan:
        return Plan(f"{label}Jev solo skill {what}".rstrip(), list(self._keys), [])

    def _learn(self, *, worked: bool) -> None:
        """Remember a fresh decision that worked; forget a remembered one that didn't."""
        if self.learned is None or self._decided is None:
            return
        os_name, utterance, image, decision, replayed = self._decided
        self._decided = None
        if worked and not replayed:
            self.learned.store(os_name, utterance, image, decision)
        elif not worked and replayed:
            self.learned.forget(os_name, utterance, image)

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
