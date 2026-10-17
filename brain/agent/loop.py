"""Single writer for HID, with fresh-screen verification and bounded recovery."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable

from brain.agent.executor import Executor, StepContext
from brain.agent.memory import Memory
from brain.cancel import Cancelled
from brain.capture import active_rect, frame_change, is_blank, thumbnail
from brain.controller import Controller
from brain.errors import describe
from brain.events import EventBus
from brain.imaging import prepare
from brain.telemetry import Telemetry
from brain.types import Action, ActionError, DeviceProfile, Frame, POINTER_ACTIONS, TERMINAL_ACTIONS
from brain.verification import visibly_changed

MAX_STALE_DISCARDS = 1  # consecutive plans dropped because the screen moved mid-inference


@dataclass(frozen=True)
class TaskResult:
    status: str
    message: str
    steps: int
    run_id: str


class Agent:
    def __init__(self, executor: Executor, controller: Controller, memory: Memory,
                 telemetry: Telemetry, max_steps: int = 30,
                 on_status: Callable[[str], None] = lambda message: None,
                 events: EventBus | None = None,
                 resume: Callable[[], None] | None = None) -> None:
        """`resume()` waits for the picture to come back after the screen blanked on the same
        machine, or raises Cancelled; without it lost video ends the run (PLAN.md §5.6)."""
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.executor, self.controller, self.memory = executor, controller, memory
        self.telemetry, self.max_steps, self.on_status = telemetry, max_steps, on_status
        self.events, self.resume = events, resume
        self._shown_scores: object = None

    def run(self, goal: str, profile: DeviceProfile, task_id: str = "interactive") -> TaskResult:
        ctl, token = self.controller, self.controller.cancel
        started = time.monotonic()
        run_id = self.telemetry.new_run()
        step = 0
        status, message = "fail", "Step limit reached"
        history: list[str] = []
        note = ""
        failures = 0
        stale_discards = 0
        recovery_used = False
        self.memory.set_task(goal, "running")
        self.memory.add_turn("user", goal)
        self.on_status("Working on your request…")
        self._emit("run", goal=goal, run_id=run_id)
        try:
            for step in range(1, self.max_steps + 1):
                times = dict(t_capture=0.0, t_upload=0.0, t_model=0.0, t_execute=0.0, t_settle=0.0)
                try:
                    times["t_settle"] += ctl.wait_settled()
                    stamp = time.monotonic()
                    frame = ctl.capture.grab()
                    if frame is None or is_blank(frame):
                        token.cancel("video lost")
                        token.check()
                    profile = replace(profile, rect=active_rect(frame))
                    before = thumbnail(frame)
                    times["t_capture"] = time.monotonic() - stamp
                    stamp = time.monotonic()
                    jpeg = prepare(frame, profile.rect, grid=getattr(self.executor, "image_grid", True))
                    times["t_upload"] = time.monotonic() - stamp
                    self._show(frame, profile, jpeg)
                    ctx = StepContext(goal, profile, self.memory, history, step,
                                      self.max_steps, note=note, run_id=run_id)
                    stamp = time.monotonic()
                    try:
                        plan = self.executor.next_actions(jpeg, ctx)
                    finally:
                        times["t_model"] = time.monotonic() - stamp
                    token.check()
                    self.on_status(f"Step {step}: {plan.reasoning}")
                    self._emit("plan", step=step, reasoning=plan.reasoning,
                               actions=[a.brief() for a in plan.actions],
                               points=[[a.x, a.y] for a in plan.actions if a.type in POINTER_ACTIONS],
                               scores=self._jev_scores(), t_model=round(times["t_model"], 3))
                    if plan.refresh_screen:
                        note = "Planner updated. Recheck the current screen before acting."
                        continue
                    if plan.actions:
                        latest = ctl.capture.grab()
                        if latest is None or is_blank(latest):
                            token.cancel("video lost")
                            token.check()
                        moved = frame_change(before, thumbnail(latest)) >= ctl.settle.threshold
                        # A screen that never stops moving (video, spinner) would
                        # discard every plan; after one discard, act anyway.
                        if moved and stale_discards < MAX_STALE_DISCARDS:
                            stale_discards += 1
                            note = "Screen changed during inference; previous actions were discarded. Reground on this screen."
                            continue
                        stale_discards = 0
                    notes = list(plan.rejected)
                    changed_input = False
                    action_failed = bool(plan.rejected)
                    terminal = None
                    for action in plan.actions:
                        token.check()
                        if action.type == "remember":
                            self.memory.remember(action.key, action.value)
                        elif action.type in TERMINAL_ACTIONS:
                            if action.type == "done" and changed_input:
                                notes.append("Verify the result on a fresh screen before returning done.")
                            else:
                                terminal = action
                            break
                        else:
                            if changed_input and action.x is not None:
                                # Aimed at the screen the batch was planned on, e.g. a tapped field
                                # opened the keyboard over it.
                                stamp = time.monotonic()
                                latest = ctl.capture.grab()
                                times["t_capture"] += time.monotonic() - stamp
                                if latest is None or frame_change(before, thumbnail(latest)) >= ctl.settle.threshold:
                                    notes.append("The screen changed mid-batch, so the rest was skipped. "
                                                 "Reground on this screen.")
                                    break
                            stamp = time.monotonic()
                            try:
                                ctl.execute(action, profile)
                                changed_input = True
                                self._emit_action(step, action, ok=True)
                            except ActionError as exc:
                                self._emit_action(step, action, ok=False)
                                notes.append(f"Rejected action: {exc}. Try another action.")
                                action_failed = True
                                break
                            finally:
                                times["t_execute"] += time.monotonic() - stamp
                            times["t_settle"] += ctl.wait_settled()
                        history.append(action.brief())
                    if terminal is not None:
                        status = terminal.type
                        message = terminal.summary or terminal.question or terminal.reason or status
                        break
                    acknowledge = getattr(self.executor, "acknowledge", None)
                    if callable(acknowledge):
                        acknowledge(bool(plan.actions) and not action_failed)
                    if changed_input:
                        after = ctl.capture.grab()
                        if after is None or is_blank(after):
                            token.cancel("video lost")
                            token.check()
                        unchanged = not visibly_changed(frame, after)
                        if unchanged:
                            notes.append("Screen unchanged; try something different.")
                        # A covered result can leave pixels unchanged. The Jev
                        # executor checks its semantic postcondition next turn;
                        # do not count the same pending result as a second failure.
                        if getattr(self.executor, "verifies_stages", False) is not True:
                            action_failed = action_failed or unchanged
                    elif not plan.actions:
                        notes.append("No executable actions returned.")
                        action_failed = True
                    if action_failed:
                        failures += 1
                        if failures >= 3:
                            token.cancel("3 failed verifications")
                            token.check()
                        if failures >= 2 and not recovery_used:
                            notes.append("Recovery required: reconsider the plan from this screen.")
                            recovery_used = True
                    else:
                        failures, recovery_used = 0, False
                    note = "\n".join(notes)
                except Cancelled as exc:
                    if self.resume is None or exc.reason != "video lost":
                        raise
                    self.on_status("The screen went blank; waiting for it to come back…")
                    self.resume()
                    note = "The screen blanked and came back; continue from this screen."
                finally:
                    self.telemetry.step(run_id, profile.os, task_id, step, **times)
        except KeyboardInterrupt:
            token.cancel("barge-in")
            status, message = "cancelled", "barge-in"
        except Cancelled as exc:
            status, message = "cancelled", exc.reason
        except Exception as exc:
            status, message = "fail", describe(exc, "task")
        finally:
            ctl.release_all()
            self.telemetry.finish(run_id, profile.os, task_id, step, status == "done",
                                  time.monotonic() - started)
        self.memory.set_task(goal, status)
        self.memory.add_turn("assistant", message)
        self._emit("result", status=status, message=message, steps=step,
                   total_s=round(time.monotonic() - started, 3))
        return TaskResult(status, message, step, run_id)

    # -- display events (no-ops without a display) -------------------------

    def _emit(self, ev: str, **fields: object) -> None:
        if self.events is not None:
            self.events.publish(ev, **fields)

    def _emit_action(self, step: int, action: Action, *, ok: bool) -> None:
        point = [action.x, action.y] if action.type in POINTER_ACTIONS else None
        self._emit("action", step=step, action=action.brief(), point=point, ok=ok)

    def _show(self, frame: Frame, profile: DeviceProfile, model_jpeg: bytes) -> None:
        """Share the step's screen. The model copy is reused unless it has an aiming grid."""
        if self.events is None:
            return
        grid = getattr(self.executor, "image_grid", True)
        self.events.set_frame(prepare(frame, profile.rect, grid=False) if grid else model_jpeg)

    def _jev_scores(self) -> dict[str, object] | None:
        """Scores from Jev's latest check, or None if this step made no new check
        (Jev replaces `last_scores` with a new dict on every call)."""
        scores = getattr(getattr(self.executor, "jev", None), "last_scores", None)
        if not isinstance(scores, dict) or scores is self._shown_scores:
            return None
        self._shown_scores = scores
        return dict(scores)
