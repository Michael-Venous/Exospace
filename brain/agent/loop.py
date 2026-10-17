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
from brain.imaging import prepare
from brain.telemetry import Telemetry
from brain.types import ActionError, DeviceProfile, TERMINAL_ACTIONS
from brain.verification import visibly_changed

# A run survives these (PLAN.md §5.6): the target restarted, was replugged or blanked.
RESUMABLE = frozenset({"target disconnected", "video lost"})


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
                 reconnect: Callable[[str], DeviceProfile | None] | None = None) -> None:
        """`reconnect(reason)` waits for the target to return and gives its new profile
        (None if unchanged), or raises Cancelled; without it a disconnect ends the run."""
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.executor, self.controller, self.memory = executor, controller, memory
        self.telemetry, self.max_steps, self.on_status = telemetry, max_steps, on_status
        self.reconnect = reconnect

    def run(self, goal: str, profile: DeviceProfile, task_id: str = "interactive") -> TaskResult:
        ctl, token = self.controller, self.controller.cancel
        started = time.monotonic()
        run_id = self.telemetry.new_run()
        step = 0
        status, message = "fail", "Step limit reached"
        history: list[str] = []
        note = ""
        failures = 0
        recovery_used = False
        self.memory.set_task(goal, "running")
        self.memory.add_turn("user", goal)
        self.on_status("Working on your request…")
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
                    ctx = StepContext(goal, profile, self.memory, history, step,
                                      self.max_steps, note=note, run_id=run_id)
                    stamp = time.monotonic()
                    try:
                        plan = self.executor.next_actions(jpeg, ctx)
                    finally:
                        times["t_model"] = time.monotonic() - stamp
                    token.check()
                    self.on_status(f"Step {step}: {plan.reasoning}")
                    if plan.refresh_screen:
                        note = "Planner updated. Recheck the current screen before acting."
                        continue
                    if plan.actions:
                        latest = ctl.capture.grab()
                        if latest is None or is_blank(latest):
                            token.cancel("video lost")
                            token.check()
                        if frame_change(before, thumbnail(latest)) >= ctl.settle.threshold:
                            note = "Screen changed during inference; previous actions were discarded. Reground on this screen."
                            continue
                    notes = list(plan.rejected)
                    changed_input = False
                    action_failed = bool(plan.rejected)
                    terminal = None
                    for action in plan.actions:
                        token.check()
                        if action.type == "remember":
                            self.memory.remember(action.key, action.value)
                            history.append(action.brief())
                        elif action.type in TERMINAL_ACTIONS:
                            if action.type == "done" and changed_input:
                                notes.append("Verify the result on a fresh screen before returning done.")
                            else:
                                terminal = action
                            break
                        else:
                            stamp = time.monotonic()
                            try:
                                ctl.execute(action, profile)
                                # Waiting sends no input, so an unchanged screen is no failure.
                                changed_input = changed_input or (action.type, action.name) != ("skill", "wait")
                            except ActionError as exc:
                                notes.append(f"Rejected action: {exc}. Try another action.")
                                action_failed = True
                                break
                            finally:
                                times["t_execute"] += time.monotonic() - stamp
                            history.append(action.brief())  # before settling: a restart lands there
                            times["t_settle"] += ctl.wait_settled()
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
                    if self.reconnect is None or exc.reason not in RESUMABLE:
                        raise
                    self.on_status(f"{exc.reason.capitalize()}; waiting for it to come back…")
                    profile = self.reconnect(exc.reason) or profile
                    note = "The target restarted, reconnected or blanked; continue from this screen."
                finally:
                    self.telemetry.step(run_id, profile.os, task_id, step, **times)
        except KeyboardInterrupt:
            token.cancel("barge-in")
            status, message = "cancelled", "barge-in"
        except Cancelled as exc:
            status, message = "cancelled", exc.reason
        except Exception as exc:
            # Keep SDK exception payloads (possibly containing request data) out of transcripts.
            status, message = "fail", f"{type(exc).__name__}: runtime operation failed; check setup and model access"
        finally:
            ctl.release_all()
            self.telemetry.finish(run_id, profile.os, task_id, step, status == "done",
                                  time.monotonic() - started)
        self.memory.set_task(goal, status)
        self.memory.add_turn("assistant", message)
        return TaskResult(status, message, step, run_id)
