"""Single writer for HID, with fresh-screen verification and bounded recovery."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from brain.agent.executor import Executor, StepContext
from brain.agent.memory import Memory
from brain.cancel import Cancelled
from brain.capture import active_rect, area_change, frame_change, is_blank, thumbnail
from brain.controller import Controller
from brain.errors import describe
from brain.events import EventBus
from brain.imaging import MAX_SIDE, prepare
from brain.telemetry import Telemetry
from brain.types import Action, ActionError, DeviceProfile, Frame, POINTER_ACTIONS, TERMINAL_ACTIONS
from brain.verification import visibly_changed

STALL_S = 20.0  # no visible progress for this long ends the run (a demo must never look frozen)
# A run survives these (PLAN.md §5.6): the target restarted, was replugged or blanked.
RESUMABLE = frozenset({"target disconnected", "video lost"})
# Plans in a row discarded for a change where they aimed before the spot counts as moving: a preview YouTube played
# under the pointer discarded 48 clicks on "the third video" until the step limit (2026-10-19).
MAX_DISCARDS = 2
# How much the area a plan aims at must change before the plan is discarded. Of 8 discards (2026-10-18..20), the 6
# whose replan clicked the same spot changed it 1.56-2.64 (a search box restyling as YouTube drew, a hover highlight),
# each costing a ~1.1 s replan; the 2 whose target moved changed it 133-173. The settle threshold (1.5) is a whole-screen
# measure. shortcut: 8 samples; recheck with the discard listing if a kept plan clicks a moved target.
AREA_MOVED = 6.0


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
                 reconnect: Callable[[str], DeviceProfile | None] | None = None,
                 stall_s: float = STALL_S, more_words: Callable[[], str | None] = lambda: None) -> None:
        """`reconnect(reason)` waits for the target to return and gives its new profile
        (None if unchanged), or raises Cancelled; without it a disconnect ends the run.
        `more_words()` gives what the user said since, in the same breath, or None: it joins the
        running request at the next step instead of cancelling it ("open chrome" ... "and search cats")."""
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.executor, self.controller, self.memory = executor, controller, memory
        self.telemetry, self.max_steps, self.on_status = telemetry, max_steps, on_status
        self.events = events
        self.reconnect = reconnect
        self.stall_s = stall_s
        self.more_words = more_words
        self.goal = ""  # the running request, with any words that joined it

    def run(self, goal: str, profile: DeviceProfile, task_id: str = "interactive") -> TaskResult:
        ctl, token = self.controller, self.controller.cancel
        started = time.monotonic()
        run_id = self.telemetry.new_run()
        step = 0
        status, message = "fail", "Step limit reached"
        history: list[str] = []
        note = ""
        failures = 0
        unchanged_steps = 0  # consecutive steps whose input left the screen as it was
        recovery_used = False
        attempts: dict[tuple[str, ...], list[np.ndarray]] = {}  # each batch's starting screens
        settled_s: float | None = None  # how long the last action's settle took; None = settle first
        discarded = 0  # plans in a row discarded for a screen change
        self.memory.set_task(goal, "running")
        self.memory.add_turn("user", goal)
        self.on_status("Working on your request…")
        self._emit("run", goal=goal, run_id=run_id)
        try:
            progress_at = time.monotonic()
            for step in range(1, self.max_steps + 1):
                extra = self.more_words()
                if extra:
                    goal = f"{goal}\nUser correction: {extra}"
                    self.memory.add_turn("user", extra)
                    progress_at = time.monotonic()  # new words: the request itself moved on
                self.goal = goal
                override = getattr(self.executor, "stall_s", None)  # deep mode allows slower steps
                stall_s = override if isinstance(override, (int, float)) and override else self.stall_s
                if stall_s and time.monotonic() - progress_at > stall_s:
                    # No visible progress for a while is a failure the user should hear about, not a silent grind.
                    status, message = "fail", "I'm stuck on this one and stopped. Could you try it another way?"
                    break
                times = dict(t_capture=0.0, t_upload=0.0, t_model=0.0, t_execute=0.0, t_settle=0.0)
                try:
                    limit = ctl.settle.timeout_s
                    if settled_s is None:  # otherwise the previous step's last action already settled
                        limit = ctl.settle.look_s
                        settled_s = ctl.wait_settled(react=False, timeout_s=limit)  # no input was just sent
                        times["t_settle"] += settled_s
                    # A screen still moving at the settle timeout (a live camera preview, a video) differs at
                    # every grab: judging it stale discarded nearly every plan (measured on Android's camera).
                    moving = settled_s >= limit
                    settled_s = None
                    stamp = time.monotonic()
                    frame = ctl.capture.grab()
                    if frame is None or is_blank(frame):
                        token.cancel("video lost")
                        token.check()
                    before = thumbnail(frame)
                    times["t_capture"] = time.monotonic() - stamp
                    if profile.input == "touch":  # a phone turns; a computer's screen was found at connect
                        profile = replace(profile, rect=active_rect(frame))
                    stamp = time.monotonic()
                    side = getattr(self.executor, "image_side", MAX_SIDE)
                    jpeg = prepare(frame, profile.rect, grid=getattr(self.executor, "image_grid", True),
                                   max_side=side if isinstance(side, int) else MAX_SIDE)
                    times["t_upload"] = time.monotonic() - stamp
                    frame_id = self._show(frame, profile, jpeg)
                    ctx = StepContext(goal, profile, self.memory, history, step,
                                      self.max_steps, note=note, run_id=run_id)
                    stamp = time.monotonic()
                    try:
                        plan = self.executor.next_actions(jpeg, ctx)
                    finally:
                        times["t_model"] += time.monotonic() - stamp
                    token.check()
                    why = f" ({plan.rejected[0][:160]})" if plan.rejected and not plan.actions else ""
                    self.on_status(f"Step {step}: {plan.reasoning}{why}")
                    self._emit("plan", step=step, reasoning=plan.reasoning,
                               actions=[a.brief() for a in plan.actions],
                               points=[[a.x, a.y] for a in plan.actions if a.type in POINTER_ACTIONS],
                               rejected=list(plan.rejected), frame=frame_id, trace=self._trace(),
                               t_model=round(times["t_model"], 3))
                    if plan.refresh_screen:
                        note = "Planner updated. Recheck the current screen before acting."
                        continue
                    # A plan that only says done aims at nothing: a Short playing on is no reason to doubt it, and
                    # discarding each "done" Jev saw for "click the third video" left it scrolling (2026-10-19).
                    # Nor does a blind one: Brave's restored page drawing in discarded "Ctrl+L, youtube" (v5-t1).
                    # What the input is judged against: Google's results finished drawing between the screen a
                    # click on their Wikipedia link was planned on and the click, which did nothing, and that
                    # drawing read as its effect (2026-10-19).
                    sent_to = frame
                    if plan.actions and plan.actions[0].type != "done" and not plan.blind:
                        latest = sent_to = ctl.capture.grab()
                        if latest is None or is_blank(latest):
                            token.cancel("video lost")
                            token.check()
                        # Only what the first action aims at must be where the plan saw it: Google's results
                        # drawing in below the address bar a plan clicked discarded that plan (2026-10-18).
                        first = plan.actions[0]
                        moved = (area_change(frame, latest, profile.rect, first.x, first.y) >= AREA_MOVED
                                 if first.x is not None and first.y is not None
                                 else ctl.changed(before, thumbnail(latest)))
                        if not moving and moved and discarded < MAX_DISCARDS:
                            discarded += 1
                            note = "Screen changed during inference; previous actions were discarded. Reground on this screen."
                            progress_at += times["t_model"]  # a plan never tried: its thinking time isn't a stall
                            continue
                    discarded = 0
                    notes = list(plan.rejected)
                    acted = sent_input = False
                    action_failed = bool(plan.rejected)
                    terminal = pending_done = None
                    for i, action in enumerate(plan.actions):
                        token.check()
                        if action.type == "remember":
                            self.memory.remember(action.key, action.value)
                            history.append(action.brief())
                        elif action.type in TERMINAL_ACTIONS:
                            if action.type == "done" and acted:
                                pending_done = action
                            else:
                                terminal = action
                            break
                        else:
                            if acted and action.x is not None and not moving:
                                # Aimed at the screen the batch was planned on, e.g. a tapped field
                                # opened the keyboard over it.
                                stamp = time.monotonic()
                                latest = ctl.capture.grab()
                                times["t_capture"] += time.monotonic() - stamp
                                if latest is None or ctl.changed(before, thumbnail(latest)):
                                    notes.append("The screen changed mid-batch, so the rest was skipped. "
                                                 "Reground on this screen.")
                                    break
                            # Keys sent next go where the click put the focus, whatever the screen shows: the
                            # page-load wait after a click on YouTube's search box held the typing ~0.8 s (v5-t1),
                            # and waiting for B1's click or "hello" to show held each next key ~0.3 s (v12-t3), YouTube's
                            # suggestions under "horses" Enter ~0.5 s (v13-t1). The Pico runs each command before the
                            # next. A click may open what they go to (a menu), and a key too (a launcher).
                            keys_next = i + 1 < len(plan.actions) and plan.actions[i + 1].type in ("type_text", "hotkey")
                            # A scroll goes where the pointer is, whatever the screen shows: waiting for the hover
                            # to show held it ~0.5 s (v20-t1).
                            scroll_next = (action.type, action.name) == ("skill", "hover") \
                                and i + 1 < len(plan.actions) and plan.actions[i + 1].type == "scroll"
                            settles = not (keys_next and action.type == "type_text") and not scroll_next
                            stamp = time.monotonic()
                            # The screen this input is judged against; the plan's own was grabbed just now.
                            base = sent_to if i == 0 and sent_to is not frame else ctl.capture.grab() if settles else None
                            try:
                                ctl.execute(action, profile)
                                acted = True
                                # shortcut: a menu that opens on hover closes once the pointer leaves it; keep the
                                # pointer there if clicks into hover menus need it.
                                # A scroll next goes where the pointer is. One done leaves it too: the thumbnail
                                # it ended on played its preview on YouTube (2026-10-19).
                                if profile.input == "mouse" and (action.type in POINTER_ACTIONS | {"scroll"} or (
                                        action.type, action.name) == ("skill", "select")) \
                                        and [a.type for a in plan.actions[i + 1:i + 2]] != ["scroll"]:
                                    ctl.park()
                                # A wait sends no input, so a still screen after it is no failure.
                                sent_input = sent_input or (action.type, action.name) != ("skill", "wait")
                                self._emit_action(step, action, ok=True)
                            except ActionError as exc:
                                self._emit_action(step, action, ok=False)
                                notes.append(f"Rejected action: {exc}. Try another action.")
                                action_failed = True
                                break
                            finally:
                                times["t_execute"] += time.monotonic() - stamp
                            history.append(action.brief())  # before settling: a restart lands there
                            if settles:
                                settled_s = ctl.wait_settled(loading=_may_load(action) and not keys_next,
                                                             react=not keys_next or action.type == "hotkey",
                                                             since=None if base is None else thumbnail(base))
                                times["t_settle"] += settled_s
                    if pending_done is not None:
                        # Check the claim on the new screen now: replanning on a screen that looks unchanged
                        # (a camera shutter) repeated the action (measured: 20+ photos for one).
                        check = getattr(self.executor, "missing", None)
                        latest = ctl.capture.grab()
                        if callable(check) and latest is not None and not is_blank(latest):
                            stamp = time.monotonic()
                            shown = prepare(latest, profile.rect, grid=getattr(self.executor, "image_grid", True))
                            missing = check(shown, ctx, pending_done.summary)
                            times["t_model"] += time.monotonic() - stamp
                            if missing:
                                notes.append(f"Not confirmed done: {missing}. Finish it; never repeat a send, "
                                             "purchase, deletion or photo.")
                            else:
                                terminal = pending_done
                        else:
                            notes.append("Verify the result on a fresh screen before returning done.")
                    if terminal is not None:
                        status = terminal.type
                        message = terminal.summary or terminal.question or terminal.reason or status
                        break
                    acknowledge = getattr(self.executor, "acknowledge", None)
                    if callable(acknowledge):
                        acknowledge(bool(plan.actions) and not action_failed)
                    if sent_input:
                        after = ctl.capture.grab()
                        if after is None or is_blank(after):
                            token.cancel("video lost")
                            token.check()
                        unchanged = not visibly_changed(sent_to, after)
                        unchanged_steps = unchanged_steps + 1 if unchanged else 0
                        if not unchanged and not action_failed:
                            progress_at = time.monotonic()
                        if unchanged:
                            notes.append("Screen unchanged; try something different.")
                        # A covered result can leave pixels unchanged. The Jev
                        # executor checks its semantic postcondition next turn,
                        # so one still screen is no failure; a second in a row is.
                        semantic = getattr(self.executor, "verifies_stages", False) is True
                        action_failed = action_failed or unchanged_steps > (1 if semantic else 0)
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
                    # The same batch from the same screen a third time is stuck, even when each try flashed
                    # something the check above saw as progress (measured: open_app looping from one page).
                    tried = attempts.setdefault(tuple(a.brief() for a in plan.actions), [])
                    tried.append(before)
                    if sent_input and sum(frame_change(t, before) < ctl.settle.threshold for t in tried) >= 3:
                        status, message = "fail", "Stuck repeating the same action"
                        break
                    note = "\n".join(notes)
                except Cancelled as exc:
                    if self.reconnect is None or exc.reason not in RESUMABLE:
                        raise
                    self.on_status(f"{exc.reason.capitalize()}; waiting for it to come back…")
                    if found := self.reconnect(exc.reason):
                        profile = found
                    settled_s = None
                    note = "The target restarted, reconnected or blanked; continue from this screen."
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
                   total_s=round(time.monotonic() - started, 3), trace=self._trace())
        return TaskResult(status, message, step, run_id)

    # -- display events (no-ops without a display) -------------------------

    def _emit(self, ev: str, **fields: object) -> None:
        if self.events is not None:
            self.events.publish(ev, **fields)

    def _emit_action(self, step: int, action: Action, *, ok: bool) -> None:
        point = [action.x, action.y] if action.type in POINTER_ACTIONS else None
        self._emit("action", step=step, action=action.brief(), point=point, ok=ok)

    def _show(self, frame: Frame, profile: DeviceProfile, model_jpeg: bytes) -> int:
        """Share and keep the step's screen; gives its frame id (0 without a display).
        The model copy is reused unless it has an aiming grid."""
        if self.events is None:
            return 0
        grid = getattr(self.executor, "image_grid", True)
        return self.events.set_frame(prepare(frame, profile.rect, grid=False) if grid else model_jpeg, keep=True)

    def _trace(self) -> list[object]:
        """The executor's decisions since the last event (Jev calls with their boxes, plans, routing)."""
        trace = getattr(getattr(self.executor, "jev", None), "trace", None)
        taken = []
        while isinstance(trace, deque) and trace:  # the race's planner thread may still append
            taken.append(trace.popleft())
        return taken


def _may_load(action: Action) -> bool:
    """A click or Enter can start a page load, which needs a longer stillness to count as settled. So does an
    app launch: Chrome's window showed before its new tab page drew, and a plan on it was discarded (2026-10-20)."""
    if action.type in ("click", "tap", "double_click") or (action.type, action.name) == ("skill", "open_app"):
        return True
    return action.type == "hotkey" and [k.upper() for k in action.keys] in (["ENTER"], ["RETURN"])
