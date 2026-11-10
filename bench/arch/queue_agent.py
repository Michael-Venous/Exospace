"""Candidate architecture (study 16): a fast LLM decides the next actions for one screen at a time, Jev clicks.

  Brain     one fast vision model. Each call sees the user's words so far, the conversation, the log of what ran
            and the screen as it is now; it checks that its last actions worked and replies with the next ones:
            as many as need no new look at the screen, or none to see the screen again. New speech cancels the
            call in flight (latest wins). The next call starts only once every action of this one has run.
  Executor  runs them in order: keys and typing go straight to the HID; a click names its target and Jev places
            it on the screen as it is now (locate + zoom). A target Jev can't find, or any error, is logged and
            drops the rest of the reply. `done` counts only if Jev's completion check (goal + steps + screen)
            confirms it; `done`, `say` and `ask` end the request, and only new speech starts the next one.
Strictly sequential: no call overlaps the actions of the one before. Pixels in, keyboard and mouse out. Bench only.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import httpx

from bench.arch.pointer import Pointer
from bench.arch.openrouter_study import ModelHTTPError
from brain.agent.jev import JevVision, probability
from brain.agent.jev_flash import COMPLETION_GATE
from brain.agent.triage import GATE, refine
from brain.cancel import CancelToken
from brain.controller import Controller
from brain.hid import HID_MAX
from brain.imaging import shrink
from brain.types import Action, DeviceProfile, Rect
from brain.voice import profiles

OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
DEVICE = os.environ.get("EXO_DEVICE", "Linux laptop, KDE Plasma desktop. The GUI key opens the application "
                                      "launcher; GUI+PAGEUP maximizes the window in front. The panel hides itself: "
                                      "GUI also shows it, with its clock.")  # the target as the model is told it
MODIFIERS = {"CTRL", "SHIFT", "ALT"}  # pressed alone they do nothing (GUI alone opens the launcher)
SEE_SIDE = int(os.environ.get("EXO_SEE_SIDE", "1280"))  # the longest side of the screenshot the brain sees
# CTRL+L then typing 0.06 s later lost the first letters ("youtube.com" became "tube.com") while Chrome focused its
# address bar: no input follows another sooner than this (input pacing, not a wait for the screen).
INPUT_GAP_S = float(os.environ.get("EXO_INPUT_GAP_S", "0.08"))
# A still screen loads nothing more: an until it doesn't show yet gives up after UNTIL_STILL_S. Over 4 rounds 48
# untils waited the full 4 s for what never came (often a guess, "the picture taken"); every one met came by 3.4 s.
UNTIL_STILL_S = 2.0
# Any spot of a 160x100 thumbnail changing this much is movement: a loading page's tab spinner counts (most slow
# untils were pages after ENTER, met by 3.4 s), and so does a blinking caret, which only keeps the full wait.
STILL_DIFF = 12
UNTIL_S = 4.0  # longest an "until" waits for what the model expects (pages appeared within 2.7 s on the laptop)
SLEEP_S = 8.0  # no speech and no action for this long: the model stops being called
HEDGE_S = 1.2  # no answer yet from the first host: the same request goes to the next one too (p90 first token 0.8 s)
FIRST_TOKEN_S = 4.0  # a reply that brings no new text for this long is asked again, on the next host. First tokens
# over 465 calls: p50 0.49 s, p95 1.3 s, p99 2.3 s; a 2.5 s limit cut 2% of calls short (one stall was 7.8 s).
READ_S = FIRST_TOKEN_S  # the connection's own read timeout: a host that sends nothing at all
RUNAWAY = 8  # the same action this many times in a row ends a reply
LOOPS = 3  # the same reply this many times in one request: it is going in circles, and stops
DONE_TAIL = 300  # actions of this request each call sees, compactly
FAILED = {"not_shown", "not_found", "error", "not_verified"}
PRIMARY_ACTIONS = {"key", "type", "click", "double_click", "triple_click", "right_click", "drag", "scroll", "hover",
                   "wait",
                   "done", "until", "say", "ask"}
STEP_BACK_EFFORT = "low"  # medium took 3.9 s, or ran out of tokens reasoning (11.6 s)
STEP_BACK = ("\n\nYou are stuck. Step back before you act: say in one sentence what is actually on the screen and why "
             "your last steps didn't work, then reply with actions for a different approach, or ask the user one "
             "specific question if you see no way.")
REPEATING = ("You sent this same reply twice in this request, and the screen still doesn't show what you meant: "
             "don't send it again. Look closer (make the window bigger, zoom in) or check another way, or try a "
             "different approach.")
LOG_TAIL = 20  # executor log entries each call sees
HISTORY = 4  # earlier calls of the request each call sees as chat turns
# Each model on one host. Qwen's quality depends on it (study 17: 192/192 pointed on ModelRun, 15/192 on Cerebras);
# DeepSeek was fastest on Together (model check: 0.33 s first token p50), Morph next (0.58 s): its fallback.
PINNED = {"qwen/qwen3.8-27b": ("modelrun/fp4",), "deepseek/deepseek-v4.1-flash-1017": ("together", "morph/fp8"),
          "z-ai/glm-5.3-flash": ("together",)}
# Pointing models that can click in Jev's place, each on the host study 17 measured it on.
# (model, host, reasoning effort: "" = off, x/y in pixels). GLM 5.3 Flash's hosts all require some reasoning, and
# it answers in pixels.
CLICKERS = {"qwen": ("qwen/qwen3.8-27b", PINNED["qwen/qwen3.8-27b"][0], "", False),
            "deepseek": ("deepseek/deepseek-v4.1-flash-1017", "inference-net/fp8", "", False),
            "glm": ("z-ai/glm-5.3-flash", PINNED["z-ai/glm-5.3-flash"][0], "minimal", True)}
SURE_BOX = 0.7  # Jev's box confidence at which a target its presence check refused still gets a zoomed look
PROMPT = """You operate a computer for a user who speaks to you. You see its screen; you act only through
its keyboard and mouse. The user's words arrive while they speak.

Each time you are called you get the screen as it is now and the log of the actions you took and their
results. First check on the screen whether your last actions did what you meant. Then reply with the next
actions, ONE JSON object per line, nothing else. Every call costs the user a second, so chain all the actions
you can take without seeing the screen again: launching an app is one reply (GUI, its name, ENTER); so is going
to an address or searching (CTRL+L or a click on a visible search box, the words, ENTER); so are shortcuts in a
row. End the reply at the first action whose result you must see first: a click on something not on the screen
yet, or a choice that depends on what loads. If the screen is still loading or changing, reply with nothing and
you will see it again.
  {"key": ["CTRL","L"]}             a key or combination (CTRL SHIFT ALT GUI ENTER ESC TAB BACKSPACE DELETE UP
                                    DOWN LEFT RIGHT HOME END PAGEUP PAGEDOWN F1-F12, letters, digits)
  {"type": "lofi hip hop"}          type these words into whatever has focus
  {"click": "Reply"}                click a visible control, named by its exact visible text when it has any
                                    ("Reply", "119.93 Hz"); an icon by its look and place ("the gear icon at the
                                    top right"); when the text shows more than once, say which ("the Home link
                                    in the first paragraph") or name it by what only it shows (an email address, a
                                    date); also "double_click", "right_click"; add "hold": ["CTRL"] to
                                    click while holding keys (CTRL or SHIFT to select several files or items)
  {"drag": ["the file report.pdf", "the Documents folder in the sidebar"]}   press on one spot, release on another
  {"scroll": 3, "at": "the list of results"}   mouse-wheel notches, positive down ("at" optional)
  {"until": "the Bluetooth settings page"}   wait until the screen shows it, when a page, an app or a dialog
                                    has to load. The actions after it in the same reply run as soon as it shows, so
                                    keep going when you know the next steps (launching an app is one reply: GUI,
                                    its name, ENTER, until its window, then maximize it). End the reply at an
                                    until only when the next step depends on what will show. Typed text and
                                    closing something are instant: no until.
  {"done": "YouTube's home page is open in Firefox"}   the request is done: say what the screen shows that
                                    proves it; it is checked on the screen before it counts
Name what until expects in a few words you are sure of: a window or page title, a heading ("the Bluetooth page"),
never details you guess (a checkbox, a thumbnail), and only what isn't on the screen yet: waiting for what already
shows passes at once and proves nothing. Describe both until and done as they look ("only the
wallpaper, no windows"), not by a name ("the desktop"): a fast vision check compares them with the screen.
  {"say": "It's running at 120 Hz."} / {"ask": "Which wallpaper would you like?"}   talk to the user
  {"resume": true}                  when the user spoke in the middle of a task ("interrupted": that request and
                                    its steps left), your new actions run first; end with resume to carry on with
                                    the steps left (they added something, like "and make it fullscreen"), or leave
                                    it out to drop them (they changed their mind, or want it to stop)
A confirmed done, or a reply that ends with say or ask, ends your turn: the user speaks next.
Rules:
- Judge what the user means from the screen and the conversation before acting.
- While you work, reply with actions only: never announce what you are about to do. Talk when you have the answer
  or the result, or to ask.
- If the log shows an action failed, or the screen shows it changed nothing, don't repeat it: try another way.
  Some actions leave no mark on the screen (taking a photo, copying, saving again): before repeating one, check
  where its result would be (the folder, the file), or the user ends up with two.
- When steps keep not showing their result, the screen may be too small to show it: make the window bigger or zoom
  in to check before doing them again.
- Maximize an app you open, unless its window already fills the screen (maximizing again restores it). Use the
  device's shortcut for it, not a title-bar button: title-bar buttons differ between desktops. A window that is
  still opening can miss it: if it doesn't fill the screen afterwards, press it again. End the reply after
  maximizing: an app can open on a welcome page or a dialog, and keys typed before you see it go astray.
- A list or page can go on past the window's edge (a scroll bar shows it): see all of it (maximize, scroll) before
  you pick the newest, last or biggest item or say what it holds. Scroll a target cut off at an edge fully into
  view before clicking it: its label alone is easily taken for the item next to it.
- A click that changed nothing probably missed: its log entry says where it landed, in the screenshot's pixels.
  If that isn't on the target, name it another way (by other text it shows, or what is next to it), use the
  keyboard, or maximize the window, before asking the user.
- Pages, apps and dialogs take a moment to appear: end such a reply with until. If you only just asked for one
  (seconds_since_last_action) and the screen doesn't show it yet, reply with nothing: never ask for it again while
  it may still be coming.
- State only what the screen shows; never guess a result. A screenshot is still: it can't show that something plays
  or moves, so don't wait for that. A video you open plays by itself (often an ad first, which
  ends by itself or offers Skip): don't press play or pause on it, since pressing stops it. Press play only when
  the screen shows it stopped: a big play button over the picture, or an error.
- The user hears say and ask, never done: when they asked to be told something, say it before done.
- Keys go where the focus is: after a click elsewhere (a sidebar, a list), click the text field before you
  select or type in it.
- Use keyboard shortcuts where they are sure.
- Type only file names and paths you read on the screen or know for sure (~ is the user's home folder): a guessed
  path or a misread name goes nowhere.
- Words typed in a browser's address bar search the web, not the site that is open: to search a site, use its
  own search box."""


# Planning ahead (`plan="ahead"`): steps on screens not drawn yet are queued too, and a click waits for its target
# to appear, so each step waits on the computer, not on the model.
APPEAR_S = 5.0  # how long a click waits for its target to appear before it fails
AHEAD = """End the reply at the first action whose result you must see first: a click on something not on the screen
yet, or a choice that depends on what loads."""
assert AHEAD in PROMPT
PROMPT_AHEAD = PROMPT.replace(AHEAD, """Plan ahead: also queue the steps on screens that will only appear after earlier ones
run; a click waits for its target to appear. End the reply at the first step that depends on something you can't
foresee (which result to pick, what a page says): you are called again when the queue runs out or a step fails.""")


@dataclass
class Record:
    t: float
    action: dict
    result: str
    detail: str = ""


@dataclass
class Shared:
    queue: deque = field(default_factory=deque)
    log: list[Record] = field(default_factory=list)
    state: str = "idle"  # running | idle
    goal: str = ""  # what the user has asked for, for the done check
    turn_start: int = 0  # the log entry the current request starts at
    finished: bool = False  # a confirmed done, say or ask ended the turn: only new speech wakes the model
    request: int = 0  # counts requests: a wait from an earlier one gives up when a new one starts
    held: list = field(default_factory=list)  # the steps an interrupted request still had queued
    held_goal: str = ""  # that request's words
    activity_at: float = 0.0  # last completed executor step, including a wait or failed check


class Executor(threading.Thread):
    def __init__(self, controller: Controller, jev: JevVision, profile: DeviceProfile, shared: Shared,
                 cond: threading.Condition, started: float, trace: list) -> None:
        super().__init__(daemon=True, name="queue-executor")
        self.controller, self.jev, self.profile = controller, jev, profile
        self.shared, self.cond, self.started, self.trace = shared, cond, started, trace
        self.closed = False
        self.inputs: list[float] = []  # time of every input sent, for first-input latency
        self.pointer: Pointer | None = None  # a pointing model that clicks in Jev's place, when one is given
        self.appear_s = 0.0  # how long a click waits for its target to appear (0: it must be there already)
        self.checks = jev
        self.strict_reports = False  # benchmark candidate: reports follow actions and the same completion gate
        self.report = lambda text: None
        self.active_request: int | None = None

    def run(self) -> None:
        while not self.closed:
            with self.cond:
                while not self.closed and not self.shared.queue:
                    self.cond.wait(0.5)
                if self.closed:
                    return
                action = self.shared.queue.popleft()
                request = self.active_request = self.shared.request
                self.shared.state = "running"
            result, detail = self._do(action)
            with self.cond:
                current = request == self.shared.request
                if self.strict_reports and not current:
                    result, detail = "superseded", "a newer request owns the queue"
                if current:
                    self.shared.activity_at = time.monotonic()
                self.shared.log.append(Record(round(time.monotonic() - self.started, 2), action, result, detail))
                if result in ("verified", "asked", "uncertain"):
                    self.shared.finished = True
                    self.shared.held, self.shared.held_goal = [], ""  # the request is done: nothing to resume
                    if self.strict_reports:
                        self.shared.queue.clear()
                if result not in ("done", "verified") and (current or not self.strict_reports):
                    self.shared.queue.clear()  # the rest of the reply assumed this worked
                if not self.shared.queue:
                    self.shared.state = "idle"
                self.cond.notify_all()

    def _do(self, action: dict) -> tuple[str, str]:
        try:
            request = self.shared.request
            kinds = [key for key in action if key in PRIMARY_ACTIONS]
            if self.strict_reports and len(kinds) != 1:
                return "error", "one primary action is required per object; use separate JSON lines"
            kind = kinds[0] if self.strict_reports else next(iter(action))
            value = action[kind]
            if kind == "key":
                keys = value if isinstance(value, list) else [value]
                if {str(k).upper() for k in keys} <= MODIFIERS:
                    return "error", "a modifier pressed alone does nothing: to hold it, click with \"hold\""
                self._input(lambda: self.controller.hotkey(keys))
            elif kind == "type":
                if action.get("into"):
                    point = self._find(str(action["into"]))
                    if point is None:
                        return "not_found", "the destination text field was not found; no text was typed"
                    self._input(lambda: self.controller.execute(Action("click", x=point[0], y=point[1]), self.profile))
                if action.get("replace"):  # also without "into": after a separate click, old text must not stay
                    self._input(lambda: self.controller.hotkey(["CTRL", "A"]))
                self._input(lambda: self.controller.type_text(str(value)))
            elif kind in ("click", "double_click", "right_click"):
                point = self._find(str(value))
                if point is None:
                    return "not_found", f"{value!r} wasn't found on the screen"
                hold = [str(k).upper() for k in action.get("hold") or []]
                if hold:  # the bench HID's click takes the keys to hold (not yet in PLAN.md §5.1's UART contract)
                    if not set(hold) <= MODIFIERS:
                        return "error", f"only CTRL, SHIFT or ALT can be held, not {hold}"
                    x, y = round(point[0] * HID_MAX), round(point[1] * HID_MAX)
                    self._input(lambda: self.controller.hid.send(
                        {"op": "click", "x": x, "y": y, "button": "right" if kind == "right_click" else "left",
                         "count": 2 if kind == "double_click" else 1, "hold": hold}))
                else:
                    self._input(lambda: self.controller.execute(Action(kind, x=point[0], y=point[1]), self.profile))
                return "done", self._where(point)
            elif kind == "drag":
                a, b = self._find(str(value[0])), self._find(str(value[1]))
                if a is None or b is None:
                    return "not_found", f"{value[0] if a is None else value[1]!r} wasn't found on the screen"
                self._input(lambda: self.controller.execute(Action("swipe", x=a[0], y=a[1], x2=b[0], y2=b[1], ms=600),
                                                            self.profile))
            elif kind == "scroll":
                if action.get("at"):
                    point = self._find(str(action["at"]))
                    if point is not None:
                        self.controller.hover(*point)
                self._input(lambda: self.controller.execute(Action("scroll", dy=int(value)), self.profile))
            elif kind == "done":
                return self._verify(str(value))
            elif kind == "until":
                return self._until(str(value))
            elif kind in ("say", "ask") and self.strict_reports:
                if kind == "say":
                    result, detail = self._verify(str(value))
                    if result != "verified":
                        return result, detail
                with self.cond:
                    if request != self.shared.request or self.closed:
                        return "superseded", "a newer request replaced this report"
                    self.report(str(value))
                return ("verified", "reported after earlier actions completed") if kind == "say" else ("asked", "asked user")
            else:
                return "error", f"unknown action {kind!r}"
            return "done", ""
        except Exception as exc:  # noqa: BLE001 - a failed action is logged for the model; the executor carries on
            return self._failure(exc)

    def _failure(self, exc: Exception) -> tuple[str, str]:
        return "error", f"{type(exc).__name__}: {exc}"[:160]

    def _where(self, point: tuple[float, float]) -> str:
        """Where a click landed, in the pixels of the screenshot the model sees: a pointer that put "Never" on its
        dropdown's box (16 px too high) was clicked three times, the model not knowing where it went (2026-10-22)."""
        r = self.profile.rect
        scale = SEE_SIDE / max(r.x1 - r.x0, r.y1 - r.y0)
        return f"at ({round(point[0] * (r.x1 - r.x0) * scale)}, {round(point[1] * (r.y1 - r.y0) * scale)})"

    def _jpeg(self) -> bytes:
        """The screen as it is now: no waiting for it to settle."""
        return cv2.imencode(".jpg", self.controller.capture.grab(), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    def _input(self, send) -> None:
        if self.inputs:  # input at a person's pace: a shortcut's focus change lands before the next key
            time.sleep(max(0.0, self.inputs[-1] + INPUT_GAP_S - time.monotonic()))
        with self.cond:
            if self.closed or (self.strict_reports and self.active_request is not None and
                               self.active_request != self.shared.request):
                raise RuntimeError("stopped or superseded: no input sent")
            self.inputs.append(time.monotonic())
            send()

    def _until(self, expected: str) -> tuple[str, str]:
        """Wait on the computer, not the model: Jev looks at each fresh screen until it shows what the model expects
        (a page, an app, a dialog), up to UNTIL_S."""
        started, request = time.monotonic(), self.shared.request
        # Plain: telling Jev a name in a search field or on a button doesn't count (after "until Kamoso" passed on the
        # launcher's row) doubled the untils it missed, 14% to 24-40% over 5 rounds ("Save" dialogs went unseen).
        ask = f"Does `screen` show this: {expected}?"
        before, moved = None, started  # the last poll's thumbnail, and when the screen last changed
        while not self.closed and self.shared.request == request:
            frame = self.controller.capture.grab()
            thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 100), interpolation=cv2.INTER_AREA)
            if before is None or cv2.absdiff(thumb, before).max() > STILL_DIFF:
                moved = time.monotonic()
            before = thumb
            jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
            answers, _, _ = self._vision_call(lambda: self.checks.evaluate(jpeg, {"shown": {
                "type": "noul", "instructions": ask}}, label="until"))
            waited = round(time.monotonic() - started, 1)
            if probability(answers["shown"].get("noul")) >= GATE:
                return "done", f"shown after {waited} s"
            if waited >= UNTIL_STILL_S and time.monotonic() - moved >= UNTIL_STILL_S - 0.5:
                return "not_shown", (f"not shown after {waited} s, and the screen has stopped changing: look at it. "
                                     "If the action had its effect, go on; if nothing changed, try another way")
            if waited >= UNTIL_S:
                return "not_shown", (f"not shown after {waited} s: look at the screen. If the action had its effect, "
                                     "go on; if nothing changed, it didn't work: try another way rather than wait")
        return "error", "stopped: a new request started"

    def _verify(self, claim: str) -> tuple[str, str]:
        """The request counts as done only if Jev's completion check sees it done on the screen."""
        with self.cond:
            goal = self.shared.goal
            request = self.shared.request
            # what ran, and what each until saw appear: closing an app leaves nothing on screen of the earlier steps
            steps = [json.dumps(r.action) + (f" ({r.detail})" if r.detail else "")
                     for r in self.shared.log[self.shared.turn_start:] if r.result == "done"]
        score = self._vision_call(lambda: self.checks.completion(self._jpeg(), goal, steps, evidence=claim))
        if score >= COMPLETION_GATE:
            return "verified", f"Completion check supports the request ({score:.2f})"
        reason = getattr(self.checks, "last_reason", "")
        if self.strict_reports and getattr(self.checks, "last_status", "") == "unknown":
            with self.cond:
                if self.closed or request != self.shared.request:
                    return "superseded", "a newer request replaced this uncertainty report"
                self.report(f"I can't fully verify the result from the screen. {reason}")
            return "uncertain", reason  # missing proof is not evidence that an effect should be repeated
        return "not_verified", (f"Completion not supported on the screen ({score:.2f}). {reason} Find what isn't "
                                "done yet. A still screen can't prove some things (that a video plays, a sound): if "
                                "the screen shows all it can, tell the user what you see with say instead of done")

    def _vision_call(self, ask):
        # The alternate checker owns its response budget and transport retry. Retrying a total
        # deadline here would double that budget and exceed the stall detector's limit.
        return _again(ask) if self.checks is self.jev else ask()

    def _find(self, target: str) -> tuple[float, float] | None:
        """The target's place, looking again on fresh screens until it appears or `appear_s` runs out."""
        deadline = time.monotonic() + self.appear_s
        while (point := self._place(target)) is None and time.monotonic() < deadline and not self.closed:
            pass
        return point

    def _place(self, target: str) -> tuple[float, float] | None:
        """The pointing model, or Jev zoomed in once and again, places the target on the screen as it is now."""
        jpeg = self._jpeg()
        started = time.monotonic()
        if self.pointer is not None:
            point = self.pointer.point(jpeg, target)
            self.trace.append({"label": "ground", "target": target, "point": point, "by": "pointer",
                               "elapsed_s": round(time.monotonic() - started, 3)})
            return point
        point = self.jev.locate(jpeg, target)
        if point is None:
            # Jev's presence answer turns on the wording ("119.93 Hz option in the open refresh rate dropdown" 0.08,
            # "119.93 Hz" 0.79, the same box both times): a box Jev is sure of, or a target it does see, gets the
            # zoomed look, which checks again.
            answers = (self.jev.trace[-1].get("answers") or {}) if self.jev.trace else {}
            answer = answers.get("location") or {}
            box = answer.get("box")
            seen = min(answers.get("present", {}).get("noul") or 0, answers.get("unique", {}).get("noul") or 0)
            if box and ((answer.get("confidence") or 0) >= SURE_BOX or seen >= GATE):
                point = refine(self.jev, jpeg, target, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
        else:
            point = refine(self.jev, jpeg, target, point) or point
        self.trace.append({"label": "ground", "target": target, "point": point, "by": "jev",
                           "elapsed_s": round(time.monotonic() - started, 3)})
        return point


def _again(ask):
    """A Jev check, asked once more if its connection drops (a ReadError once failed a done that was right)."""
    try:
        return ask()
    except Exception:  # noqa: BLE001 - one retry; a second failure is the step's error
        return ask()


def _brief(action: dict) -> str:
    """An action in a few characters: "key CTRL+L", "type 'Add Cone'", "click Save"."""
    kind = next((key for key in action if key in PRIMARY_ACTIONS), next(iter(action)))
    value = action[kind]
    return f"{kind} {'+'.join(value) if isinstance(value, list) else repr(value) if kind == 'type' else value}"


class Brain(threading.Thread):
    """The model loop: one call, its actions run, then the next call."""

    def __init__(self, target, shared: Shared, cond: threading.Condition, executor: Executor, model: str,
                 effort: str, started: float) -> None:
        super().__init__(daemon=True, name="queue-brain")
        self.target, self.shared, self.cond, self.executor = target, shared, cond, executor
        self.model, self.effort, self.started = model, effort, started
        self.client = self._client()
        self.phase = ("waiting", 0.0)  # what the loop is doing, and since when (for stall reports)
        self.conversation: list[dict] = []  # earlier turns and what was said to the user
        self.words = ""  # the turn being spoken
        self.speaking = False
        self.active_until = 0.0  # a request is active until then; afterwards only speech wakes the model
        self.dirty = False
        self.calling = False  # a model call is in flight
        self.cancel_call = threading.Event()
        self.closed = False
        self.calls: list[dict] = []
        self.said: list[str] = []
        self.shots: Path | None = None  # a folder to keep each call's screenshot in
        self.prompt = PROMPT
        self.see_side = SEE_SIDE
        self.stepped_back = -1  # the log length at the last stepping-back call: once per stuck spell
        self.replies: dict[str, int] = {}  # this request's replies and how often each came (a loop guard)
        self.repeated = False  # the last reply was one sent before
        self.hosts = list(PINNED.get(model, ()))  # the providers in the order tried; empty: the fastest serves
        self.history: list[dict] = []  # this request's earlier calls: what each saw and replied
        self.observe = None  # optional bench checker: chronological pixels, never oracle state
        self.frame_observer = None  # optional instrumentation/observation of this exact planner frame
        self.loop_handler = None  # optional bench policy owns terminal cancellation and reporting
        self.note = ""  # optional bench instruction appended to this call's context

    @staticmethod
    def _client() -> httpx.Client:
        # One connection per call: one reused after a cancelled stream failed with "bad record mac". A reply that
        # stalls READ_S (a provider's first token once took 8.4 s against a usual 0.45 s) fails and is asked again.
        return httpx.Client(timeout=httpx.Timeout(60, read=READ_S), limits=httpx.Limits(max_keepalive_connections=0),
                            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})

    # -- speech, from the scripted user -------------------------------------------------------------------

    def hear(self, words: str, speaking: bool) -> None:
        with self.cond:
            if not self.words:  # a new request starts here: what the last one left queued is held, not run
                self.replies, self.repeated = {}, False
                busy = not self.shared.finished and (self.shared.queue or self.shared.state == "running")
                last = next((t["user"] for t in reversed(self.conversation) if "user" in t), "")
                self.shared.held, self.shared.held_goal = (list(self.shared.queue), last) if busy else ([], "")
                self.shared.turn_start = len(self.shared.log)
                self.shared.queue.clear()
                self.shared.request += 1
                self.history = []
            self.words, self.speaking, self.shared.finished = words, speaking, False
            self.shared.goal = words  # this request alone: the done check judges what was just asked
            self.active_until = time.monotonic() + SLEEP_S
            self.dirty = True
            self.cancel_call.set()  # latest wins
            self.cond.notify_all()

    def end_turn(self) -> None:
        with self.cond:
            if self.words:
                self.conversation.append({"user": self.words})
            self.words, self.speaking = "", False

    def cancel_turn(self) -> None:
        """Drop the request being spoken (a command for the box, private mode, nothing heard): its words were
        streamed while the user talked, so the call in flight is cut and nothing it queued runs."""
        with self.cond:
            self.words, self.speaking, self.dirty = "", False, False
            self.shared.queue.clear()
            self.shared.finished = True
            self.cancel_call.set()
            self.cond.notify_all()

    def busy(self) -> bool:
        """Still working: speech pending, a call running, or the executor with something to do."""
        with self.cond:
            return self.dirty or self.calling or self.shared.state == "running" or bool(self.shared.queue)

    # -- the loop -------------------------------------------------------------------------------------------

    def run(self) -> None:
        while not self.closed:
            self.phase = ("waiting", round(time.monotonic() - self.started, 2))
            with self.cond:
                while not self.closed and not self._due():
                    self.cond.wait(0.25)
                if self.closed:
                    return
                self.dirty = False
                self.calling = True
                self.cancel_call.clear()
            try:
                self._call()
            except Exception as exc:  # noqa: BLE001 - a dropped call or a failed screenshot: log it, call again
                self.calls.append({"t": round(time.monotonic() - self.started, 2), "speaking": self.speaking,
                                   "words": self.words, "lines": [], "cancelled": False,
                                   "error": f"{type(exc).__name__}: {exc}"[:200]})
                self.client.close()
                self.client = self._client()
                if isinstance(exc, (TimeoutError, httpx.TimeoutException)) and len(self.hosts) > 1:
                    self.hosts.append(self.hosts.pop(0))  # the host that stalled goes to the back: retry elsewhere
            finally:
                with self.cond:
                    self.calling = False
                    self.cond.notify_all()

    def _due(self) -> bool:
        if self.dirty:
            return True
        if self.shared.finished or self.shared.queue or self.shared.state == "running":
            return False
        until = self.active_until
        if self.executor.strict_reports:
            until = max(until, self.shared.activity_at + SLEEP_S)
        return time.monotonic() < until

    def _call(self) -> None:
        self.phase = ("grabbing", round(time.monotonic() - self.started, 2))
        frame = self.target.grab()
        if self.frame_observer is not None:
            self.frame_observer(frame)
        image = self._encode(frame)
        if self.observe is not None:
            self.observe(image, self.shared.request)
        if self.shots is not None:  # what each call saw, for debugging
            (self.shots / f"call-{len(self.calls)}.jpg").write_bytes(image)
        with self.cond:
            context = {
                "device": profiles.device(DEVICE),  # $EXO_DEVICE, else the active computer profile's text
                "user_is_speaking": self.speaking, "user_words_so_far" if self.speaking else "user_said": self.words,
                "conversation_before": self.conversation[-12:],
                "log": [{"t": r.t, "action": r.action, "result": r.result, "detail": r.detail}
                        for r in self.shared.log[-LOG_TAIL:]]}
            # Everything this request has run, compactly: the log's tail alone held one long reply, and a snowman
            # model forgot the nose and eyes it had made and made them again and again (2026-10-22).
            context["done_this_request"] = [_brief(r.action) for r in self.shared.log[self.shared.turn_start:]
                                            if r.result == "done"][-DONE_TAIL:]
            again = [json.loads(r) for r, n in self.replies.items() if n >= 2]
            if again:  # it is going round: say so, so it changes approach before the guard stops it
                context["repeating"] = {"replies_sent_twice": again[-2:], "note": REPEATING}
            if self.shared.held:  # the user spoke in the middle of a request: the model decides what becomes of it
                context["interrupted"] = {"request": self.shared.held_goal, "steps_left": self.shared.held[:20],
                                          "count": len(self.shared.held)}
            if self.shared.log:  # the screen is this old relative to the last action: loading takes a moment
                context["seconds_since_last_action"] = round(time.monotonic() - self.started - self.shared.log[-1].t, 1)
        # The request so far as a chat: each earlier call's situation and the model's own reply (text only), so it
        # remembers what it meant to do and already checked; the screenshot rides on this call alone.
        history = []
        for past in self.history[-HISTORY:]:
            history += [{"role": "user", "content": past["context"]},
                        {"role": "assistant", "content": past["reply"] or "(nothing)"}]
        # Stuck (a reply sent twice, or two of the last 4 results failed): one call reasons and steps back. At a loop
        # it found the image viewer the model kept clicking in (2/2, 2.2-2.7 s), where the usual call repeated its
        # click. Failures, not only exact repeats: "click Paris, until Paris" ran 3 times, the first in a longer reply.
        recent = [r.result for r in self.shared.log[self.shared.turn_start:][-4:]]
        stuck = self.repeated or sum(r in FAILED for r in recent) >= 2
        # Once per stuck spell: one breeze-light repeat once made every later call step back (7 calls).
        step_back = stuck and len(self.shared.log) - self.stepped_back >= len(recent)
        if step_back:
            self.stepped_back = len(self.shared.log)
        effort = STEP_BACK_EFFORT if step_back else self.effort
        provider = ({"order": self.hosts, "allow_fallbacks": False} if self.hosts else {"sort": "latency"})
        body = {"model": self.model, "stream": True, "temperature": 0, "provider": provider,
                "max_tokens": 2048,
                "reasoning": {"effort": effort} if effort else {"enabled": False},
                "messages": [{"role": "system", "content": self.prompt}, *history,
                             {"role": "user", "content": [
                                 {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"
                                                                     + base64.b64encode(image).decode()}},
                                 {"type": "text", "text": json.dumps(context) + (STEP_BACK if step_back else "")
                                  + self.note}]}]}
        started = time.monotonic()
        self.phase = ("streaming", round(started - self.started, 2))
        record = {"t": round(started - self.started, 2), "speaking": self.speaking, "words": self.words,
                  "lines": [], "cancelled": False, "step_back": step_back}
        self.calls.append(record)
        text, full = "", ""
        for piece in self._stream(body, record):
            if self.cancel_call.is_set():
                record["cancelled"] = True
                break
            if "ttft_s" not in record:
                record["ttft_s"] = round(time.monotonic() - started, 3)
            text, full = text + piece, full + piece
            while "\n" in text:
                done, text = text.split("\n", 1)
                self._apply(done, record, started)
        cut = self.cancel_call.is_set()
        # A reply its own plan ended (an until, a failed step) still counts: it is remembered and loop-checked.
        record["cancelled"] = cut and not self._batch_ended(record)
        if not cut:
            self._apply(text, record, started)  # the last line, which has no newline after it
            if (not self.executor.strict_reports and record["lines"] and
                    record["lines"][-1].keys() & {"say", "ask"}):
                with self.cond:  # a reply that ends by talking hands the turn back ("I'll open it" before acting doesn't)
                    self.shared.finished = True
        if not record["lines"]:
            record["raw"] = full[:600]  # a reply with no action in it: what it said instead
        if not record["cancelled"] and record["lines"] and not record["speaking"] and self._counts_as_repeat(record):
            # only replies to the finished request: while the user speaks, calls on growing words repeat the same
            # first steps by design (a profile switch was stopped as "in circles" after 2.6 s)
            reply = json.dumps([{k: v for k, v in line.items() if k != "at_s"} for line in record["lines"]])
            self.replies[reply] = self.replies.get(reply, 0) + 1
            self.repeated = self.replies[reply] > 1
            if self.replies[reply] >= LOOPS:  # the same reply again and again: stop and say so
                with self.cond:
                    if self.loop_handler is not None:
                        self.loop_handler()
                    else:
                        self.shared.queue.clear()
                        self.shared.finished = True
                        self.said.append("I'm going in circles, so I've stopped. Tell me what to change.")
                    self.cond.notify_all()
        if not record["cancelled"] and full.strip():  # a call that added nothing isn't worth remembering
            past = {k: v for k, v in context.items() if k != "conversation_before"}
            self.history.append({"context": json.dumps(past), "reply": full.strip()})
        record["total_s"] = round(time.monotonic() - started, 3)

    def _counts_as_repeat(self, record: dict) -> bool:
        """Whether this reply counts toward the same-reply loop guard (bench subclasses exempt some)."""
        return True

    def _batch_ended(self, record: dict) -> bool:
        """Whether the executor, not new speech, cut this reply short (bench subclasses track it)."""
        return False

    def _encode(self, frame) -> bytes:
        """The screenshot as the model gets it."""
        return shrink(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes(), self.see_side)

    def _stream(self, body: dict, record: dict):
        """The reply's text as it streams, hedged: if no host has started answering within HEDGE_S, the same
        request goes to the next host too, and the first to answer is used (the other is dropped). A reply that
        brings nothing new for FIRST_TOKEN_S fails, and the run loop asks again. One host's first token took
        3.4 s, another sent nothing for 4 s, while the other host answered in about 0.5 s (2026-10-22)."""
        events: queue.Queue = queue.Queue()
        stop = threading.Event()

        def attempt(k: int, hosts: list[str]) -> None:
            request = {**body, "provider": {"order": hosts, "allow_fallbacks": False}} if hosts else body
            try:
                with self.client.stream("POST", OPENROUTER, json=request) as response:
                    if response.status_code != 200:
                        events.put((k, "error", ModelHTTPError(response.status_code, response.read())))
                        return
                    for line in response.iter_lines():
                        if stop.is_set():
                            return
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        for choice in json.loads(line[6:]).get("choices", []):
                            delta = choice.get("delta") or {}
                            if delta.get("content"):
                                events.put((k, "text", delta["content"]))
                            elif delta.get("reasoning"):  # a stepping-back call reasons before it answers
                                events.put((k, "alive", ""))
                events.put((k, "end", ""))
            except Exception as exc:  # noqa: BLE001 - reported to the reader, which decides
                events.put((k, "error", exc))

        hosts = list(self.hosts)
        threading.Thread(target=attempt, args=(0, hosts), daemon=True).start()
        started = heard = time.monotonic()
        tries, failed, winner = 1, set(), None
        try:
            while True:
                if record.get('deadline_s') and time.monotonic() - started > record['deadline_s']:
                    raise TimeoutError(f"planner stream exceeded {record['deadline_s']:g} s total deadline")
                if winner is None and tries == 1 and len(hosts) > 1 and (failed or time.monotonic() - started > HEDGE_S):
                    threading.Thread(target=attempt, args=(1, hosts[1:] + hosts[:1]), daemon=True).start()
                    tries += 1
                    record["hedged"] = True
                if time.monotonic() - heard > FIRST_TOKEN_S:
                    raise TimeoutError(f"no new tokens for {FIRST_TOKEN_S} s")
                if self.cancel_call.is_set():
                    return
                try:
                    k, kind, value = events.get(timeout=0.1)
                except queue.Empty:
                    continue
                if winner is not None and k != winner:
                    continue
                if kind == "error":
                    if isinstance(value, ModelHTTPError) and value.permanent:
                        raise value
                    failed.add(k)
                    if k == winner or len(failed) == tries and (tries > 1 or len(hosts) < 2):
                        raise value if isinstance(value, Exception) else RuntimeError(value)
                    continue
                heard = time.monotonic()
                if kind == "end":
                    if winner is None:
                        winner = k  # a reply with no text at all is still a reply
                    return
                if kind == "text":
                    if winner is None:
                        winner, record["host"] = k, (hosts[k:] + hosts[:k] or ["fastest"])[0]
                    yield value
        finally:
            stop.set()

    def _apply(self, line: str, record: dict, started: float) -> None:
        line = line.strip().strip("`").strip()
        decoder, at = json.JSONDecoder(), 0
        while line.startswith("{", at):  # one object per line, but some replies put several on one
            try:
                action, at = decoder.raw_decode(line, at)
            except json.JSONDecodeError:
                record["lines"].append({"unparsed": line[at:at + 120]})
                return
            while at < len(line) and line[at] in " ,":
                at += 1
            if isinstance(action, dict) and action:
                self._take(action, record, started)

    def _runaway(self, action: dict, record: dict) -> bool:
        recent = [{k: v for k, v in line.items() if k != "at_s"} for line in record["lines"][-RUNAWAY:]]
        if len(recent) == RUNAWAY and all(line == action for line in recent):
            # The same action again and again in one reply is a stuck generation, not a plan (655 CTRL+Z once).
            record["runaway"] = True
            self.cancel_call.set()
            return True
        return False

    def _take(self, action: dict, record: dict, started: float) -> None:
        if self._runaway(action, record):
            return
        record["lines"].append({"at_s": round(time.monotonic() - started, 3), **action})
        with self.cond:
            if self.executor.strict_reports and self.speaking and action.keys() & {"done", "say", "ask"}:
                return  # the remaining words can change the request; report only after speech ends
            if "resume" in action:  # carry on with the interrupted request's steps, after the new ones
                if self.shared.held:
                    self.shared.queue.extend(self.shared.held)
                    self.shared.goal = f"{self.shared.held_goal}; and: {self.words}"
                    self.shared.held, self.shared.held_goal = [], ""
                self.cond.notify_all()
                return
            if ("say" in action or "ask" in action) and not self.executor.strict_reports:
                self.said.append(str(action.get("say") or action.get("ask")))
                self.conversation.append({"you": self.said[-1]})
            else:
                self.shared.queue.append(action)
                self.active_until = time.monotonic() + SLEEP_S
            self.cond.notify_all()


class QueueAgent:
    """The model and executor threads, taking turns, on one target."""

    def __init__(self, target, model: str, effort: str, jev_key: str, log_dir: Path, clicker: str = "qwen",
                 plan: str = "screen", *, executor_cls=Executor, brain_cls=Brain, before_start=None) -> None:
        """`plan` "screen": each reply covers one screen; "ahead": replies plan across screens, clicks wait."""
        self.cancel = CancelToken()
        self.trace: list[dict] = []
        self.started = time.monotonic()
        self.controller = Controller(target, target, self.cancel)
        self.jev = JevVision(jev_key, self.cancel, log_dir / "jev.jsonl", min_presence=GATE, min_confidence=GATE)
        self.jev.trace = self.trace  # every Jev call is kept, beside the grounding records
        frame = target.grab()
        h, w = frame.shape[:2]
        profile = DeviceProfile("linux", Rect(0, 0, w, h), "mouse", time.time())
        self.shared, self.cond = Shared(), threading.Condition()
        self.executor = executor_cls(self.controller, self.jev, profile, self.shared, self.cond, self.started, self.trace)
        if clicker in CLICKERS:
            self.executor.pointer = Pointer(self.cancel, self.jev, *CLICKERS[clicker])
        self.brain = brain_cls(target, self.shared, self.cond, self.executor, model, effort, self.started)
        if plan == "ahead":
            self.executor.appear_s, self.brain.prompt = APPEAR_S, PROMPT_AHEAD
        self.brain.shots = log_dir
        if before_start is not None:
            before_start(self)
        self.executor.start()
        self.brain.start()

    def close(self) -> None:
        self.brain.closed = self.executor.closed = True
        self.brain.cancel_call.set()
        with self.cond:
            self.cond.notify_all()
        self.brain.join(5)
        self.executor.join(5)
        self.jev.close()
        if self.executor.pointer is not None:
            self.executor.pointer.close()
        self.brain.client.close()
