"""Candidate architecture (study 16): a fast LLM at the front, a shared action queue, Jev as clicker.

  Brain     one fast vision model. Called on new speech (latest wins: an update cancels the call in flight)
            and whenever the executor stops (end of queue, a `look`, a failure) while a task is active; after
            SLEEP_S with no speech and nothing to do, only speech wakes it. Each call sees the user's words so
            far, the conversation, the executor's log and queue, and the screen as it is. It streams one action
            per line and each is applied as it arrives: appended to the queue, or `stop` (the executor stops
            after its current action and the queue is cleared, then the rest of the reply refills it).
  Executor  runs the queue in order: keys and typing go straight to the HID; a click names its target and Jev
            places it on the fresh screen (locate + zoom, gate 0.5). A target Jev can't find, or any error,
            pauses the queue and is logged; so does `look`. The brain's next call sees why and carries on.
Pixels in, keyboard and mouse out: nothing else of the target is read. Bench only.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import httpx

from brain.agent.jev import JevVision
from brain.agent.triage import GATE, refine
from brain.cancel import CancelToken
from brain.capture import thumbnail
from brain.controller import Controller
from brain.hid import HIDError
from brain.imaging import shrink
from brain.types import Action, ActionError, DeviceProfile

OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
DEVICE = ("Linux laptop, KDE Plasma desktop. The GUI key opens the application launcher: type to search, and look "
          "at the results before choosing one, as they mix apps, files and settings. Websites open from a browser's "
          "address bar (CTRL+L). The panel hides itself: an empty black screen is the desktop.")
# How the executor's state reads to the model: a bare "paused" read as waiting on the user ("the queued actions
# are paused, do you want me to continue?", study 16).
STATES = {"running": "running your actions", "paused": "stopped, waiting for your next actions",
          "idle": "finished all the actions you queued"}
SLEEP_S = 8.0  # no speech and nothing to do for this long: screen changes stop calling the model
LOG_TAIL = 20
MAX_LINES = 24  # a reply longer than this is degenerate (one repeated "type text" for 120 s, study 16)
CLICK_SETTLE_S = 1.2  # before Jev looks for a click's target
LOOK_SETTLE_S = 2.5  # at a look, for a page to finish drawing
SURE_BOX = 0.7  # Jev's box confidence at which a target its presence check refused gets a zoomed look anyway
CALL_SETTLE_S = 1.0  # a call right after the executor's input waits up to this for the screen to settle  # executor log entries each call sees
PROMPT = """You operate a computer for a user who speaks to you. You see its screen; you act only through
its keyboard and mouse. Never open a terminal or type shell commands unless the user explicitly asks for one:
do everything through the desktop's own interface, as the user would. The user's words arrive while they speak; you are called again as they grow, when the
screen changes, and whenever your queued actions run out, pause or fail.

Reply with actions, ONE JSON object per line, nothing else. They run in order as soon as each line arrives:
  {"key": ["CTRL","L"]}             a key or combination (CTRL SHIFT ALT GUI ENTER ESC TAB BACKSPACE DELETE UP
                                    DOWN LEFT RIGHT HOME END PAGEUP PAGEDOWN F1-F12, letters, digits)
  {"type": "lofi hip hop"}          type these words into whatever has focus
  {"click": "Reply"}                click a visible control, named by its exact visible text when it has any
                                    ("Reply", "119.93 Hz"); an icon by its look and place ("the gear icon at the
                                    top right"); also "double_click", "right_click"
  {"drag": ["the file report.pdf", "the Documents folder in the sidebar"]}   press on one spot, release on another
  {"scroll": 3, "at": "the list of results"}   mouse-wheel notches, positive down ("at" optional)
  {"wait": 1.5}                     seconds
                                    No key can be held during a click or drag: to act on several items, select
                                    them by keyboard (CTRL+A, SHIFT with arrow keys) or handle them one at a time.
  {"look": "the search results page"}   stop here and show me the screen again before going on (if the screen
                                    hasn't changed since your last call, looking again shows nothing new: act, or
                                    say what's wrong)
  {"stop": true}                    first, to cancel queued actions (e.g. the user changed their mind)
  {"say": "It's running at 60 Hz."} / {"ask": "Which wallpaper would you like?"}   talk to the user
Judge what the user means from the screen and the conversation before acting. Ask only when you can't make a
sensible choice; when one choice is clearly reasonable ("the house" means all of it), act and say what you
assumed. A screenshot can't show
motion: a video page showing its player is a video playing; don't keep clicking it to check. Use keyboard shortcuts where
they are sure; click when the target is visible. After anything that changes the screen in a way you need to
see (a page or app opening, a menu, a search), end with look. When the request is done, or you are waiting
for the user's answer, reply with nothing more to do: an empty reply or a say. Never repeat actions the log
shows done. The executor log tells you what ran, what failed and why."""


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
    state: str = "idle"  # running | paused | idle
    stop: bool = False


class Executor(threading.Thread):
    def __init__(self, controller: Controller, jev: JevVision, profile: DeviceProfile, shared: Shared,
                 cond: threading.Condition, started: float, trace: list) -> None:
        super().__init__(daemon=True, name="queue-executor")
        self.controller, self.jev, self.profile = controller, jev, profile
        self.shared, self.cond, self.started, self.trace = shared, cond, started, trace
        self.closed = False
        self.inputs: list[float] = []  # time of every input sent, for first-input latency

    def run(self) -> None:
        while not self.closed:
            with self.cond:
                while not self.closed and (not self.shared.queue or self.shared.state == "paused"):
                    self.cond.wait(0.5)
                if self.closed:
                    return
                action = self.shared.queue.popleft()
                self.shared.state = "running"
            result, detail = self._do(action)
            with self.cond:
                self.shared.log.append(Record(round(time.monotonic() - self.started, 2), action, result, detail))
                if self.shared.stop or result != "done" or "look" in action:
                    if result != "done":
                        self.shared.queue.clear()  # what was queued after a failure assumed it worked
                    # A stop cleared the queue when it came; what its reply queued after it stays.
                    self.shared.state, self.shared.stop = "paused", False
                elif not self.shared.queue:
                    self.shared.state = "idle"
                self.cond.notify_all()

    def _do(self, action: dict) -> tuple[str, str]:
        try:
            kind = next(iter(action))
            value = action[kind]
            if kind == "key":
                self._input(lambda: self.controller.hotkey(value if isinstance(value, list) else [value]))
            elif kind == "type":
                self._input(lambda: self.controller.type_text(str(value)))
            elif kind in ("click", "double_click", "right_click"):
                point = self._find(str(value))
                if point is None:
                    return "not_found", f"Jev couldn't find {value!r} on the screen"
                self._input(lambda: self.controller.execute(Action(kind, x=point[0], y=point[1]), self.profile))
            elif kind == "drag":
                a, b = self._find(str(value[0])), self._find(str(value[1]))
                if a is None or b is None:
                    return "not_found", f"Jev couldn't find {value[0] if a is None else value[1]!r}"
                self._input(lambda: self.controller.execute(Action("swipe", x=a[0], y=a[1], x2=b[0], y2=b[1], ms=600),
                                                            self.profile))
            elif kind == "scroll":
                if action.get("at"):
                    point = self._find(str(action["at"]))
                    if point is not None:
                        self.controller.hover(*point)
                self._input(lambda: self.controller.execute(Action("scroll", dy=int(value)), self.profile))
            elif kind == "wait":
                time.sleep(min(float(value), 10.0))
            elif kind == "look":
                self._settle(LOOK_SETTLE_S)
            else:
                return "error", f"unknown action {kind!r}"
            return "done", ""
        except (ActionError, HIDError, ValueError, TypeError, KeyError, StopIteration) as exc:
            return "error", f"{type(exc).__name__}: {exc}"[:160]

    def _settle(self, limit_s: float):
        """The screen once two grabs in a row match, or after `limit_s` whatever moves (a video never settles)."""
        deadline = time.monotonic() + limit_s
        frame = self.controller.capture.grab()
        while time.monotonic() < deadline:
            following = self.controller.capture.grab()
            if not self.controller.changed(thumbnail(frame), thumbnail(following)):
                return following
            frame = following
        return frame

    def _input(self, send) -> None:
        self.inputs.append(time.monotonic())
        send()

    def _find(self, target: str) -> tuple[float, float] | None:
        """Jev places the target on the screen as it is now (after it settles), zoomed in once and again."""
        frame = self._settle(CLICK_SETTLE_S)
        jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        started = time.monotonic()
        point = self.jev.locate(jpeg, target)
        if point is None:
            # Jev's presence answers turn on the wording ("119.93 Hz option in the open refresh rate dropdown" 0.08,
            # "119.93 Hz" 0.79, the same box both times, study 16): a sure box gets a zoomed look, which still checks.
            # Likewise a wide list row ("Display Configuration": present 0.98, unique 0.96, box 0.43-0.48).
            answers = (self.jev.trace[-1].get("answers") or {}) if self.jev.trace else {}
            answer = answers.get("location") or {}
            box = answer.get("box")
            seen = min(answers.get("present", {}).get("noul") or 0, answers.get("unique", {}).get("noul") or 0)
            if box and ((answer.get("confidence") or 0) >= SURE_BOX or seen >= GATE):
                point = refine(self.jev, jpeg, target, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
        else:
            point = refine(self.jev, jpeg, target, point) or point
        self.trace.append({"label": "ground", "target": target, "point": point,
                           "elapsed_s": round(time.monotonic() - started, 3)})
        return point


class Brain(threading.Thread):
    """The model loop: calls on speech or executor events, streams actions into the shared queue."""

    def __init__(self, target, shared: Shared, cond: threading.Condition, executor: Executor, model: str,
                 effort: str, started: float, trace: list) -> None:
        super().__init__(daemon=True, name="queue-brain")
        self.target, self.shared, self.cond, self.executor = target, shared, cond, executor
        self.model, self.effort, self.started, self.trace = model, effort, started, trace
        # No kept-alive connections: reusing one after a cancelled stream failed with "bad record mac" (study 16).
        self.client = httpx.Client(timeout=60, limits=httpx.Limits(max_keepalive_connections=0),
                                   headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
        self.retry = False  # the last call failed: call again at once
        self.phase = ("waiting", 0.0)  # what the loop is doing, and since when (for stall reports)
        self.conversation: list[dict] = []  # earlier turns and what was said to the user
        self.words = ""  # the turn being spoken
        self.speaking = False
        self.heard_at = 0.0  # last new speech
        self.active_until = 0.0  # a task is active until then; afterwards only speech wakes the model
        self.dirty = False
        self.cancel_call = threading.Event()
        self.closed = False
        self.calls: list[dict] = []
        self.said: list[str] = []
        self.last_seen = -1  # executor log length the last call saw
        self.shots: Path | None = None  # a folder to keep each call's screenshot in
        self.last_thumb = None  # the screen the last call saw
        self.unchanged = False  # this call's screen is the last call's
        self.last_batch: list[str] = []  # the last reply's actions
        self.repeats = 0  # replies in a row that repeated the one before
        self.looked_only = 0  # replies in a row that did nothing but look

    # -- speech, from the scripted user -------------------------------------------------------------------

    def hear(self, words: str, speaking: bool) -> None:
        with self.cond:
            self.words, self.speaking = words, speaking
            self.heard_at = time.monotonic()
            self.active_until = self.heard_at + SLEEP_S
            self.dirty = True
            self.cancel_call.set()  # latest wins
            self.cond.notify_all()

    def end_turn(self) -> None:
        with self.cond:
            if self.words:
                self.conversation.append({"user": self.words})
            self.words, self.speaking = "", False

    calling = False  # a model call is in flight

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
                self._reconnect()
                self.retry = True
            finally:
                with self.cond:
                    self.calling = False
                    if self.shared.state == "paused":
                        self.shared.state = "idle" if not self.shared.queue else "running"  # ready again
                    self.cond.notify_all()

    def _due(self) -> bool:
        if self.retry:
            self.retry = False
            return True
        if self.dirty:
            return time.monotonic() - self.heard_at >= 0.25  # a short debounce between words
        # Paused (a look or a failure) waits for this call even with actions still queued behind it.
        stopped = self.shared.state == "paused" or (self.shared.state == "idle" and not self.shared.queue)
        news = len(self.shared.log) != self.last_seen
        return stopped and news and time.monotonic() < self.active_until

    def _call(self) -> None:
        self.phase = ("grabbing", round(time.monotonic() - self.started, 2))
        last = self.shared.log[-1].t if self.shared.log else -1.0
        if time.monotonic() - self.started - last < CALL_SETTLE_S:
            # Called on the executor's own input: a menu or popup it opened is still drawing (a dropdown clicked
            # again and again, each screenshot taken before its list showed, study 16).
            frame = self.executor._settle(CALL_SETTLE_S)
        else:
            frame = self.target.grab()
        image = shrink(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
        thumb = thumbnail(frame)
        unchanged = self.last_thumb is not None and not self.executor.controller.changed(self.last_thumb, thumb)
        self.last_thumb = thumb
        self.unchanged = unchanged
        if self.shots is not None:  # what each call saw, for debugging
            (self.shots / f"call-{len(self.calls)}.jpg").write_bytes(image)
        with self.cond:
            self.last_seen = len(self.shared.log)
            context = {
                "rule": "No terminal, no shell commands: only the desktop's own windows, menus and settings.",
                "screen_changed_since_your_last_call": not unchanged,
                **({"note": "Your last reply repeated the actions you had already sent and run, so they were not run "
                            "again. Look at the screen: is the request already done? If so reply with nothing."}
                   if self.repeats else {}),
                "device": DEVICE,
                "user_is_speaking": self.speaking, "user_words_so_far" if self.speaking else "user_said": self.words,
                "conversation_before": self.conversation[-12:],
                "executor": {"state": STATES[self.shared.state],
                             "queued": list(self.shared.queue),
                             "log": [{"t": r.t, "action": r.action, "result": r.result, "detail": r.detail}
                                     for r in self.shared.log[-LOG_TAIL:]]}}
        body = {"model": self.model, "stream": True, "temperature": 0, "provider": {"sort": "latency"},
                "reasoning": {"effort": self.effort} if self.effort else {"enabled": False},
                "messages": [{"role": "system", "content": PROMPT},
                             {"role": "user", "content": [
                                 {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"
                                                                     + base64.b64encode(image).decode()}},
                                 {"type": "text", "text": json.dumps(context)}]}]}
        started = time.monotonic()
        self.phase = ("streaming", round(started - self.started, 2))
        record = {"t": round(started - self.started, 2), "speaking": self.speaking, "words": self.words,
                  "lines": [], "cancelled": False}
        self.calls.append(record)
        text, full, first = "", "", None
        with self.client.stream("POST", OPENROUTER, json=body) as response:
            if response.status_code != 200:
                record["error"] = f"HTTP {response.status_code}: {response.read()[:160]!r}"
                return
            for line in response.iter_lines():
                if self.cancel_call.is_set():
                    record["cancelled"] = True
                    break
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                for choice in json.loads(line[6:]).get("choices", []):
                    piece = (choice.get("delta") or {}).get("content") or ""
                    if piece and first is None:
                        first = time.monotonic() - started
                        record["ttft_s"] = round(first, 3)
                    text += piece
                    full += piece
                    while "\n" in text:
                        done, text = text.split("\n", 1)
                        self._apply(done, record, started)
                if len(record["lines"]) >= MAX_LINES:  # a runaway reply repeating itself: cut it off
                    record["cut"] = True
                    break
        if record["cancelled"]:
            self._reconnect()  # a stream left half-read: reusing its TLS connection failed ("bad record mac")
        else:
            self._apply(text, record, started)  # the last line, which has no newline after it
        if not record["lines"]:
            record["raw"] = full[:600]  # a reply with no action in it: what it said instead
        self._check_repeat(record)
        acted = [line for line in record["lines"] if not {"skipped", "say", "ask"} & line.keys()]
        self.looked_only = self.looked_only + 1 if acted and all("look" in line for line in acted) else 0
        record["total_s"] = round(time.monotonic() - started, 3)

    def _check_repeat(self, record: dict) -> None:
        """A reply that is exactly the previous reply's actions again: the first time, the next call is told so and
        asked whether the request is already done; the second time it counts as done. ("S X 1.5 Enter look" sent a
        dozen times widened a Blender house to 7e14 m, study 16.)"""
        batch = [json.dumps({k: v for k, v in line.items() if k != "at_s"}, sort_keys=True)
                 for line in record["lines"] if "skipped" not in line and "say" not in line and "ask" not in line]
        if batch and batch == self.last_batch:
            self.repeats += 1
            record["repeat"] = self.repeats
            with self.cond:
                self.shared.queue.clear()  # those actions already ran once
                if self.repeats >= 2:
                    self.active_until = 0.0  # done: only new speech wakes the model now
                self.cond.notify_all()
        else:
            self.repeats = 0
        self.last_batch = batch

    def _reconnect(self) -> None:
        self.client.close()
        self.client = httpx.Client(timeout=60, limits=httpx.Limits(max_keepalive_connections=0),
                                   headers=dict(self.client.headers))

    def _apply(self, line: str, record: dict, started: float) -> None:
        line = line.strip().strip("`").strip()
        if not line.startswith("{"):
            return
        try:
            action = json.loads(line)
        except json.JSONDecodeError:
            record["lines"].append({"unparsed": line[:120]})
            return
        if not isinstance(action, dict) or not action:
            return
        if "look" in action and self.looked_only >= 1 and not any(
                not {"skipped", "say", "ask"} & line.keys() for line in record["lines"]):
            # A reply that only looks again right after one that only looked: the model is stuck, not waiting (looks
            # repeated until the watchdog stopped a finished Blender house, study 16); a page still loading has wait.
            record["lines"].append({"at_s": round(time.monotonic() - started, 3), "skipped": action})
            return
        record["lines"].append({"at_s": round(time.monotonic() - started, 3), **action})
        with self.cond:
            if "stop" in action:
                self.shared.queue.clear()
                self.shared.stop = self.shared.state == "running"
            elif "say" in action or "ask" in action:
                self.said.append(str(action.get("say") or action.get("ask")))
                self.conversation.append({"you": self.said[-1]})
            else:
                self.shared.queue.append(action)
                self.active_until = time.monotonic() + SLEEP_S
                if self.shared.state == "paused" and not self.shared.stop:
                    self.shared.state = "running"  # this call saw why it paused: ready again
            self.cond.notify_all()


class QueueAgent:
    """The two threads and their shared queue, on one target."""

    def __init__(self, target, model: str, effort: str, jev_key: str, log_dir: Path) -> None:
        self.cancel = CancelToken()
        self.trace: list[dict] = []
        self.started = time.monotonic()
        self.controller = Controller(target, target, self.cancel)
        self.jev = JevVision(jev_key, self.cancel, log_dir / "jev.jsonl", min_presence=GATE, min_confidence=GATE)
        self.jev.trace = self.trace  # every look Jev takes is kept, beside the grounding records
        frame = target.grab()
        h, w = frame.shape[:2]
        profile = DeviceProfile("linux", (0, 0, w, h), "mouse", time.time())
        self.shared, self.cond = Shared(), threading.Condition()
        self.executor = Executor(self.controller, self.jev, profile, self.shared, self.cond, self.started, self.trace)
        self.brain = Brain(target, self.shared, self.cond, self.executor, model, effort, self.started, self.trace)
        self.brain.shots = log_dir
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
        self.brain.client.close()
