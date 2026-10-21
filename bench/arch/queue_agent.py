"""Candidate architecture (study 16): a fast LLM at the front, a shared action queue, Jev as clicker and checker.

  Brain     one fast vision model. Called on new speech (latest wins: an update cancels the call in flight)
            and whenever the executor stops (end of queue, a `look`, a failure) while a task is active. Each
            call sees the user's words so far, the conversation, the executor's log and queue, and the screen as
            it is. It streams one action per line and each is applied as it arrives: appended to the queue, or
            `stop` (the queue is cleared; the rest of the reply refills it).
  Executor  runs the queue in order: keys and typing go straight to the HID; a click names its target and Jev
            places it on the fresh screen (locate + zoom). A target Jev can't find, or any error, pauses the
            queue and is logged, and the brain's next call sees why.
  Checks    the model says what it expects; Jev confirms it on the screen. At a `look` Jev reports whether the
            screen shows what the model expected. A request ends only with `done`, which Jev's completion check
            (goal + executed steps + screen) must confirm; otherwise the model is called again with the reason.
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

from brain.agent.jev import JevVision, probability
from brain.agent.jev_flash import COMPLETION_GATE
from brain.agent.triage import GATE, refine
from brain.cancel import CancelToken
from brain.controller import Controller
from brain.hid import HIDError
from brain.imaging import shrink
from brain.types import Action, ActionError, DeviceProfile

OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
DEVICE = "Linux laptop, KDE Plasma desktop. The GUI key opens the application launcher. The panel hides itself."
STATES = {"running": "running your actions", "paused": "stopped, waiting for your next actions",
          "idle": "finished all the actions you queued"}
SLEEP_S = 8.0  # no speech and nothing to do for this long: screen changes stop calling the model
LOG_TAIL = 20  # executor log entries each call sees
SURE_BOX = 0.7  # Jev's box confidence at which a target its presence check refused still gets a zoomed look
PROMPT = """You operate a computer for a user who speaks to you. You see its screen; you act only through
its keyboard and mouse. Never open a terminal or type shell commands unless the user explicitly asks for one:
do everything through the desktop's own interface, as the user would. The user's words arrive while they
speak; you are called again as they grow, and whenever your queued actions run out, pause or fail.

Reply with actions, ONE JSON object per line, nothing else. They run in order as soon as each line arrives:
  {"key": ["CTRL","L"]}             a key or combination (CTRL SHIFT ALT GUI ENTER ESC TAB BACKSPACE DELETE UP
                                    DOWN LEFT RIGHT HOME END PAGEUP PAGEDOWN F1-F12, letters, digits)
  {"type": "lofi hip hop"}          type these words into whatever has focus
  {"click": "Reply"}                click a visible control, named by its exact visible text when it has any
                                    ("Reply", "119.93 Hz"); an icon by its look and place ("the gear icon at the
                                    top right"); also "double_click", "right_click"
  {"drag": ["the file report.pdf", "the Documents folder in the sidebar"]}   press on one spot, release on another
  {"scroll": 3, "at": "the list of results"}   mouse-wheel notches, positive down ("at" optional)
  {"wait": 1.5}                     seconds, when something you expect is still loading or drawing
  {"look": "the search results page"}   stop and show me the screen again; say what you expect to see, and
                                    the log will say whether the screen shows it
  {"done": "YouTube's home page is open in Firefox"}   the request is done: say what the screen shows that
                                    proves it; it is checked on the screen before it counts
  {"stop": true}                    first, to cancel queued actions (e.g. the user changed their mind)
  {"say": "It's running at 120 Hz."} / {"ask": "Which wallpaper would you like?"}   talk to the user
No key can be held during a click or drag. Judge what the user means from the screen and the conversation
before acting. Use keyboard shortcuts where they are sure; click when the target is visible. After anything
that changes the screen in a way you need to see (a page or app opening, a menu, a search), end with look.
When you need the user's answer, end with ask. The executor log tells you what ran, what failed and why, and
what each look and done found; never repeat what it shows done."""


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
    goal: str = ""  # what the user has asked for, for the done check
    turn_start: int = 0  # the log entry the current request starts at
    finished: bool = False  # a done that Jev confirmed: only new speech wakes the model


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
                if result == "verified":
                    self.shared.finished = True
                if self.shared.stop or result not in ("done", "verified") or "look" in action:
                    if result not in ("done", "verified", "checked"):
                        self.shared.queue.clear()  # what was queued after a failure assumed it worked
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
                return "checked", self._shown(str(value))
            elif kind == "done":
                return self._verify(str(value))
            else:
                return "error", f"unknown action {kind!r}"
            return "done", ""
        except (ActionError, HIDError, ValueError, TypeError, KeyError, StopIteration) as exc:
            return "error", f"{type(exc).__name__}: {exc}"[:160]

    def _jpeg(self) -> bytes:
        """The screen as it is now: no waiting for it to settle, so motion on screen (a video) costs nothing; the
        model says `wait` when it expects something still to draw."""
        return cv2.imencode(".jpg", self.controller.capture.grab(), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    def _input(self, send) -> None:
        self.inputs.append(time.monotonic())
        send()

    def _shown(self, expected: str) -> str:
        """Jev's answer to whether the screen shows what the model expected at this look."""
        answers, _, _ = self.jev.evaluate(self._jpeg(), {"shown": {"type": "noul", "instructions":
                                          f"Does `screen` show this: {expected}?"}}, label="look")
        p = probability(answers["shown"].get("noul"))
        return f"Jev: the screen {'shows' if p >= GATE else 'does not show'} that ({p:.2f})"

    def _verify(self, claim: str) -> tuple[str, str]:
        """The request counts as done only if Jev's completion check sees it done on the screen."""
        with self.cond:
            goal = self.shared.goal
            steps = [json.dumps(r.action) for r in self.shared.log[self.shared.turn_start:]
                     if r.result == "done" and not {"look", "wait"} & r.action.keys()]
        score = self.jev.completion(self._jpeg(), goal, steps, evidence=claim)
        if score >= COMPLETION_GATE:
            return "verified", f"Jev sees the request done ({score:.2f})"
        return "not_verified", f"Jev does not see the request done on the screen ({score:.2f}): it isn't done yet"

    def _find(self, target: str) -> tuple[float, float] | None:
        """Jev places the target on the screen as it is now, zoomed in once and again."""
        jpeg = self._jpeg()
        started = time.monotonic()
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
        self.trace.append({"label": "ground", "target": target, "point": point,
                           "elapsed_s": round(time.monotonic() - started, 3)})
        return point


class Brain(threading.Thread):
    """The model loop: calls on speech or executor events, streams actions into the shared queue."""

    def __init__(self, target, shared: Shared, cond: threading.Condition, executor: Executor, model: str,
                 effort: str, started: float) -> None:
        super().__init__(daemon=True, name="queue-brain")
        self.target, self.shared, self.cond, self.executor = target, shared, cond, executor
        self.model, self.effort, self.started = model, effort, started
        self.client = self._client()
        self.retry = False  # the last call failed: call again at once
        self.phase = ("waiting", 0.0)  # what the loop is doing, and since when (for stall reports)
        self.conversation: list[dict] = []  # earlier turns and what was said to the user
        self.words = ""  # the turn being spoken
        self.speaking = False
        self.heard_at = 0.0  # last new speech
        self.active_until = 0.0  # a task is active until then; afterwards only speech wakes the model
        self.dirty = False
        self.calling = False  # a model call is in flight
        self.cancel_call = threading.Event()
        self.closed = False
        self.calls: list[dict] = []
        self.said: list[str] = []
        self.last_seen = -1  # executor log length the last call saw
        self.shots: Path | None = None  # a folder to keep each call's screenshot in

    @staticmethod
    def _client() -> httpx.Client:
        # One connection per call: one reused after a cancelled stream failed with "bad record mac".
        return httpx.Client(timeout=60, limits=httpx.Limits(max_keepalive_connections=0),
                            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})

    # -- speech, from the scripted user -------------------------------------------------------------------

    def hear(self, words: str, speaking: bool) -> None:
        with self.cond:
            if not self.words:  # a new request starts here
                self.shared.turn_start, self.shared.finished = len(self.shared.log), False
            self.words, self.speaking = words, speaking
            self.shared.goal = "\n".join([*(t["user"] for t in self.conversation if "user" in t), words])
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
        if self.shared.finished:
            return False
        # Paused (a look, a failure, an unconfirmed done) waits for this call, even with actions queued behind it.
        stopped = self.shared.state == "paused" or (self.shared.state == "idle" and not self.shared.queue)
        news = len(self.shared.log) != self.last_seen
        return stopped and news and time.monotonic() < self.active_until

    def _call(self) -> None:
        self.phase = ("grabbing", round(time.monotonic() - self.started, 2))
        frame = self.target.grab()
        image = shrink(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
        if self.shots is not None:  # what each call saw, for debugging
            (self.shots / f"call-{len(self.calls)}.jpg").write_bytes(image)
        with self.cond:
            self.last_seen = len(self.shared.log)
            context = {
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
        text, full = "", ""
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
                    if piece and "ttft_s" not in record:
                        record["ttft_s"] = round(time.monotonic() - started, 3)
                    text, full = text + piece, full + piece
                    while "\n" in text:
                        done, text = text.split("\n", 1)
                        self._apply(done, record, started)
        if not record["cancelled"]:
            self._apply(text, record, started)  # the last line, which has no newline after it
        if not record["lines"]:
            record["raw"] = full[:600]  # a reply with no action in it: what it said instead
        record["total_s"] = round(time.monotonic() - started, 3)

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
        self.brain = Brain(target, self.shared, self.cond, self.executor, model, effort, self.started)
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
