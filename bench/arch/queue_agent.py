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
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import httpx

from bench.arch.pointer import Pointer
from brain.agent.jev import JevVision
from brain.agent.jev_flash import COMPLETION_GATE
from brain.agent.triage import GATE, refine
from brain.cancel import CancelToken
from brain.controller import Controller
from brain.imaging import shrink
from brain.types import Action, DeviceProfile

OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
DEVICE = "Linux laptop, KDE Plasma desktop. The GUI key opens the application launcher. The panel hides itself."
SLEEP_S = 8.0  # no speech and no action for this long: the model stops being called
READ_S = 3.0  # longest wait for the model's next streamed bytes, its first token included
LOG_TAIL = 20  # executor log entries each call sees
HISTORY = 4  # earlier calls of the request each call sees as chat turns
# Each model on one host. Qwen's quality depends on it (study 17: 192/192 pointed on ModelRun, 15/192 on Cerebras);
# DeepSeek was fastest on Together (model check: 0.33 s first token p50).
PINNED = {"qwen/qwen3.8-27b": "modelrun/fp4", "deepseek/deepseek-v4.1-flash-1017": "together",
          "z-ai/glm-5.3-flash": "together"}
# Pointing models that can click in Jev's place, each on the host study 17 measured it on.
# (model, host, reasoning effort: "" = off, x/y in pixels). GLM 5.3 Flash's hosts all require some reasoning, and
# it answers in pixels.
CLICKERS = {"qwen": ("qwen/qwen3.8-27b", PINNED["qwen/qwen3.8-27b"], "", False),
            "deepseek": ("deepseek/deepseek-v4.1-flash-1017", "inference-net/fp8", "", False),
            "glm": ("z-ai/glm-5.3-flash", PINNED["z-ai/glm-5.3-flash"], "minimal", True)}
SURE_BOX = 0.7  # Jev's box confidence at which a target its presence check refused still gets a zoomed look
PROMPT = """You operate a computer for a user who speaks to you. You see its screen; you act only through
its keyboard and mouse. The user's words arrive while they speak.

Each time you are called you get the screen as it is now and the log of the actions you took and their
results. First check on the screen whether your last actions did what you meant. Then reply with the next
actions, ONE JSON object per line, nothing else: as many as you can take without seeing the screen in between
(CTRL+L, typing an address and ENTER are one reply), stopping at the first whose result you need to see. If the
screen is still loading or changing, reply with nothing and you will see it again.
  {"key": ["CTRL","L"]}             a key or combination (CTRL SHIFT ALT GUI ENTER ESC TAB BACKSPACE DELETE UP
                                    DOWN LEFT RIGHT HOME END PAGEUP PAGEDOWN F1-F12, letters, digits)
  {"type": "lofi hip hop"}          type these words into whatever has focus
  {"click": "Reply"}                click a visible control, named by its exact visible text when it has any
                                    ("Reply", "119.93 Hz"); an icon by its look and place ("the gear icon at the
                                    top right"); also "double_click", "right_click"
  {"drag": ["the file report.pdf", "the Documents folder in the sidebar"]}   press on one spot, release on another
  {"scroll": 3, "at": "the list of results"}   mouse-wheel notches, positive down ("at" optional)
  {"done": "YouTube's home page is open in Firefox"}   the request is done: say what the screen shows that
                                    proves it; it is checked on the screen before it counts
  {"say": "It's running at 120 Hz."} / {"ask": "Which wallpaper would you like?"}   talk to the user
A confirmed done, or a reply that ends with say or ask, ends your turn: the user speaks next.
Rules:
- Judge what the user means from the screen and the conversation before acting.
- While you work, reply with actions only: never announce what you are about to do. Talk when you have the answer
  or the result, or to ask.
- If the log shows an action failed, or the screen shows it changed nothing, don't repeat it: try another way.
- State only what the screen shows; never guess a result.
- Use keyboard shortcuts where they are sure. No key can be held during a click or drag."""


# Planning ahead (`plan="ahead"`): steps on screens not drawn yet are queued too, and a click waits for its target
# to appear, so each step waits on the computer, not on the model.
APPEAR_S = 5.0  # how long a click waits for its target to appear before it fails
PROMPT_AHEAD = PROMPT.replace(
    """Then reply with the next
actions, ONE JSON object per line, nothing else: as many as you can take without seeing the screen in between
(CTRL+L, typing an address and ENTER are one reply), stopping at the first whose result you need to see. If the
screen is still loading or changing, reply with nothing and you will see it again.""",
    """Then reply with the next
actions, ONE JSON object per line, nothing else. Plan ahead: queue every step you can already foresee, also on
screens that will only appear after earlier steps run; a click waits for its target to appear. Stop at the first
step that depends on something you can't foresee (which result to pick, what a page says): you are called again
when the queue runs out or a step fails. If the screen is still loading or changing, reply with nothing and you
will see it again.""")


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

    def run(self) -> None:
        while not self.closed:
            with self.cond:
                while not self.closed and not self.shared.queue:
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
                if result not in ("done", "verified"):
                    self.shared.queue.clear()  # the rest of the reply assumed this worked
                if not self.shared.queue:
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
                    return "not_found", f"{value!r} wasn't found on the screen"
                self._input(lambda: self.controller.execute(Action(kind, x=point[0], y=point[1]), self.profile))
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
            else:
                return "error", f"unknown action {kind!r}"
            return "done", ""
        except Exception as exc:  # noqa: BLE001 - a failed action is logged for the model; the executor carries on
            return "error", f"{type(exc).__name__}: {exc}"[:160]

    def _jpeg(self) -> bytes:
        """The screen as it is now: no waiting for it to settle."""
        return cv2.imencode(".jpg", self.controller.capture.grab(), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    def _input(self, send) -> None:
        self.inputs.append(time.monotonic())
        send()

    def _verify(self, claim: str) -> tuple[str, str]:
        """The request counts as done only if Jev's completion check sees it done on the screen."""
        with self.cond:
            goal = self.shared.goal
            steps = [json.dumps(r.action) for r in self.shared.log[self.shared.turn_start:] if r.result == "done"]
        score = self.jev.completion(self._jpeg(), goal, steps, evidence=claim)
        if score >= COMPLETION_GATE:
            return "verified", f"Jev sees the request done ({score:.2f})"
        return "not_verified", f"Jev does not see the request done on the screen ({score:.2f}): it isn't done yet"

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
        self.history: list[dict] = []  # this request's earlier calls: what each saw and replied

    @staticmethod
    def _client() -> httpx.Client:
        # One connection per call: one reused after a cancelled stream failed with "bad record mac". A reply that
        # stalls READ_S (a provider's first token once took 8.4 s against a usual 0.45 s) fails and is asked again.
        return httpx.Client(timeout=httpx.Timeout(60, read=READ_S), limits=httpx.Limits(max_keepalive_connections=0),
                            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})

    # -- speech, from the scripted user -------------------------------------------------------------------

    def hear(self, words: str, speaking: bool) -> None:
        with self.cond:
            if not self.words:  # a new request starts here
                self.shared.turn_start = len(self.shared.log)
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
            finally:
                with self.cond:
                    self.calling = False
                    self.cond.notify_all()

    def _due(self) -> bool:
        if self.dirty:
            return True
        if self.shared.finished or self.shared.queue or self.shared.state == "running":
            return False
        return time.monotonic() < self.active_until

    def _call(self) -> None:
        self.phase = ("grabbing", round(time.monotonic() - self.started, 2))
        frame = self.target.grab()
        image = shrink(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
        if self.shots is not None:  # what each call saw, for debugging
            (self.shots / f"call-{len(self.calls)}.jpg").write_bytes(image)
        with self.cond:
            context = {
                "device": DEVICE,
                "user_is_speaking": self.speaking, "user_words_so_far" if self.speaking else "user_said": self.words,
                "conversation_before": self.conversation[-12:],
                "log": [{"t": r.t, "action": r.action, "result": r.result, "detail": r.detail}
                        for r in self.shared.log[-LOG_TAIL:]]}
        # The request so far as a chat: each earlier call's situation and the model's own reply (text only), so it
        # remembers what it meant to do and already checked; the screenshot rides on this call alone.
        history = []
        for past in self.history[-HISTORY:]:
            history += [{"role": "user", "content": past["context"]},
                        {"role": "assistant", "content": past["reply"] or "(nothing)"}]
        provider = ({"order": [PINNED[self.model]], "allow_fallbacks": False} if self.model in PINNED
                    else {"sort": "latency"})
        body = {"model": self.model, "stream": True, "temperature": 0, "provider": provider,
                "reasoning": {"effort": self.effort} if self.effort else {"enabled": False},
                "messages": [{"role": "system", "content": self.prompt}, *history,
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
            if record["lines"] and record["lines"][-1].keys() & {"say", "ask"}:
                with self.cond:  # a reply that ends by talking hands the turn back ("I'll open it" before acting doesn't)
                    self.shared.finished = True
        if not record["lines"]:
            record["raw"] = full[:600]  # a reply with no action in it: what it said instead
        if not record["cancelled"] and full.strip():  # a call that added nothing isn't worth remembering
            past = {k: v for k, v in context.items() if k != "conversation_before"}
            self.history.append({"context": json.dumps(past), "reply": full.strip()})
        record["total_s"] = round(time.monotonic() - started, 3)

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

    def _take(self, action: dict, record: dict, started: float) -> None:
        record["lines"].append({"at_s": round(time.monotonic() - started, 3), **action})
        with self.cond:
            if "say" in action or "ask" in action:
                self.said.append(str(action.get("say") or action.get("ask")))
                self.conversation.append({"you": self.said[-1]})
            else:
                self.shared.queue.append(action)
                self.active_until = time.monotonic() + SLEEP_S
            self.cond.notify_all()


class QueueAgent:
    """The model and executor threads, taking turns, on one target."""

    def __init__(self, target, model: str, effort: str, jev_key: str, log_dir: Path, clicker: str = "qwen",
                 plan: str = "screen") -> None:
        """`plan` "screen": each reply covers one screen; "ahead": replies plan across screens, clicks wait."""
        self.cancel = CancelToken()
        self.trace: list[dict] = []
        self.started = time.monotonic()
        self.controller = Controller(target, target, self.cancel)
        self.jev = JevVision(jev_key, self.cancel, log_dir / "jev.jsonl", min_presence=GATE, min_confidence=GATE)
        self.jev.trace = self.trace  # every Jev call is kept, beside the grounding records
        frame = target.grab()
        h, w = frame.shape[:2]
        profile = DeviceProfile("linux", (0, 0, w, h), "mouse", time.time())
        self.shared, self.cond = Shared(), threading.Condition()
        self.executor = Executor(self.controller, self.jev, profile, self.shared, self.cond, self.started, self.trace)
        if clicker in CLICKERS:
            self.executor.pointer = Pointer(self.cancel, self.jev, *CLICKERS[clicker])
        self.brain = Brain(target, self.shared, self.cond, self.executor, model, effort, self.started)
        if plan == "ahead":
            self.executor.appear_s, self.brain.prompt = APPEAR_S, PROMPT_AHEAD
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
        if self.executor.pointer is not None:
            self.executor.pointer.close()
        self.brain.client.close()
