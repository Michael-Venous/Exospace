"""Pure-Jev stand-in for race's planner (EXECUTOR=jev): no language model is called.

Race keeps everything Jev does alone: confident clicks, remembered remarks and the
irreversible-action gate. A request that needs a plan runs here one step per screen:
one batched Jev call judges completion, picks the next action and its target, and picks
the text to type from runs of the goal's own words (Jev selects; it never writes text).
A step Jev can't take, or one with no visible effect, fails and names EXECUTOR=race.

Each request first finds its app the way a person would: the app in front if Jev sees it can do
the request, else a visible window, tab or icon of one, else the OS launcher's list. Nothing
names an app or a site; Jev picks from what the screen shows.
"""

from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from typing import cast

from brain.agent.executor import Plan, StepContext
from brain.agent.jev import COMPLETION_QUESTION, JevVision, probability
from brain.agent.prompts import OS_PROMPT
from brain.agent.triage import GATE, box_center, box_contains, refine
from brain.cancel import CancelToken
from brain.config import Config
from brain.skills import OPEN_APP_KEYS
from brain.types import Action, OSName

OS_CRITERIA = {
    "windows": "Microsoft Windows desktop or app.",
    "macos": "Apple macOS desktop or app.",
    "linux": "A Linux desktop (GNOME, KDE, Xfce and the like) or app.",
    "chromeos": "A Chromebook (ChromeOS shelf and launcher).",
    "android": "An Android phone or tablet.",
    "ios": "An iPhone or iPad.",
    "bios": "Firmware setup, a boot menu or another pre-OS screen.",
    "unknown": "None of the above can be told from this screen.",
}
NEXT_QUESTION = ("The user's `goal` is being carried out on the device in `screen`; `executed_steps` were already "
                 "done. Which single action comes next?")
NEXT_CRITERIA = {
    "click": "Click or tap one visible control: a button, link, tab, menu or list item, or an input field "
             "with nothing to type into it.",
    "fill": "Click an input field and type into it text that the goal gives word for word.",
    "double_click": "Open an icon on a computer desktop, or a file or folder in a desktop file manager.",
    "right_click": "Open the context menu of a visible item.",
    "type": "Type text that the goal gives word for word into the input field that is focused right now.",
    "enter": "Press Enter to submit or confirm the text that was just typed.",
    "scroll_down": "Scroll down to reveal something the goal needs that is below the visible area.",
    "scroll_up": "Scroll up to reveal something the goal needs that is above the visible area.",
    "shortcut": "Press one standard key or keyboard shortcut other than Enter: New (a new note, document or "
                "window), new or close tab, find, save, undo, copy, paste, go back, Escape, Tab, an arrow key...",
    "stuck": "The next step needs reasoning, writing new text, information that isn't visible, or the goal "
             "is unclear.",
}
SHORTCUT_QUESTION = "Which standard key or keyboard shortcut does the next step press?"
SHORTCUT_CRITERIA = {
    "new": "New: create a new note, document, file, message or window in the app in front",
    "new_tab": "Open a new browser tab", "close_tab": "Close the current tab or document window",
    "quit_app": "Quit the app in front", "find": "Find text on this page or document", "save": "Save",
    "undo": "Undo", "redo": "Redo", "select_all": "Select all", "copy": "Copy", "cut": "Cut", "paste": "Paste",
    "back": "Go back to the previous page", "forward": "Go forward to the next page",
    "refresh": "Reload the page", "escape": "Escape: close a menu, dialog or search box",
    "tab": "Tab: move to the next field", "backspace": "Backspace: delete the character before the cursor",
    "delete": "Delete", "space": "Space bar", "up": "Up arrow", "down": "Down arrow", "left": "Left arrow",
    "right": "Right arrow", "page_up": "Page up", "page_down": "Page down", "home": "Home", "end": "End",
}
_PLAIN_KEYS = {"escape": "ESC", "tab": "TAB", "backspace": "BACKSPACE", "delete": "DELETE", "space": "SPACE",
               "up": "UP", "down": "DOWN", "left": "LEFT", "right": "RIGHT", "page_up": "PAGEUP",
               "page_down": "PAGEDOWN", "home": "HOME", "end": "END"}
_COMMAND_LETTERS = {"new": "N", "new_tab": "T", "close_tab": "W", "find": "F", "save": "S", "undo": "Z",
                    "select_all": "A", "copy": "C", "cut": "X", "paste": "V"}
# The only per-OS knowledge: which modifier carries the standard shortcuts, and a few that differ by OS.
_OS_SHORTCUTS: dict[str, dict[str, tuple[str, ...]]] = {
    "macos": {"redo": ("GUI", "SHIFT", "Z"), "quit_app": ("GUI", "Q"), "back": ("GUI", "LEFT"),
              "forward": ("GUI", "RIGHT"), "refresh": ("GUI", "R")},
    "windows": {"redo": ("CTRL", "Y"), "quit_app": ("ALT", "F4"), "back": ("ALT", "LEFT"),
                "forward": ("ALT", "RIGHT"), "refresh": ("F5",)},
    "linux": {"redo": ("CTRL", "SHIFT", "Z"), "quit_app": ("CTRL", "Q"), "back": ("ALT", "LEFT"),
              "forward": ("ALT", "RIGHT"), "refresh": ("F5",)},
    "chromeos": {"redo": ("CTRL", "SHIFT", "Z"), "back": ("ALT", "LEFT"), "forward": ("ALT", "RIGHT"),
                 "refresh": ("CTRL", "R")},
}
SHORTCUT_GATE = 0.6  # the chosen shortcut's probability; below it the step fails instead of guessing a key
SAME_SPOT = 20  # history coordinates are 0-1000: a click this close to the last one is the same click
TEXT_QUESTION = ("Which text does the next step type? It must be copied word for word from the `goal`, without the "
                 "command words or field name around it. Choose none if the text must be computed, looked up, "
                 "translated or written rather than copied.")
REPLACE_QUESTION = ("Is the input the next step types into a single-line box, such as a search, address or form "
                    "field, that already holds text?")
FITS_QUESTION = ("Can the `goal` be carried out in the app or web page in front in `screen`, without switching to "
                 "or opening another app?")
APP_PRESENT = ("Is a window, tab or app icon of an app suited to the `goal` visible in `screen`, such as in a "
               "taskbar, dock, desktop or open app launcher?")
APP_TARGET = "The visible window, tab or app icon in `screen` of the app best suited to the `goal`."
APP_NAME = ("Which words of the `goal` name the app, or the kind of app, to open? They are typed into the app "
            "launcher's search.")
LAUNCHER_OPEN = "Is an app launcher or start menu open in `screen`, with a search field for typing an app's name?"
STEP_CONTROL = "the one control that the next step of the `goal` acts on, after the `executed_steps`"
WAIT = Action("skill", name="wait")
# Race's COMPLETION_GATE (0.3) let two unfinished goals through here, at 0.57 and 0.63 (2026-10-18: Google
# results for "go to Wikipedia", a typed search never sent); finished ones scored 0.68-0.89.
# shortcut: set from 5 live runs; move it if finished goals below 0.65 keep taking an extra step.
COMPLETION_GATE = 0.65
NEAR_PRESENCE = 0.75  # a target this sure is clicked only where both zoomed looks confirm it
# shortcut: the "+" zooms landed 0.002 apart; a neighbouring row is ~0.02 away. Both zooms see the same crop, so they
# can stray together; tighten or drop this if a run clicks a neighbour both zooms picked.
ZOOMS_AGREE = 0.01
MAX_SPANS = 250  # Jev's choice budget (bench/interpreter_screen.py)
DEADLINE_S = 20.0  # user decision (raised from 5 s): a request Jev alone can't finish in 20 s isn't reliable enough


class JevOnlyPlanner:
    def __init__(self, cfg: Config, cancel: CancelToken) -> None:
        self.jev = JevVision(cfg.typesafe_api_key, cancel, cfg.model_log, cfg.jev_model,
                             cfg.jev_timeout_s, cfg.jev_min_presence, cfg.jev_min_confidence)
        self.routed: str | None = None  # the run whose app Jev has seen can do its goal

    def acknowledge(self, executed: bool) -> None:
        pass  # no plan cursor: every step is chosen afresh from the screen and the history

    def next_actions(self, image: bytes, ctx: StepContext) -> Plan:
        texts = spans(ctx.goal)
        options = {str(i): json.dumps(text, ensure_ascii=False) for i, text in enumerate(texts)}
        # Until Jev sees the app in front can do the goal, each step also picks the app to use.
        routing = ctx.profile.os in OPEN_APP_KEYS and self.routed != ctx.run_id  # firmware has no apps
        route = {"fits": {"type": "noul", "instructions": FITS_QUESTION},
                 "app_present": {"type": "noul", "instructions": APP_PRESENT},
                 "app": {"type": "box", "instructions": APP_TARGET},
                 "app_name": {"type": "choice", "instructions": APP_NAME,
                              "criteria": {"none": "The goal names no app or kind of app."} | options},
                 "launcher": {"type": "noul", "instructions": LAUNCHER_OPEN}} if routing else {}
        answers, size, padded = self.jev.evaluate(image, route | {
            "complete": {"type": "noul", "instructions": COMPLETION_QUESTION},
            "next": {"type": "choice", "instructions": NEXT_QUESTION, "criteria": NEXT_CRITERIA},
            "present": {"type": "noul", "instructions":
                        "Is the one control that the next step of the `goal` acts on visible in `screen`, "
                        "with no ambiguity?"},
            "target": {"type": "box", "instructions":
                       "The single visible control in `screen` that the next step of the `goal` acts on."},
            "text": {"type": "choice", "instructions": TEXT_QUESTION,
                     "criteria": {"none": "No text to type."} | options},
            "replace": {"type": "noul", "instructions": REPLACE_QUESTION},
            "shortcut": {"type": "choice", "instructions": SHORTCUT_QUESTION, "criteria": SHORTCUT_CRITERIA},
        }, run_id=ctx.run_id, step=ctx.step, state={"goal": ctx.goal, "executed_steps": ctx.history}, label="step",
           gates={"complete": COMPLETION_GATE, "present": self.jev.min_presence, "replace": GATE, "fits": GATE,
                  "app_present": self.jev.min_presence, "launcher": self.jev.min_presence})
        # Jev's own next step is Enter right after typing: the text isn't submitted, whatever the completion score.
        unsent = bool(ctx.history) and ctx.history[-1].startswith("type_text") and answers["next"].get(
            "choice") == "enter"
        if probability(answers["complete"].get("noul")) >= COMPLETION_GATE and not unsent:
            return Plan("Jev verified the completed goal", [Action("done", summary="Done.")], [])
        if "unchanged" in ctx.note:
            return _fail("the last step had no visible effect", ctx.history)  # Jev would choose it again
        if routing:
            # Before the fits check: Jev can judge the open launcher itself as fitting and type the whole goal.
            search = self._search(answers, ctx, texts)
            if search:
                return search
            if probability(answers["fits"].get("noul")) < GATE:
                return self._route(answers, size, padded, ctx)
            self.routed = ctx.run_id
        step = answers["next"].get("choice")
        actions: list[Action] = []
        if step in ("click", "double_click", "right_click", "fill"):
            point = box_center(answers["target"], size, padded)
            sure = probability(answers["present"].get("noul"))
            if point is None or sure < NEAR_PRESENCE:
                return _fail("its target isn't clearly visible", ctx.history)
            # Two more looks at an enlarged crop: the first box can land a row off (a missed search box), but
            # one zoom also strays to a neighbouring row about 1 in 6 times (Chrome's new tab: search box vs
            # shortcuts). Keeping the zoom nearer the first box is a majority of the three looks.
            state = {"goal": ctx.goal, "executed_steps": ctx.history}
            with ThreadPoolExecutor(2) as pool:
                looks = [p for p in pool.map(lambda _: refine(self.jev, image, state, point, control=STEP_CONTROL,
                                                               run_id=ctx.run_id, step=ctx.step), range(2)) if p]
            # Jev's doubt about which step comes next also lowers its sureness that the target is visible (a
            # small YouTube search box: 0.77-0.84 on 5 tries, each box on it), so agreeing looks stand in for it.
            # Both zooms moving to one spot agree too: for "open a new tab and search ..." the first box was the
            # page's search box and both zooms found the tab strip's "+" (frame 49, 20/20 on 10 replays).
            agree = len(looks) == 2 and (all(box_contains(answers["target"], size, padded, p) for p in looks)
                                         or math.dist(*looks) < ZOOMS_AGREE)
            if not looks or sure < self.jev.min_presence and not agree:
                return _fail("its target isn't clearly visible", ctx.history)
            point = min(looks, key=lambda p: math.dist(p, point))
            actions.append(Action("click" if step == "fill" else step, x=point[0], y=point[1]))
        if step in ("type", "fill"):
            choice = answers["text"].get("choice")
            if choice not in {str(i) for i in range(len(texts))}:
                return _fail("the text to type isn't in the request", ctx.history)
            if _is_the_request(texts[int(choice)], ctx.goal):
                # "write a short note about my day": the request asks for new text, it doesn't dictate it.
                return _fail("the request asks for text to be written, not typed word for word", ctx.history)
            if probability(answers["replace"].get("noul")) >= GATE and ctx.profile.os != "bios":
                # Select-all only in a filled single-line box: in a document it would replace everything.
                actions.append(Action("hotkey", keys=("GUI" if ctx.profile.os in ("macos", "ios") else "CTRL", "A")))
            actions.append(Action("type_text", text=texts[int(choice)]))
        if step == "enter":
            actions.append(Action("hotkey", keys=("ENTER",)))
        if step in ("scroll_down", "scroll_up") and ctx.profile.os != "bios":  # firmware has no wheel
            actions += scroll_actions(5 if step == "scroll_down" else -5, ctx.profile.input)
        if step == "shortcut":
            answer = answers["shortcut"]
            keys = shortcut_keys(answer.get("choice"), ctx.profile.os)
            if keys is None or _chosen_probability(answer) < SHORTCUT_GATE:
                return _fail("it isn't sure which key the step needs", ctx.history)
            actions.append(Action("hotkey", keys=keys))
        if not actions:
            return _fail("it needs reasoning or content Jev can't supply", ctx.history)
        if _repeats(actions, ctx.history):
            # The same click, text or key again right after it ran: a click on "1" twice typed "11" for "12",
            # and Enter twice submitted twice. Never repeat input to find out whether it worked.
            return _fail("it would repeat the step it just took", ctx.history)
        return Plan(f"Jev next step: {step}", actions, [])

    def _search(self, answers: dict, ctx: StepContext, texts: list[str]) -> Plan | None:
        """Search the launcher this run opened for the app the goal names, once. Before any launcher icon is
        clicked: an icon has no name to check (a camera icon was the screenshot tool)."""
        launcher = Action("hotkey", keys=tuple(OPEN_APP_KEYS[ctx.profile.os])).brief()
        if launcher not in ctx.history or any(a.startswith("type_text")
                                              for a in ctx.history[ctx.history.index(launcher):]):
            return None
        name = answers["app_name"].get("choice")
        # Typed only into a launcher Jev sees, so one still opening can't take the text.
        if name in {str(i) for i in range(len(texts))} and probability(
                answers["launcher"].get("noul")) >= self.jev.min_presence:
            return Plan("Search the launcher", [Action("type_text", text=texts[int(name)])], [])
        return None

    def _route(self, answers: dict, size, padded, ctx: StepContext) -> Plan:
        """Switch to the visible app best suited to the goal, else open the OS launcher to show more."""
        launcher = Action("hotkey", keys=tuple(OPEN_APP_KEYS[ctx.profile.os]))
        point = box_center(answers["app"], size, padded)
        if point is not None and probability(answers["app_present"].get("noul")) >= self.jev.min_presence:
            # shortcut: one click opens taskbar, dock and launcher entries but not desktop icons; double-click
            # when the app box is a desktop icon if those get missed.
            return Plan("Switch to the app for the request", [Action("click", x=point[0], y=point[1])], [])
        if launcher.brief() not in ctx.history:
            return Plan("Open the app launcher", [launcher], [])
        if ctx.history[-1] != WAIT.brief():  # the launcher, its results or the chosen app still appearing
            return Plan("Waiting for the app", [WAIT], [])
        return _fail("no app suited to the request is visible", ctx.history)

    def classify_os(self, image: bytes) -> OSName:
        answers, _, _ = self.jev.evaluate(image, {"os": {"type": "choice", "instructions": OS_PROMPT,
                                                         "criteria": OS_CRITERIA}})
        os = answers["os"].get("choice")
        if os not in OS_CRITERIA:
            raise ValueError("invalid operating system classification")
        return cast(OSName, os)

    def close(self) -> None:
        self.jev.close()


def spans(goal: str) -> list[str]:
    """Each run of whole words in the goal without wrapping quotes or trailing punctuation,
    shortest first. Whole runs bind text better than separate first/last words (HANDOFF.md)."""
    words = list(re.finditer(r"\S+", goal))
    runs = (goal[words[a].start():words[a + n].end()].strip("\"'“”‘’,;.")
            for n in range(len(words)) for a in range(len(words) - n))
    # shortcut: a long goal loses its longest runs past the budget; split the goal if long literals get cut.
    return [run for run in dict.fromkeys(runs) if run][:MAX_SPANS]


def shortcut_keys(choice: object, os: str) -> tuple[str, ...] | None:
    """The keys for a SHORTCUT_CRITERIA choice on this OS, or None where it has none (phones, firmware)."""
    if choice in _PLAIN_KEYS:
        return (_PLAIN_KEYS[choice],)
    if os in _OS_SHORTCUTS and choice in _OS_SHORTCUTS[os]:
        return _OS_SHORTCUTS[os][choice]
    if os in _OS_SHORTCUTS and choice in _COMMAND_LETTERS:
        return ("GUI" if os == "macos" else "CTRL", _COMMAND_LETTERS[choice])
    return None


def scroll_actions(dy: int, input_mode: str) -> list[Action]:
    """A scroll, first moving the pointer to the middle of the screen with a mouse: a Mac scrolls whatever is
    under the pointer, so one resting on a toolbar or the dock scrolls nothing ("no visible effect")."""
    hover = [Action("skill", name="hover", args={"x": 0.5, "y": 0.5})] if input_mode == "mouse" else []
    return hover + [Action("scroll", dy=dy)]


def _chosen_probability(answer: dict) -> float:
    value = (answer.get("probabilities") or {}).get(answer.get("choice"), answer.get("confidence", 1.0))
    try:
        return probability(value)
    except ValueError:
        return 0.0


def _words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _is_the_request(text: str, goal: str) -> bool:
    """True when the text to type is (nearly) the whole request: an instruction, not dictated words."""
    request = _words(goal.splitlines()[-1])
    return len(request) >= 3 and len(_words(text)) >= len(request) - 1


_POINT = re.compile(r"^(click|double_click|right_click)\((\d+),(\d+)\)$")


def _repeats(actions: list[Action], history: list[str]) -> bool:
    """True when these actions are the same input as the last ones that ran (a click within SAME_SPOT, the same
    text or the same keys). Scrolls, waits and pointer moves may repeat."""
    inputs = [a.brief() for a in actions if a.type not in ("scroll", "skill")]
    recent = [h for h in history if not h.startswith(("scroll(", "skill("))][-len(inputs):] if inputs else []
    if not inputs or len(recent) < len(inputs):
        return False
    for new, old in zip(inputs, recent):
        a, b = _POINT.match(new), _POINT.match(old)
        if a and b:
            if a[1] != b[1] or abs(int(a[2]) - int(b[2])) > SAME_SPOT or abs(int(a[3]) - int(b[3])) > SAME_SPOT:
                return False
        elif new != old:
            return False
    return True


def _fail(why: str, done: list[str] | tuple[str, ...] = ()) -> Plan:
    """Fail, saying what already ran: "typed it, pressed Enter, then couldn't confirm" reads very differently
    from "did nothing" to someone who can't see the screen."""
    so_far = f" Done so far: {', '.join(done[-4:])}." if done else ""
    return Plan("Jev can't take the next step", [Action(
        "fail", reason=f"Jev alone can't take the next step: {why}.{so_far} "
                       "The language-model planner (EXECUTOR=race) can.")], [])
