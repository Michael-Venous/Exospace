"""Jev's own next step, with no language model: race's planner under EXECUTOR=jev, and the first try
of brain/agent/jev_first.py, which hands a request to Flash once a step fails here.

Race keeps everything Jev does alone: confident clicks, remembered remarks and the
irreversible-action gate. A request that needs a plan runs here one step per screen:
one batched Jev call judges completion, picks the next action and its target, and picks
the text to type from runs of the goal's own words (Jev selects; it never writes text).
A step Jev can't take, or one with no visible effect, fails.

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
from brain.agent.quick import CORRECTION, JUST_DID
from brain.agent.triage import GATE, box_center, look, zoom_window
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
    "done": "The screen shows every part of the goal already carried out; no step remains.",
    "stuck": "The next step needs reasoning, writing new text, information that isn't visible, a key other "
             "than Enter, or the goal is unclear.",
}
TEXT_QUESTION = ("Which text does the next step type? It must be copied word for word from the `goal`, without the "
                 "command words or field name around it. Choose none if the text must be computed, looked up, "
                 "translated or written rather than copied.")
# Asked of the chosen text and each shorter run ending with it: Jev's choice keeps a span's command words ("search for
# embedded systems videos": 0.40-0.45, the bare query 0.05-0.10; reworded questions did no better). Asked of each run,
# the bare text scores highest: "embedded systems videos" 0.77-0.79 vs 0.54-0.65, "cats" 0.55-0.61 vs 0.14-0.26,
# while "hello world" and "how to fix a bike chain" stay whole. Word by word it wasn't steady: "for" 0.39-0.66 (2026-10-18).
TEXT_EXACTLY = ("The next step types text from the `goal`. Is {} exactly the text that gets typed, with none of the "
                "instruction to the agent (what to do, where, or for whom) left in it and none of the text left out?")
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
QUESTION = ("Does the `goal` ask for information, an answer or an explanation, rather than for something to be "
            "done on the device?")
COMPOSED = ("Does what remains of the `goal`, after the `executed_steps`, need text, numbers or an expression "
            "that the `goal` doesn't contain word for word, to be computed, translated or composed?")
STEP_CONTROL = "the one control that the next step of the `goal` acts on, after the `executed_steps`"
# A fill's zoom is told its text: asked only for the next step's control, it judged the step anew ("click Peter", with
# Peter's chat already open) and lost Discord's message bar (present 0.21-0.25, 2026-10-18).
FILL_CONTROL = "the input field that the next step of the `goal` types {} into"
WAIT = Action("skill", name="wait")
LOADING = ("Is a page or app in `screen` still loading: a browser's stop (X) button where reload would be, "
           "a spinner, a progress bar, or a page only partly drawn?")
MAX_WAITS = 3  # a page that never settles (lazy-loading feeds) still gets its next step
# Race's COMPLETION_GATE (0.3) let two unfinished goals through here, at 0.57 and 0.63 (2026-10-18: Google
# results for "go to Wikipedia", a typed search never sent); finished ones scored 0.68-0.89.
# shortcut: set from 5 live runs; move it if finished goals below 0.65 keep taking an extra step.
COMPLETION_GATE = 0.65
# shortcut: the "+" zooms landed 0.002 apart; a neighbouring row is ~0.02 away. Both zooms see the same crop, so they
# can stray together; tighten or drop this if a run clicks a neighbour both zooms picked.
ZOOMS_AGREE = 0.01
MAX_SPANS = 250  # Jev's choice budget (bench/interpreter_screen.py)
FAILED = "Jev can't take the next step"
DEADLINE_S = 20.0  # user decision (raised from 5 s): a request Jev alone can't finish in 20 s isn't reliable enough


class JevOnlyPlanner:
    def __init__(self, jev: JevVision) -> None:
        self.jev = jev
        self.routed: str | None = None  # the run whose app Jev has seen can do its goal

    def acknowledge(self, executed: bool) -> None:
        pass  # no plan cursor: every step is chosen afresh from the screen and the history

    def forget_plan(self) -> None:
        pass  # nothing speculative to drop (RaceExecutor calls this after a launch)

    def next_actions(self, image: bytes, ctx: StepContext) -> Plan:
        texts = spans(ctx.goal)
        options = {str(i): json.dumps(text, ensure_ascii=False) for i, text in enumerate(texts)}
        # Until Jev sees the app in front can do the goal, each step also picks the app to use.
        # An app a quick command opened is the one asked for: Jev judged Display's own window unfit for "open up
        # display" and opened the launcher over it. Firmware has no apps.
        routing = ctx.profile.os in OPEN_APP_KEYS and self.routed != ctx.run_id and not any(
            a.startswith("skill(open_app") for a in ctx.history)
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
            "question": {"type": "noul", "instructions": QUESTION},
            "composed": {"type": "noul", "instructions": COMPOSED},
            "loading": {"type": "noul", "instructions": LOADING},
        }, run_id=ctx.run_id, step=ctx.step, state={"goal": ctx.goal, "executed_steps": ctx.history}, label="step",
           gates={"complete": COMPLETION_GATE, "present": self.jev.min_presence, "replace": GATE, "fits": GATE,
                  "app_present": self.jev.min_presence, "launcher": self.jev.min_presence,
                  "question": GATE, "composed": GATE})
        # Jev's own next step is Enter right after typing: the text isn't submitted, whatever the completion score.
        unsent = bool(ctx.history) and ctx.history[-1].startswith("type_text") and answers["next"].get(
            "choice") == "enter"
        if probability(answers["question"].get("noul")) >= GATE:
            # Jev called "what is this page about?" complete and said "Done." with no answer.
            return _fail("the request asks for an answer, not an action")
        if probability(answers["composed"].get("noul")) >= GATE:
            # "work out 12 times 9": Jev pressed keypad keys one per step and missed one (41.7 s; Flash typed
            # "12*9=" in one step, 12.2 s).
            return _fail("the rest needs text the request doesn't give word for word")
        # A click that starts loading a page leaves the old page on screen, unchanged, for about a second: acting
        # on it typed into Google's box while YouTube loaded, and then the typed spaces scrolled YouTube.
        waits = next((i for i, a in enumerate(reversed(ctx.history)) if a != WAIT.brief()), len(ctx.history))
        if probability(answers["loading"].get("noul")) >= GATE and waits < MAX_WAITS:
            return Plan("Waiting for the page", [WAIT], [])
        # Jev choosing no next step counts too: two sent Discord chats scored "complete" 0.29 and 0.59 but "done"
        # 0.85-0.98, a message left unsent behind an open "+" menu 0.33-0.35 (2026-10-18).
        done = (answers["next"].get("probabilities") or {}).get("done", 0)
        if max(probability(answers["complete"].get("noul")), probability(done)) >= COMPLETION_GATE and not unsent:
            return Plan("Jev verified the completed goal", [Action("done", summary="Done.")], [])
        if "unchanged" in ctx.note:
            return _fail("the last step had no visible effect")  # Jev would choose it again
        if routing:
            # Before the fits check: Jev can judge the open launcher itself as fitting and type the whole goal.
            search = self._search(answers, ctx, texts)
            if search:
                return search
            # Only a request that names an app is routed: gibberish ("asdkjh qwe zxcv") opened the launcher.
            if probability(answers["fits"].get("noul")) < GATE and answers["app_name"].get("choice") in {
                    str(i) for i in range(len(texts))}:
                return self._route(answers, size, padded, ctx)
            # The open launcher can look fitting: routing, and its wait for the chosen app, lasts until it closes.
            if probability(answers["launcher"].get("noul")) < self.jev.min_presence:
                self.routed = ctx.run_id
        step = answers["next"].get("choice")
        # A screenshot can't show focus: typing holds only right after a click put it there, else the target is
        # clicked first. YouTube's unfocused search box took "type" and the query went nowhere (2026-10-18).
        if step == "type" and not next((a for a in reversed(ctx.history) if a != WAIT.brief()), "").startswith(
                ("click(", "double_click(")):
            step = "fill"
        choice = answers["text"].get("choice")
        text = texts[int(choice)] if choice in {str(i) for i in range(len(texts))} else None
        if step in ("type", "fill") and text is None:
            return _fail("the text to type isn't in the request")
        def unseen() -> Plan:
            # A page still drawing hides the target or splits the looks, so look again before failing: YouTube's
            # search box 0.38 sure with loading 0.46; the third Short 0.82 with the zooms apart, 0.86 on a later
            # screen (2026-10-18).
            if waits < MAX_WAITS:
                return Plan("Looking again for the target", [WAIT], [])
            return _fail("its target isn't clearly visible")

        actions: list[Action] = []
        state = {"goal": ctx.goal, "executed_steps": ctx.history}
        with ThreadPoolExecutor(3) as pool:
            # Alongside a fill's zooms: the check costs a type step a Jev round-trip, a fill none.
            typed = pool.submit(self._text_only, image, ctx, state, text) if step in ("type", "fill") else None
            if step in ("click", "double_click", "right_click", "fill"):
                point = box_center(answers["target"], size, padded)
                sure = probability(answers["present"].get("noul"))
                # Leaning seen is enough to look closer: YouTube's search box, boxed with confidence 0.8, was 0.69
                # sure once (0.77-0.84 on other tries, 2026-10-18); below min_presence the zooms must agree.
                if point is None or sure < GATE:
                    return unseen()
                # Two more looks at an enlarged crop: the first box can land a row off (a missed search box), but
                # one zoom also strays to a neighbouring row about 1 in 6 times (Chrome's new tab: search box vs
                # shortcuts). Keeping the zoom nearer the first box is a majority of the three looks.
                window = zoom_window(answers["target"], size, padded)
                control = FILL_CONTROL.format(json.dumps(text, ensure_ascii=False)) if step == "fill" else STEP_CONTROL
                looks = [seen for seen in pool.map(lambda _: look(self.jev, image, state, control, point, window,
                                                                  ctx.run_id, ctx.step), range(2)) if seen]
                if not looks:
                    return unseen()
                # Jev's doubt about which step comes next also lowers its sureness that the target is visible (a
                # small YouTube search box: 0.77-0.84 on 5 tries, each box on it), so agreeing looks stand in for
                # it. Both zooms moving to one spot agree too: for "open a new tab and search ..." the first box
                # was the page's search box and both zooms found the tab strip's "+" (frame 49, 20/20 on 10
                # replays), and a Discord fill boxed the user panel (confidence 0.14) while both zooms, though only
                # 0.72 sure, found the message bar (2026-10-18). A zoom that boxed elsewhere never confirms it.
                moved = [seen.center for seen in looks if not seen.contains]
                together = len(moved) == 2 and math.dist(*moved) < ZOOMS_AGREE
                if sure < self.jev.min_presence and not (together or len(looks) == 2 and not moved):
                    return unseen()
                if together:
                    point = moved[0]
                elif len(moved) == len(looks) and all(seen.sure >= self.jev.min_presence for seen in looks):
                    # Sure zooms that all moved: the one nearer the first box is a majority of the three looks.
                    point = min(moved, key=lambda p: math.dist(p, point))
                actions.append(Action("click" if step == "fill" else step, x=point[0], y=point[1]))
            text = typed and typed.result()
        if step in ("type", "fill"):
            replace = probability(answers["replace"].get("noul")) >= GATE
            if replace:
                # Select-all only in a filled single-line box: in a document it would replace everything.
                actions.append(Action("hotkey", keys=("GUI" if ctx.profile.os in ("macos", "ios") else "CTRL", "A")))
            actions.append(Action("type_text", text=text))
            if replace:  # the box now holds only the text: Delete drops an inline completion ("cats" -> "cats videos")
                actions.append(Action("hotkey", keys=("DELETE",)))
        if step == "enter":
            actions.append(Action("hotkey", keys=("ENTER",)))
        if step in ("scroll_down", "scroll_up"):
            actions.append(Action("scroll", dy=5 if step == "scroll_down" else -5))
        if step == "done":
            # Jev's own choice, under the gate: a played YouTube Short scored 0.60, a message left in Discord's "+"
            # menu 0.61 (2026-10-18). Neither "done" nor "fail" is true; the user can tell.
            return Plan("Jev can't confirm the request is finished",
                        [Action("ask_user", question="That looks finished, but I can't confirm it. Is it?")], [])
        if not actions:
            return _fail("it needs reasoning or content Jev can't supply")
        return Plan(f"Jev next step: {step}", actions, [])

    def _text_only(self, image: bytes, ctx: StepContext, state: dict, text: str) -> str:
        """`text` without the leading words Jev reads as the instruction ("search for", "that")."""
        words = list(re.finditer(r"\S+", text))
        if len(words) < 2:
            return text
        rests = [text[w.start():] for w in words]
        answers, _, _ = self.jev.evaluate(image, {
            f"rest{i}": {"type": "noul", "instructions": TEXT_EXACTLY.format(json.dumps(rest, ensure_ascii=False))}
            for i, rest in enumerate(rests)}, run_id=ctx.run_id, step=ctx.step, state=state, label="text")
        # The longest of the best: ties keep words.
        return max(rests, key=lambda rest: probability(answers[f"rest{rests.index(rest)}"].get("noul")))

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
        return _fail("no app suited to the request is visible")

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
    """Each run of whole words within one line of the goal, without wrapping quotes or trailing punctuation,
    shortest first. Whole runs bind text better than separate first/last words (HANDOFF.md). A line of several
    words is never one whole: it holds the request's command too. Offered whole, it was typed whole ("Go to
    Discord and spam Peter 10 times" into the chat, "Text swamp tech" into the launcher, 2026-10-18)."""
    runs = []
    for line in goal.splitlines():
        line = line.removeprefix(JUST_DID).removeprefix(CORRECTION)  # app.py's labels: the request follows
        words = list(re.finditer(r"\S+", line))
        runs += [line[words[a].start():words[a + n].end()].strip("\"'“”‘’,;.")
                 for n in range(len(words) - (len(words) > 1)) for a in range(len(words) - n)]
    # shortcut: a long goal loses its longest runs past the budget; split the goal if long literals get cut.
    return [run for run in dict.fromkeys(runs) if run][:MAX_SPANS]


def _fail(why: str) -> Plan:
    return Plan(FAILED, [Action(
        "fail", reason=f"Jev alone can't take the next step: {why}. "
                       "The language-model planner (EXECUTOR=race) can.")], [])
