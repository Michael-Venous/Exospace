"""Jev's own next step, with no language model (ported from origin/jev-only, nrebolloso).

A request that needs a plan runs here one step per screen: one batched Jev call judges
completion, picks the next action and its target, and picks the text to type from runs of
the goal's own words (Jev selects; it never writes text). A step Jev can't take, or one with
no visible effect, fails; race's planner (brain/agent/jev_first.py) then hands the request
to Flash.

Each request first finds its app the way a person would: the app in front if Jev sees it can do
the request, else a visible window, tab or icon of one, else the OS launcher's list. Nothing
names an app or a site; Jev picks from what the screen shows.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor

from brain.agent.executor import Plan, StepContext
from brain.agent.jev import COMPLETION_QUESTION, JevVision, probability
from brain.agent.triage import GATE, box_center, box_contains, refine, spans
from brain.skills import OPEN_APP_KEYS
from brain.types import Action

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
    "stuck": "The next step needs reasoning, writing new text, information that isn't visible, a key other "
             "than Enter, or the goal is unclear.",
}
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
QUESTION = ("Does the `goal` ask for information, an answer or an explanation, rather than for something to be "
            "done on the device?")
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
FAILED = "Jev can't take the next step"


class JevOnlyPlanner:
    def __init__(self, jev: JevVision) -> None:
        self.jev = jev
        self.routed: str | None = None  # the run whose app Jev has seen can do its goal

    def acknowledge(self, executed: bool) -> None:
        pass  # no plan cursor: every step is chosen afresh from the screen and the history

    def next_actions(self, image: bytes, ctx: StepContext) -> Plan:
        texts = spans(ctx.goal, MAX_SPANS)
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
            "question": {"type": "noul", "instructions": QUESTION},
            "next": {"type": "choice", "instructions": NEXT_QUESTION, "criteria": NEXT_CRITERIA},
            "present": {"type": "noul", "instructions":
                        "Is the one control that the next step of the `goal` acts on visible in `screen`, "
                        "with no ambiguity?"},
            "target": {"type": "box", "instructions":
                       "The single visible control in `screen` that the next step of the `goal` acts on."},
            "text": {"type": "choice", "instructions": TEXT_QUESTION,
                     "criteria": {"none": "No text to type."} | options},
            "replace": {"type": "noul", "instructions": REPLACE_QUESTION},
        }, run_id=ctx.run_id, step=ctx.step, state={"goal": ctx.goal, "executed_steps": ctx.history})
        # Jev's own next step is Enter right after typing: the text isn't submitted, whatever the completion score.
        unsent = bool(ctx.history) and ctx.history[-1].startswith("type_text") and answers["next"].get(
            "choice") == "enter"
        if probability(answers["question"].get("noul")) >= GATE:
            # Jev called "what is this page about?" complete and said "Done." with no answer.
            return _fail("the request asks for an answer, not an action")
        if probability(answers["complete"].get("noul")) >= COMPLETION_GATE and not unsent:
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
            self.routed = ctx.run_id
        step = answers["next"].get("choice")
        actions: list[Action] = []
        if step in ("click", "double_click", "right_click", "fill"):
            point = box_center(answers["target"], size, padded)
            sure = probability(answers["present"].get("noul"))
            if point is None or sure < NEAR_PRESENCE:
                return _fail("its target isn't clearly visible")
            # Two more looks at an enlarged crop: the first box can land a row off (a missed search box), but
            # one zoom also strays to a neighbouring row about 1 in 6 times (Chrome's new tab: search box vs
            # shortcuts). Keeping the zoom nearer the first box is a majority of the three looks.
            state = {"goal": ctx.goal, "executed_steps": ctx.history}
            with ThreadPoolExecutor(2) as pool:
                looks = [p for p in pool.map(lambda _: refine(self.jev, image, ctx.goal, point, state=state,
                                                               control=STEP_CONTROL, run_id=ctx.run_id,
                                                               step=ctx.step), range(2)) if p]
            # Jev's doubt about which step comes next also lowers its sureness that the target is visible (a
            # small YouTube search box: 0.77-0.84 on 5 tries, each box on it), so agreeing looks stand in for it.
            # Both zooms moving to one spot agree too: for "open a new tab and search ..." the first box was the
            # page's search box and both zooms found the tab strip's "+" (frame 49, 20/20 on 10 replays).
            agree = len(looks) == 2 and (all(box_contains(answers["target"], size, padded, p) for p in looks)
                                         or math.dist(*looks) < ZOOMS_AGREE)
            if not looks or sure < self.jev.min_presence and not agree:
                return _fail("its target isn't clearly visible")
            point = min(looks, key=lambda p: math.dist(p, point))
            actions.append(Action("click" if step == "fill" else step, x=point[0], y=point[1]))
        if step in ("type", "fill"):
            choice = answers["text"].get("choice")
            if choice not in {str(i) for i in range(len(texts))}:
                return _fail("the text to type isn't in the request")
            if probability(answers["replace"].get("noul")) >= GATE:
                # Select-all only in a filled single-line box: in a document it would replace everything.
                actions.append(Action("hotkey", keys=("GUI" if ctx.profile.os in ("macos", "ios") else "CTRL", "A")))
            actions.append(Action("type_text", text=texts[int(choice)]))
        if step == "enter":
            actions.append(Action("hotkey", keys=("ENTER",)))
        if step in ("scroll_down", "scroll_up"):
            actions.append(Action("scroll", dy=5 if step == "scroll_down" else -5))
        if not actions:
            return _fail("it needs reasoning or content Jev can't supply")
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
        return _fail("no app suited to the request is visible")


def _fail(why: str) -> Plan:
    return Plan(FAILED, [Action("fail", reason=f"Jev alone can't take the next step: {why}.")], [])
