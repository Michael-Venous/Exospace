"""Jev's own next step, with no language model (ported from origin/jev-only, nrebolloso).

A request that needs a plan runs here one step per screen: one batched Jev call judges
completion, picks the next action and its target, and picks the text to type from runs of
the goal's own words (Jev selects; it never writes text). A step Jev can't take, or one with
no visible effect, fails; race's planner (brain/agent/jev_first.py) then hands the request
to Flash.

Each request first finds its app the way a person would: the app in front if Jev sees it can take
the next step, else a visible window, tab or icon of one, else the one the request names from the
OS launcher's search. A website that names no browser opens a web browser. Jev sees one line of the
request at a time (words said later join as new lines), and each line finds its app again
("... and open a terminal"). Nothing names an app or a site; Jev picks from what the screen shows
and the request's words.
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from brain.agent.executor import Plan, StepContext
from brain.agent.jev import COMPLETION_QUESTION, IRREVERSIBLE_KINDS, JevVision, probability
from brain.agent.triage import GATE, ZOOM_WINDOW, box_center, box_contains, box_size, mark, refine, spans
from brain.skills import OPEN_APP_KEYS
from brain.types import Action

NEXT_QUESTION = ("The user's `goal` is being carried out on the device in `screen`; `executed_steps` were already "
                 "done. Which single action comes next?")
NEXT_CRITERIA = {
    "click": "Click or tap one visible control: a button, link, shortcut, tab, menu or list item, or an input "
             "field with nothing to type into it.",
    "fill": "Click an input field and type into it text that the goal gives word for word.",
    "double_click": "Open an icon on a computer desktop, or a file or folder in a desktop file manager.",
    "right_click": "Open the context menu of a visible item.",
    "type": "Type text that the goal gives word for word into the input field that is focused right now.",
    "enter": "Press Enter to submit or confirm the text that was just typed.",
    "address": "Go to a website or web page that the goal names by its name or address, by typing it into the "
               "web browser's address bar, when no visible link or shortcut leads there.",
    "new_tab": "Open a new tab in the web browser in front.",
    "close_tab": "Close the tab in front, in a web browser or a tabbed app.",
    "close_app": "Close or quit the app in front.",
    "erase": "Erase text that an earlier step typed.",
    "scroll_down": "Scroll down to reveal something the goal needs that is below the visible area.",
    "scroll_up": "Scroll up to reveal something the goal needs that is above the visible area.",
    "stuck": "The next step needs reasoning, writing new text, information that isn't visible, a key not "
             "named above, or the goal is unclear.",
}
TEXT_QUESTION = ("Which text does the next step type, or erase? It must be copied word for word from the `goal`, without the "
                 "command words or field name around it. Choose none if the text must be computed, looked up, "
                 "translated or written rather than copied.")
REPLACE_QUESTION = ("Is the input the next step types into a single-line box, such as a search, address or form "
                    "field, that already holds text?")
# Asked of the goal, not the next step: with a click on YouTube's search box next, "does the next step type text
# that the goal asks to search for" scored "search for horses" 0.24-0.50 (and 0.35 with a type next); this 0.87-0.95,
# typing into a cell or a document 0.01-0.33 (3 tries each).
SUBMIT_QUESTION = "Does the `goal` ask to search for words it gives, so Enter must run the search once they are typed?"
# Text typed into a spreadsheet cell is entered with Enter, which took a look of its own (~1.1 s, v11-t3): cells
# scored 0.87-0.95, a document, search box, chat box, clean terminal or address bar 0.10-0.21, and a terminal in
# front of a spreadsheet 0.51-0.57 (3 tries each), so only a sure answer adds it.
COMMIT = ("Once the text is typed into the input the next step of the `goal` types into, does it still need Enter to "
          "be entered there, as in a spreadsheet cell, without sending a message, running a command or submitting "
          "a form?")
# "youtube" typed in Brave's address bar searched Brave Search, a step and a page load from YouTube (v5-t1). Ctrl+Enter
# adds www. and .com: site names scored 0.82-0.97; "email", "my bank" and "horses" 0.05-0.21 (3 tries each).
# shortcut: a site off .com reached by its bare name (bbc.co.uk) goes through the search engine; ask for its domain
# if those matter.
DOTCOM = "Does the `goal` name a website by the word that comes before .com in its web address, such as youtube for youtube.com?"
DOTCOM_OS = ("windows", "linux", "chromeos")  # a Mac's browsers differ in the key for it
# A click on a box the line then types into is one step with the typing: Jev chose a click, not "fill", on YouTube's
# search box (6/6) and a type took another look (~1.5 s on the Pi). Inputs scored 0.85-0.98 on 6 frames (3 tries
# each); Google Docs' blank-document tile 0.73, which typing right after its click would miss.
INPUT = ("Is the one control that the next step of the `goal` acts on an input that takes typed text, such as a text "
         "field, search box, spreadsheet cell or open document page? One that creates or opens a document isn't.")
# Asked after the line typed: Brave still loading YouTube showed its new tab page, and "horses" was typed into
# nothing, Enter on it searched nothing, and YouTube's home page with horse videos passed the line (left 0.17, v13-t1).
# On 14 frames (4 tries each) text where it was typed scored 0.57-1.00, text not typed 0.00-0.06, a "Draft horses"
# chip on that home page included.
TYPED = ("Does `screen` show the text {text} where it was typed, such as in an input field, a cell, a document or a "
         "search query, rather than only in content the page already showed?")
MISSING = 0.2  # below this the text typed isn't there: it is typed again, so only a clear miss counts
# Jev chose "type" on Google Docs' home page, before any document was open (0.34 here; 0.85 in a blank document).
FOCUSED = ("Would text typed right now go into the input field, cell or document where the next step of the `goal` "
           "types it?")
# The routing questions ask about the next step: with "open discord, close discord, and open a terminal" the
# whole goal fits no single app.
FITS_QUESTION = ("Can the next step of the `goal`, after the `executed_steps`, be carried out in the app or web page "
                 "in front in `screen`, without switching to or opening another app?")
APP_PRESENT = ("Is a window, tab or app icon of an app suited to the next step of the `goal` visible in `screen`, "
               "such as in a taskbar, dock, desktop or open app launcher? For a website, a web browser is suited.")
APP_TARGET = ("The visible window, tab or app icon in `screen` of the app best suited to the next step of the "
              "`goal`; for a website, a web browser.")
# A panel's icons sit side by side: "open a terminal" boxed VS Code's icon beside the terminal's (3 steps, v7-t2),
# and one zoomed look at that box chose the terminal's (12/12 on 2 frames).
APP_CONTROL = "the window, tab or app icon of the app best suited to the next step of the `goal`; for a website, a web browser"
# "as an app launcher would list it": without it Jev chose "open a terminal" (3/3), with it "terminal".
APP_NAME = ("Which words of the `goal` are the name of the app, or the kind of app, that its next step opens or uses, "
            "as an app launcher would list it, without the command words or articles around it?")
SITE = ("Does the next step of the `goal` go to a website or web service, rather than open or use an app "
        "installed on the device?")
# Asked apart from SITE: "open up chrome, ..., make a new tab" with Chrome closed scored 0.76-0.81 as a website.
BROWSER_NAMED = "Does the `goal` name a specific web browser by its name?"
# The kind of app a website opens in, typed into the launcher's search: a site's own name found nothing there
# ("youtube" on Cinnamon), and the panel's browser icon went unrecognised (0.28-0.39).
BROWSER = "web browser"
LOADING = ("Is an app or page in `screen` still starting or loading, so the next step must wait: a start-up or "
           "splash screen (a logo, tip or progress on a plain background), a blank or partly drawn window, or grey "
           "placeholder shapes where content will appear?")
# Asked right after a launch: Brave's restored YouTube Short drew its search box as a blank shape (0.90-0.95 on
# 3 frames, 3 tries each; drawn pages 0.07-0.38 on 23), and Jev typed the search into the address bar, which
# LOADING (0.04) let through. The box was drawn 0.6 s later (v4-t1, 2026-10-19).
DRAWING = ("Are any controls in `screen`, such as a search box or round buttons, drawn as blank grey or dark shapes, "
           "without the text or icons they will show once the page has loaded?")
# An app's box this big is the window in front: "click the third video" while YouTube's results loaded fitted the
# page 0.44 and boxed the browser [0.01-0.99 x 0.04-0.98], and a switch clicked the middle of the page (v6-t1).
FRONT = 0.9
WINDOW = "Is an app window open in `screen`, rather than only the desktop or home screen?"
LAUNCHER_OPEN = "Is an app launcher or start menu open in `screen`, with a search field for typing an app's name?"
QUESTION = ("Does the `goal` ask for information, an answer or an explanation, rather than for something to be "
            "done on the device?")
COMPOSED = ("Does what remains of the `goal`, after the `executed_steps`, need text, numbers or an expression "
            "that the `goal` doesn't contain word for word, to be computed, translated or composed?")
# Zoomed in on the right spot, Jev saw this wording 0.57-0.90 sure on 7 targets (3 tries each); "the one control
# that the next step acts on" 0.44-0.82, and below 0.5 for a spreadsheet cell and LibreOffice's Calc button.
# "add hello to b1" on LibreOffice's Start Center: the step's control was clear 0.24-0.52 and boxed its recent
# documents half the time; the control that creates or opens what the step works in, 0.88-0.92 on Calc's entry.
OPENER = "When what it works in isn't open yet, that control is the one that creates or opens it."
STEP_CONTROL = "the spot that the next step of the `goal`, after the `executed_steps`, clicks"
# A box this big holds the spot it was drawn on without a closer look, and a zoom loses where it sits among
# its neighbours: on "click the third video" zooms chose the second or fourth of YouTube's Shorts row.
BIG_TARGET = 0.1  # of the screen's width and height
# A zoom shows the whole target with this margin: one cutting YouTube's search box to its last letter left Jev
# 0.26 sure it was there (4/4), one holding all of it 0.75.
ZOOM_MARGIN = 1.15
NEXT_IRREVERSIBLE = "Would the next step of the `goal`, after the `executed_steps`, " + IRREVERSIBLE_KINDS
WAIT = Action("skill", name="wait", args={"s": 3.0})  # the app or its results still appearing
LOOK_AGAIN = Action("skill", name="wait", args={"s": 1.0})  # a page still drawing; a drawn one waited 10 s
MAX_WAITS = 2  # looks again before a target still not visible hands the request to Flash
# Waits in a row while Jev sees the app or page still opening: Discord's cold start showed its splash for 4 s
# (LOADING 0.90 on it, 0.98 on YouTube's placeholder page, 0.02-0.13 on drawn pages and the desktop).
MAX_LOADING_WAITS = 6
TAB_KEYS = {"new_tab": "T", "close_tab": "W"}  # with Ctrl, or Cmd on Apple devices
# A line is judged with the line before it in view, asked what is left: on 23 frames from the four laptop tasks
# (3 tries each) finished lines scored 0.30 at most and unfinished ones 0.66 at least. The whole-goal question
# scored finished lines "close out the tab" 0.31 and "delete hello" on the emptied document 0.24; asked whether
# the last line was done, "delete hello" still scored 0.32.
LINE_LEFT = ("The `goal` lists requests one per line, and those before its last line are already done. Is anything "
             "still left to do for the request on its last line, after the `executed_steps`, as `screen` shows?")
# After a click the screen it led to can't tell what the click did: a Short playing left "click the third video"
# 0.57-0.77 left on every step, and Jev scrolled for another. The screen it was planned on, the click marked on it,
# can: on 12 frames (3 tries each) clicks that finished their line left 0.16-0.36, others 0.87-0.95.
LINE_LEFT_CLICKED = ("The `goal` lists requests one per line, and those before its last line are already done. "
                     "`screen` is the device just before the last of the `executed_steps`, a click on the red circle. "
                     "Once that click is done, is anything still left to do for the request on its last line?")
CLICKS = ("click", "double_click", "right_click")
# The agent joins words said mid-run on as "User correction: ..." lines (loop.py, app.py). Jev read "add hello to b1"
# so marked as a fix to "open up libre office" and waited on its Start Center (3/3); unmarked it clicked Calc (2/2).
CORRECTION = "User correction: "
MAX_SPANS = 250  # Jev's choice budget (bench/interpreter_screen.py)
FAILED = "Jev can't take the next step"


class JevOnlyPlanner:
    def __init__(self, jev: JevVision, approved_run: Callable[[StepContext], bool] | None = None) -> None:
        self.jev = jev
        self.approved_run = approved_run
        self.run_id: str | None = None
        self.line = 0  # the goal's first line not yet done: the one Jev sees
        self.since = 0  # where that line's steps start in the run's history
        self.prev = 0  # where the line before it started: Jev sees its steps too ("delete hello" after typing it)
        self.routed = False  # Jev has seen the app in front can take the line's next step
        self.seen = False  # Jev planned the line's next step at an earlier step: it wasn't done then
        self.finished = ""  # the last goal Jev saw done: a request continuing it starts at its new lines
        self.goal = ""  # the goal of the last step: a run whose goal goes on from it continues that request
        self.opened: str | None = None  # the app the request last opened, also in an earlier run it continues
        self.clicked: tuple[bytes, tuple[float, float], str] | None = None  # the last plan's click: screen, point, step

    def acknowledge(self, executed: bool) -> None:
        pass  # no plan cursor: every step is chosen afresh from the screen and the history

    def next_actions(self, image: bytes, ctx: StepContext) -> Plan:
        if self.run_id != ctx.run_id:
            continued = bool(self.finished) and ctx.goal.startswith(self.finished + "\n")
            self.run_id, self.line, self.since = ctx.run_id, self.finished.count("\n") + 1 if continued else 0, 0
            self.prev, self.routed, self.clicked, self.seen = 0, False, None, False
            if not ctx.goal.startswith(self.goal + "\n"):
                self.opened = None  # a new request: what an earlier one opened isn't its app
        # A run that failed starts again with the words said since joined on, its history empty: "make a new tab"
        # after closing Chrome's last tab found no app to reopen (v11-t4).
        self.goal, self.opened = ctx.goal, _last_opened(ctx.history) or self.opened
        lines = [line.removeprefix(CORRECTION) for line in ctx.goal.splitlines()]
        clicked, self.clicked = self.clicked, None
        if clicked and (ctx.history[-1:] != [clicked[2]] or "unchanged" in ctx.note):
            clicked = None  # not the step just taken, or one with no visible effect: nothing it did to judge
        guard = self.approved_run is not None and not self.approved_run(ctx)
        opened = _just_opened(ctx.history)
        # One line at a time: "click the first email" as the last of 8 lines boxed Gmail's tab row (0.78 sure,
        # 3/3), and "close discord" with "open a terminal" after it was routed to the terminal (2/2).
        # shortcut: a correction said mid-line waits for that line to finish; show Jev the later lines too if
        # corrections need to cut in.
        # The next line is asked alongside, so its first step needs no look of its own once this line is done.
        before = lines[self.line - 1] if self.line else ""
        asks = [(lines[self.line], ctx.history[self.since:], _seen(ctx.history, before, self.prev, self.since),
                 not self.routed)]
        if self.line + 1 < len(lines):
            asks.append((lines[self.line + 1], [], _seen(ctx.history, lines[self.line], self.since), True))
        typed = _typed_into_field(ctx.history[self.since:])
        with ThreadPoolExecutor(len(asks) + 3) as pool:
            done = [pool.submit(self._line_done, image, ctx, lines)]
            if clicked:
                done.append(pool.submit(self._line_done, mark(*clicked[:2]), ctx, lines, LINE_LEFT_CLICKED))
            shown = pool.submit(self._shown, image, ctx, lines[self.line], typed) if typed else None
            answered = list(pool.map(lambda ask: self._ask(image, ctx, *ask, guard, opened), asks))
            missing = shown is not None and shown.result() < MISSING
            finished = not missing and _finished(answered[0][0], ctx.history[self.since:],
                                                 max(d.result() for d in done), self.seen)
        if finished and len(answered) > 1:
            self.line, self.prev, self.since, self.routed = self.line + 1, self.since, len(ctx.history), False
            answered, finished = answered[1:], False  # the next line's first step
        self.seen = True
        answers, size, padded = answered[0]
        line, steps = lines[self.line], ctx.history[self.since:]
        seen = _seen(ctx.history, lines[self.line - 1] if self.line else "", self.prev, self.since)
        texts = spans(line, MAX_SPANS)
        named = {str(i) for i in range(len(texts))}
        routing = "fits" in answers
        if probability(answers["question"].get("noul")) >= GATE:
            # Jev called "what is this page about?" complete and said "Done." with no answer.
            return _fail("the request asks for an answer, not an action")
        if probability(answers["composed"].get("noul")) >= GATE and answers["next"].get("choice") in ("type", "fill"):
            # Only a step that types needs the text: "open camera" on a phone (a tap) was refused here, and so was
            # "go to google docs" (0.55) when the address bar takes a site's name as the request gives it.
            # "work out 12 times 9": Jev pressed keypad keys one per step and missed one (41.7 s; Flash typed
            # "12*9=" in one step, 12.2 s).
            return _fail("the rest needs text the request doesn't give word for word")
        if finished:
            return self._next_line(image, ctx, lines, "Jev verified the completed goal")
        if "unchanged" in ctx.note and steps:
            return _fail("the last step had no visible effect")  # Jev would choose it again
        # Only when nothing can be done yet: YouTube's placeholder page already has a search box to click. An app
        # just opened is waited for before its first step: its own controls may not be there yet.
        drawing = opened and probability(answers["drawing"].get("noul")) >= GATE
        loading = drawing or probability(answers["loading"].get("noul")) >= GATE
        wait = _look_again(ctx, MAX_LOADING_WAITS, anytime=True) if loading else None
        if opened and wait:
            return wait
        if routing:
            # A website that names no browser is no app to search the launcher for: "youtube" found nothing there.
            site = probability(answers["site"].get("noul")) >= GATE \
                and probability(answers["browser_named"].get("noul")) < GATE
            choice = answers["app_name"].get("choice")
            name = BROWSER if site else texts[int(choice)] if choice in named else None
            # "make a new tab" after closing Chrome's last tab: a line that names no app, with no app window open,
            # goes on in the app the request last opened, by name, since a visible app suited to the line can be
            # another browser.
            closed = probability(answers["window"].get("noul")) < GATE
            again = self.opened if closed and not name else None
            # The open_app skill knows each launcher's keys (a phone's drawer needs Ctrl+F before typing). Once per
            # app in a clause: one line can name two ("open up discord, close discord, and open a terminal").
            launch = Action("skill", name="open_app", args={"name": name or again})
            if ctx.profile.input == "touch" and name and launch.brief() not in steps:
                return Plan("Open the app", [launch], [])  # the demo phone ignores taps; its keyboard launcher works
            # Before the fits check: Jev can judge the open launcher itself as fitting and type the whole goal.
            search = self._search(answers, ctx, name, steps)
            if search:
                return search
            fits = probability(answers["fits"].get("noul")) >= GATE
            if not fits and again and launch.brief() not in steps:
                return wait or Plan("Open the app again", [launch], [])
            # Only a request that names an app or a site is routed: gibberish ("asdkjh qwe zxcv") opened the launcher.
            if not fits and name:
                # An app still opening is the one just launched: clicking its panel icon minimised it.
                route = wait or self._route(image, line, answers, size, padded, ctx, launch, steps)
                if route:
                    return route
            else:
                self.routed = True
        step = answers["next"].get("choice")
        choice = answers["text"].get("choice")
        text = texts[int(choice)] if choice in named else None
        if step == "erase" and text is None:
            text = _last_typed(ctx.history[self.prev:])  # "delete hello" right after typing it: Jev picked no words (2/2)
        if missing:
            step, text = "fill", typed  # into the field this time, clicked first
        mod = "GUI" if ctx.profile.os in ("macos", "ios") else "CTRL"
        actions: list[Action] = []
        if step == "address" and (text is None or _is_the_request(text, line)):
            # "go to email" on Chrome's new tab: Jev chose the address bar with nothing to type (6/6), and boxed
            # the page's Gmail link as the step's control.
            step = "click"
        if step == "type" and not (ctx.history and ctx.history[-1].startswith("click(")):
            # Focus doesn't show in a screenshot: a "type" not right after a click went nowhere, so the field
            # is clicked first (nrebolloso, origin/live-speech b219465).
            step = "fill"
        elif step == "type" and probability(answers["focused"].get("noul")) < GATE:
            step = "click"  # what was clicked opened no place to type: open the document or field first
        if step == "click" and text is not None and not _is_the_request(text, line) \
                and probability(answers["input"].get("noul")) >= self.jev.min_presence:
            step = "fill"
        if step in ("click", "double_click", "right_click", "fill"):
            point = box_center(answers["target"], size, padded)
            sure = probability(answers["present"].get("noul"))
            if point is None or sure < GATE:
                return wait or _look_again(ctx) or _fail("its target isn't clearly visible")
            width, height = box_size(answers["target"], size, padded)
            if min(width, height) < BIG_TARGET:
                # Two more looks at an enlarged crop: the first box can land a row off (a missed search box), but
                # one zoom also strays to a neighbouring row about 1 in 6 times (Chrome's new tab: search box vs
                # shortcuts). Keeping the zoom nearer the first box is a majority of the three looks. No tighter
                # look after it: on cell B1 it lost the row and column headers and chose the selected A1 (5 of 8).
                # typing that went nowhere left out: told the text was typed, both zooms doubted the search box (0.41-0.48)
                state = {"goal": line, "executed_steps": seen[:self.since - self.prev] if missing else seen}
                # the crop's side is a fraction of the screen's longer side, and its height 0.66 of its width
                window = max(ZOOM_WINDOW, ZOOM_MARGIN * max(width * size[0], height * size[1] / 0.66) / max(size))
                with ThreadPoolExecutor(2) as pool:
                    looks = [p for p in pool.map(lambda _: refine(self.jev, image, line, point, state=state,
                                                                   control=STEP_CONTROL, run_id=ctx.run_id,
                                                                   step=ctx.step, fine=False, window=window),
                                                   range(2)) if p]
                # Jev's doubt about which step comes next also lowers its sureness that the target is visible (a
                # small YouTube search box: 0.77-0.84 on 5 tries, each box on it; a spreadsheet cell 0.79-0.81), so
                # a zoom that lands in its box stands in for it: on cell B1 one zoom in 3 drifted to the selected
                # A1 beside it. Zooms elsewhere don't: both left Chrome's Gmail link for the address bar.
                if sure < self.jev.min_presence:
                    looks = [p for p in looks if box_contains(answers["target"], size, padded, p)]
                if not looks:
                    return wait or _look_again(ctx) or _fail("its target isn't clearly visible")
                point = min(looks, key=lambda p: math.dist(p, point))
            actions.append(Action("click" if step == "fill" else step, x=point[0], y=point[1]))
        if step in ("type", "fill", "address", "erase"):
            if text is None:
                return _fail("the text to type isn't in the request")
            if _is_the_request(text, line):
                # "write a short note about my day" was typed as the note (Diego, agent/jev-only-fixes).
                return _fail("the request asks for text to be written, not typed word for word")
            if step == "erase":
                # shortcut: one word-erase per word assumes the caret ends the typed text and Ctrl+Backspace
                # takes a word with its punctuation; select the text instead if punctuated text is left behind.
                word = ("ALT" if ctx.profile.os in ("macos", "ios") else "CTRL", "BACKSPACE")
                actions += [Action("hotkey", keys=word) for _ in _words(text)]
            else:
                if step == "address":
                    actions.append(Action("hotkey", keys=(mod, "L")))  # focuses the address bar, its text selected
                elif probability(answers["replace"].get("noul")) >= GATE:
                    # Select-all only in a filled single-line box: in a document it would replace everything.
                    actions.append(Action("hotkey", keys=(mod, "A")))
                actions.append(Action("type_text", text=text))
                # Enter in the same batch: a separate step for it cost one more Jev call (0.5-0.7 s).
                if step == "address" and ctx.profile.os in DOTCOM_OS and not re.search(r"\s", text) \
                        and probability(answers["dotcom"].get("noul")) >= GATE:
                    actions.append(Action("hotkey", keys=("CTRL", "ENTER")))
                elif step == "address" or probability(answers["submit"].get("noul")) >= GATE \
                        or probability(answers["commit"].get("noul")) >= self.jev.min_presence:
                    actions.append(Action("hotkey", keys=("ENTER",)))
        if step in TAB_KEYS:
            actions.append(Action("hotkey", keys=(mod, TAB_KEYS[step])))
        if step == "close_app":
            # shortcut: phones have no quit key, so Alt+F4 leaves the screen unchanged and the step fails; close
            # from the app switcher if phones need it.
            actions.append(Action("hotkey", keys=("GUI", "Q") if ctx.profile.os == "macos" else ("ALT", "F4")))
        if step == "enter":
            actions.append(Action("hotkey", keys=("ENTER",)))
        if step in ("scroll_down", "scroll_up"):
            if ctx.profile.input == "mouse":  # a Mac scrolls what is under the pointer: a toolbar scrolled nothing
                actions.append(Action("skill", name="hover", args={"x": 0.5, "y": 0.5}))
            actions.append(Action("scroll", dy=5 if step == "scroll_down" else -5))
        if not actions:
            # Discord's update splash has nothing to act on for 4 s after its icon is clicked. A finished request
            # of three clauses scored 0.63-0.65 complete, with nothing left to do.
            if not wait and probability(answers["complete"].get("noul")) >= GATE:
                return self._next_line(image, ctx, lines, "Jev sees nothing left to do")
            # Jev's answers vary between looks: "open up youtube" on a browser's new tab was stuck 1 time in 3.
            return wait or _look_again(ctx) or _fail("it needs reasoning or content Jev can't supply")
        if _repeats(actions, steps):
            # Clicking "1" twice typed 11 for "12 times eight", and Enter twice submitted twice: never repeat
            # input to find out whether it worked (Diego, agent/jev-only-fixes). Only the line's own: Docs' Blank
            # document tile was where the Google Docs search result just clicked had been (v11-t4).
            return _fail("it would repeat the step it just took")
        if guard and probability(answers["irreversible"].get("noul")) >= GATE:
            return Plan("Irreversible step needs approval", [], [], held=f'the next step of "{line}"')
        if step in CLICKS:
            self.clicked = image, point, actions[0].brief()
        return Plan(f"Jev next step: {step}", actions, [], blind=step in ("address", "new_tab"))

    def _ask(self, image: bytes, ctx: StepContext, line: str, steps: list[str], context: list[str], route: bool,
             guard: bool, opened: bool) -> tuple[dict, tuple[int, int], tuple[int, int]]:
        """Jev's batched questions on one line of the request: its next step, asked with the `context` of steps
        since the line before it began ("type in hello" on a just-opened blank document was a click with the line's
        own steps alone, 2/2), and with `route` the questions that find its app, asked apart with the line's own
        `steps` ("close discord" after the clicks that opened it fitted Discord in front 0.41, 0.88 alone)."""
        options = {str(i): json.dumps(text, ensure_ascii=False) for i, text in enumerate(spans(line, MAX_SPANS))}
        # Until Jev sees the app in front can take the line's next step, each step also picks the app to use.
        routing = {"fits": {"type": "noul", "instructions": FITS_QUESTION},
                   "app_present": {"type": "noul", "instructions": APP_PRESENT},
                   "app": {"type": "box", "instructions": APP_TARGET},
                   "app_name": {"type": "choice", "instructions": APP_NAME,
                                "criteria": {"none": "The goal names no app or kind of app, or only a website."}
                                | options},
                   "site": {"type": "noul", "instructions": SITE},
                   "browser_named": {"type": "noul", "instructions": BROWSER_NAMED},
                   "launcher": {"type": "noul", "instructions": LAUNCHER_OPEN},
                   "window": {"type": "noul", "instructions": WINDOW}} \
            if route and ctx.profile.os in OPEN_APP_KEYS else {}  # firmware has no apps
        irreversible = {"irreversible": {"type": "noul", "instructions": NEXT_IRREVERSIBLE}} if guard else {}
        drawing = {"drawing": {"type": "noul", "instructions": DRAWING}} if opened else {}
        stepping = irreversible | drawing | {
            "complete": {"type": "noul", "instructions": COMPLETION_QUESTION},
            "question": {"type": "noul", "instructions": QUESTION},
            "composed": {"type": "noul", "instructions": COMPOSED},
            "next": {"type": "choice", "instructions": NEXT_QUESTION, "criteria": NEXT_CRITERIA},
            "present": {"type": "noul", "instructions":
                        "Is the one control that the next step of the `goal` acts on visible in `screen`, "
                        f"with no ambiguity? {OPENER}"},
            "target": {"type": "box", "instructions":
                       f"The single visible control in `screen` that the next step of the `goal` acts on. {OPENER}"},
            "text": {"type": "choice", "instructions": TEXT_QUESTION,
                     "criteria": {"none": "No text to type."} | options},
            "replace": {"type": "noul", "instructions": REPLACE_QUESTION},
            "submit": {"type": "noul", "instructions": SUBMIT_QUESTION},
            "commit": {"type": "noul", "instructions": COMMIT},
            "input": {"type": "noul", "instructions": INPUT},
            "dotcom": {"type": "noul", "instructions": DOTCOM},
            "focused": {"type": "noul", "instructions": FOCUSED},
            "loading": {"type": "noul", "instructions": LOADING},
        }

        def ask(questions: dict, done: list[str]) -> tuple[dict, tuple[int, int], tuple[int, int]]:
            return self.jev.evaluate(image, questions, run_id=ctx.run_id, step=ctx.step,
                                     state={"goal": line, "executed_steps": done})

        with ThreadPoolExecutor(1) as pool:
            found = pool.submit(ask, routing, steps) if routing else None
            answers, size, padded = ask(stepping, context)
            return (found.result()[0] if found else {}) | answers, size, padded

    def _line_done(self, image: bytes, ctx: StepContext, lines: list[str], question: str = LINE_LEFT) -> float:
        """How sure Jev is that the line it sees is done, asked with the line before it in view."""
        shown = "\n".join(lines[max(self.line - 1, 0):self.line + 1])
        answers, _, _ = self.jev.evaluate(image, {"left": {"type": "noul", "instructions": question}},
                                          run_id=ctx.run_id, step=ctx.step, label="line-done",
                                          state={"goal": shown, "executed_steps": ctx.history[self.prev:]})
        return 1 - probability(answers["left"].get("noul"))

    def _shown(self, image: bytes, ctx: StepContext, line: str, text: str) -> float:
        """How sure Jev is that the text the line typed shows where it was typed."""
        question = TYPED.format(text=json.dumps(text.removesuffix("…"), ensure_ascii=False))
        answers, _, _ = self.jev.evaluate(image, {"typed": {"type": "noul", "instructions": question}},
                                          run_id=ctx.run_id, step=ctx.step, label="typed",
                                          state={"goal": line, "executed_steps": ctx.history[self.since:]})
        return probability(answers["typed"].get("noul"))

    def _next_line(self, image: bytes, ctx: StepContext, lines: list[str], why: str) -> Plan:
        """The line Jev sees is done: go on to the next one, or the request is done."""
        if self.line + 1 < len(lines):
            self.line, self.prev, self.since, self.routed = self.line + 1, self.since, len(ctx.history), False
            self.seen = False
            return self.next_actions(image, ctx)
        self.finished = ctx.goal
        return Plan(why, [Action("done", summary="Done.")], [])

    def _search(self, answers: dict, ctx: StepContext, name: str | None, steps: list[str]) -> Plan | None:
        """Search the launcher this clause opened for the app to open, once. Before any launcher icon is
        clicked: an icon has no name to check (a camera icon was the screenshot tool)."""
        launcher = Action("hotkey", keys=tuple(OPEN_APP_KEYS[ctx.profile.os])).brief()
        if launcher not in steps or any(a.startswith("type_text") for a in steps[steps.index(launcher):]):
            return None
        # Typed only into a launcher Jev sees, so one still opening can't take the text.
        if name and probability(answers["launcher"].get("noul")) >= self.jev.min_presence:
            return Plan("Search the launcher", [Action("type_text", text=name)], [])
        return None

    def _route(self, image: bytes, line: str, answers: dict, size, padded, ctx: StepContext, launch: Action,
               steps: list[str]) -> Plan | None:
        """Switch to the visible app best suited to the clause, else `launch` the one it names (a website: a web
        browser), else open the OS launcher to show more. None: that app is the window in front already."""
        launcher = Action("hotkey", keys=tuple(OPEN_APP_KEYS[ctx.profile.os]))
        sure = probability(answers["app_present"].get("noul")) >= self.jev.min_presence
        point = box_center(answers["app"], size, padded) if sure else None
        side = min(box_size(answers["app"], size, padded))
        if point is not None and side >= FRONT:
            return None
        # The icon just clicked isn't looked at closer: a zoom on Discord's found no icon and launched it again (v9-t2).
        if point is not None and side < BIG_TARGET and not _repeats([Action("click", x=point[0], y=point[1])],
                                                                    ctx.history):
            point = refine(self.jev, image, line, point, state={"goal": line, "executed_steps": steps},
                           control=APP_CONTROL, run_id=ctx.run_id, step=ctx.step, fine=False)
        if point is not None:
            switch = Action("click", x=point[0], y=point[1])
            if _repeats([switch], ctx.history):
                # The app the last click opened is still coming: "close discord" boxed Discord's panel icon on the
                # desktop 2 s after "open up discord" clicked it, and clicking it again would minimise it (v8-t2).
                return _look_again(ctx, MAX_LOADING_WAITS) or _fail("the app the last click opened never appeared")
            # shortcut: one click opens taskbar, dock and launcher entries but not desktop icons; double-click
            # when the app box is a desktop icon if those get missed.
            return Plan("Switch to the app for the request", [switch], [])
        if launch.brief() not in steps:
            return Plan("Open the app", [launch], [])
        if launcher.brief() not in steps:
            return Plan("Open the app launcher", [launcher], [])
        if ctx.history[-1] != WAIT.brief():  # the launcher, its results or the chosen app still appearing
            return Plan("Waiting for the app", [WAIT], [])
        return _fail("no app suited to the request is visible")


def _look_again(ctx: StepContext, limit: int = MAX_WAITS, anytime: bool = False) -> Plan | None:
    """A target not visible yet is often a page still loading: right after a launch, YouTube's search box
    wasn't there and Flash took the rest (30 s on the Pi). Wait briefly instead, up to `limit` in a row
    (nrebolloso, origin/live-speech cfb822b); on a run's first screen only when Jev sees it still `anytime`
    loading."""
    waits = 0
    for done in reversed(ctx.history):
        if done != LOOK_AGAIN.brief():
            break
        waits += 1
    return Plan("Jev waits for the page", [LOOK_AGAIN], []) if (ctx.history or anytime) and waits < limit else None


def _just_opened(history: list[str]) -> bool:
    """The last input opened an app, perhaps with waits since: what it shows may still be drawing."""
    inputs = [done for done in history if done not in (LOOK_AGAIN.brief(), WAIT.brief())]
    return bool(inputs) and inputs[-1].startswith("skill(open_app ")


def _words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _last_typed(steps: list[str]) -> str | None:
    """The text the last of these steps typed.
    shortcut: history keeps 40 characters of it, so longer text is erased only in part; keep the full text in the
    planner if dictated text that long needs erasing."""
    typed = [s for s in steps if s.startswith("type_text(")]
    return ast.literal_eval(typed[-1][len("type_text("):-1]) if typed else None


def _seen(history: list[str], before: str, start: int, since: int | None = None) -> list[str]:
    """The steps Jev sees with a line: from `start` to `since` those of the line `before` it, each marked with it.
    Unmarked, the click on Brave's YouTube tile that opened YouTube read as one on its search box: "search for
    horses" was typed, or YouTube's Horses chip clicked, without a click on the box (4/4, v14-t1)."""
    since = len(history) if since is None else since
    return [f"{s} for {json.dumps(before, ensure_ascii=False)}" for s in history[start:since]] + history[since:]


def _typed_into_field(steps: list[str]) -> str | None:
    """The text these steps' last input typed, Enter after it or not. None after other input, or for text typed into
    the address bar: the page it opens replaces it."""
    inputs = [s for s in steps if s not in (LOOK_AGAIN.brief(), WAIT.brief())]
    while inputs and _ENTER.fullmatch(inputs[-1]):
        inputs.pop()
    if inputs[-2:-1] in (["hotkey(CTRL+L)"], ["hotkey(GUI+L)"]):
        return None
    return _last_typed(inputs[-1:])


def _last_opened(steps: list[str]) -> str | None:
    """The name of the app the last open_app step among these opened."""
    opened = [s for s in steps if s.startswith("skill(open_app ")]
    return ast.literal_eval(opened[-1][len("skill(open_app "):-1])["name"] if opened else None


def _finished(answers: dict, steps: list[str], done: float, seen: bool) -> bool:
    """Jev verified the line done. Its own next step is Enter right after typing: the text isn't submitted,
    whatever the line-done score (0.94 for "add hello to b1" with hello typed into B1, not entered)."""
    unsent = bool(steps) and steps[-1].startswith("type_text") and answers["next"].get("choice") == "enter"
    # A line already seen not done that took no input since isn't done by itself: "close discord" scored done
    # between Discord's splash and its window (v7-t2), and Alt+F4 never ran; on the desktop before Discord's
    # window showed, its one plan discarded as the screen changed (v8-t2).
    # shortcut: a line the screen finishes on its own ("wait for the video to end") can't be done; let the
    # line-done check see why the screen changed if such lines matter.
    idle = seen and all(s in (LOOK_AGAIN.brief(), WAIT.brief()) for s in steps)
    return done >= GATE and not unsent and not idle


def _is_the_request(text: str, line: str) -> bool:
    """True when the text to type is (nearly) the whole request line: an instruction, not dictated words."""
    request = _words(line)
    return len(request) >= 3 and len(_words(text)) >= len(request) - 1


_POINT = re.compile(r"^(click|double_click|right_click)\((\d+),(\d+)\)$")
_ENTER = re.compile(r"hotkey\((\w+\+)?ENTER\)")
SAME_SPOT = 20  # history coordinates are 0-1000: a click this close to the last one is the same click


def _repeats(actions: list[Action], history: list[str]) -> bool:
    """True when these actions are the same input as the last ones that ran (a click within SAME_SPOT, the
    same text or the same keys). Scrolls, waits and pointer moves may repeat."""
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


def _fail(why: str) -> Plan:
    return Plan(FAILED, [Action("fail", reason=f"Jev alone can't take the next step: {why}.")], [])
