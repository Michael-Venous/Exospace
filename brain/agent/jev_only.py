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

import cv2
import numpy as np

from brain.agent.executor import Plan, StepContext
from brain.agent.jev import COMPLETION_QUESTION, IRREVERSIBLE_KINDS, JevVision, probability
from brain.agent.triage import (APPROVED, GATE, ZOOM_WINDOW, Box, beside, box_center, box_contains, box_size, encloses, mark, refine,
                                spans, zoom)
from brain.skills import OPEN_APP_KEYS
from brain.types import Action
from brain.verification import visibly_changed

NEXT_QUESTION = ("The user's `goal` is being carried out on the device in `screen`; `executed_steps` were already "
                 "done. Which single action comes next?")
NEXT_CRITERIA = {
    "click": "Click or tap one visible control: a button, link, shortcut, tab, menu or list item, or an input "
             "field with nothing to type into it.",
    "fill": "Click an input field and type into it text that the goal gives word for word.",
    "double_click": "Open an icon on a computer desktop, or a file or folder in a desktop file manager.",
    # "copy the title" on a Wikipedia article chose right_click 0.38-0.51 to copy 0.26-0.31 (6/6), copy 0.37-0.45
    # once told (6/6); "right click the title" stayed right_click 0.70-0.74 (2026-10-19).
    "right_click": "Open the context menu of a visible item. Text to copy is copied with copy, not from this menu.",
    "type": "Type text that the goal gives word for word into the input field that is focused right now.",
    "enter": "Press Enter to submit or confirm the text that was just typed.",
    "address": "Go to a website or web page that the goal names by its name or address, by typing it into the "
               "web browser's address bar, when no visible link or shortcut leads there.",
    "new_tab": "Open a new tab in the web browser in front.",
    "close_tab": "Close the tab in front, in a web browser or a tabbed app.",
    "close_app": "Close or quit the app in front.",
    "erase": "Erase text that an earlier step typed.",
    "copy": "Copy one visible heading, line or paragraph of text that the goal names.",
    "paste": "Paste what was copied into the input field, cell or document where the goal puts it.",
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
# "open up a google doc template" typed its words in the address bar, or clicked them from its history, and searched
# the web into a third-party template site (5 runs). Template lines scored 0.64-0.65, "open a resume template" 0.94;
# "go to email", gmail, google docs, wikipedia and youtube 0.01-0.08 (3 tries each, 2026-10-19).
LOOKUP = ("Does the `goal` ask for a particular kind of page, document or template that must be looked for on the web, "
          "rather than going to a place or service it names?")
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
# A line already true when it is reached takes no step: "scroll down to the wiki pedia link" with the link in view
# scored 0.86-0.90 complete and the link was clicked, doing the next line's step. On 12 frames (3 tries each) lines
# already done scored 0.86-0.98 there, all but a click credited to the line before; lines not yet done 0.46 at most.
ALREADY = 0.7
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
# A website's line fits the browser in front, whatever page it shows: "copy the header" on a Wikipedia article fitted
# it 0.45, and "open up a google doc template" 0.10, and each opened another browser (2026-10-19). Browsers in front
# scored 0.99-1.00 on 5 frames, Discord, LibreOffice and the desktop 0.01-0.03 on 4 (3 tries each).
BROWSER_FRONT = "Is the window in front in `screen` a web browser, with tabs and an address bar?"
# The kind of app a website opens in, typed into the launcher's search: a site's own name found nothing there
# ("youtube" on Cinnamon), and the panel's browser icon went unrecognised (0.28-0.39).
BROWSER = "web browser"
LOADING = ("Is an app or page in `screen` still starting or loading, so the next step must wait: a start-up or "
           "splash screen (a logo, tip or progress on a plain background), a blank or partly drawn window, or grey "
           "placeholder shapes where content will appear?")
# A site the browser is going to shows under the page before it until it draws: Brave's new tab page under youtube.com
# was clicked for YouTube's search box, the click landed on YouTube as it drew, on nothing, and "horses" was typed
# into nothing (4.5 s, 2026-10-20). Address bar and page at odds read 0.66-0.97 on 2 such frames, LOADING 0.03-0.20;
# 31 drawn pages 0.01-0.09 (3 tries each).
PENDING = ("Does `screen` show a web browser whose address bar names a different website than the page shown below "
           "it, as when it has started going to a site and still shows the page before?")
# Asked right after a launch: Brave's restored YouTube Short drew its search box as a blank shape (0.90-0.95 on
# 3 frames, 3 tries each; drawn pages 0.07-0.38 on 23), and Jev typed the search into the address bar, which
# LOADING (0.04) let through. The box was drawn 0.6 s later (v4-t1, 2026-10-19).
# Asked of the step's own control: asked of any control, the round buttons YouTube draws last held a search box
# already drawn to two looks again (1.4 s, 2026-10-20). On 33 frames after launches, a blank search box read
# 0.44-0.91 (0.44-0.63 on frames caught mid-resize), a drawn one 0.02-0.21 beside blank buttons (3 tries each).
DRAWING = ("Is the control that the next step of the `goal` uses in `screen`, such as a search box, drawn as a blank "
           "grey or dark shape, without the text or icon it will show once the page has loaded?")
# A browser a line opened may show its new tab page: a new tab key then adds another. Chrome relaunched for "make a new
# tab" after "close out the tab" closed its window read 0.41-0.56 left there, and Ctrl+T cost 1.0 s (2026-10-20). Its
# new tab page read 0.79-0.92 so; Docs, Gmail, YouTube, Calc and restored Brave pages 0.00-0.04 (3 tries each).
FRESH = "Is the tab in front in `screen` a web browser's new tab page, with nothing opened in it yet?"
# An app's box this big is the window in front: "click the third video" while YouTube's results loaded fitted the
# page 0.44 and boxed the browser [0.01-0.99 x 0.04-0.98], and a switch clicked the middle of the page (v6-t1).
FRONT = 0.9
WINDOW = "Is an app window open in `screen`, rather than only the desktop or home screen?"
OPENED_FRONT = "Is the window in front in `screen` the app {app}, rather than another app or only the desktop?"
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
# Asked apart: on LibreOffice's Start Center "add hello to b1" read its control clear 0.41-0.77 in live runs and 0.26-0.39
# offline, failing one run, where this read 0.89-0.92 (a tooltip over a recent file or not); Calc's open sheet, YouTube's
# home and results pages 0.01-0.11. Its zooms saw Calc's entry 0.22-0.33 as the step's spot, 0.89-0.94 as its opener.
OPENS = ("Is what the next step of the `goal` works in (a document, sheet, page or window) not open yet in `screen`, "
         "while a control that creates or opens it is visible?")
OPENER_CONTROL = f"{STEP_CONTROL}: when what it works in isn't open yet, the control that creates or opens it"
COPY_CONTROL = "the text that the first line of the `goal` copies (a later line names it by what it pastes)"
# A box this big holds the spot it was drawn on without a closer look, and a zoom loses where it sits among
# its neighbours: on "click the third video" zooms chose the second or fourth of YouTube's Shorts row. Only when
# Jev is sure of it: "copy the header" boxed four rows of Wikipedia's citations 0.54 sure, and the select at its
# centre clicked a citation's link (2026-10-19).
BIG_TARGET = 0.1  # of the screen's width and height
# Nor when the box itself is doubtful: Jev's sureness can be that what the step works in isn't open yet. On
# LibreOffice's Start Center "add hello to b1" boxed the recent documents 0.06 sure, opens 0.97, and the click
# opened one of them (2026-10-20); big boxes clicked unzoomed that held their target were 0.37-0.66 (14 clicks). A
# look again on that screen boxed Calc's small entry 5 times in 6. Doubted presence still zooms: a blank document's
# body, 0.11 sure, present 0.56.
BIG_SURE = 0.3
# A zoom shows the whole target with this margin: one cutting YouTube's search box to its last letter left Jev
# 0.26 sure it was there (4/4), one holding all of it 0.75.
ZOOM_MARGIN = 1.15
# A zoom's box this many times the first look's along an axis boxed a run of controls along it: of a desktop's
# LockDown Browser and Chrome icons, one above the other, both zooms boxed the two, 2.1 times the first box's height,
# and the double click at its centre landed between them (2026-10-20). Along it the click goes where the boxes overlap.
SPAN = 1.5
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
# can: on 12 frames (3 tries each) clicks that finished their line left 0.16-0.36, others 0.87-0.95. It alone judges
# the click: the screen after it read a Wikipedia link or the [20] that cites the birth 0.42-0.58 left, and Docs'
# template gallery, which Chrome's Docs shortcut opened for "open up a google doc template", 0.47-0.54; marked, the
# first read 0.18-0.34 left and the gallery 0.57-0.72 (8 tries each, 2026-10-19).
LINE_LEFT_CLICKED = ("The `goal` lists requests one per line, and those before its last line are already done. "
                     "`screen` is the device just before the last of the `executed_steps`, a click on the red circle. "
                     "Once that click is done, is anything still left to do for the request on its last line?")
# A click's line is judged so only for a click Jev was sure of: "click the date of birth reference" clicked the lead's [b], a pronunciation
# note beside the birth name, its zooms 0.50-0.58 sure (9 runs); the marked look read it done and the copy that followed
# took a heading (2026-10-19). Footnotes that did cite the birth zoomed 0.62-0.68, Wikipedia links 0.90-0.97. The screen
# a doubtful click led to judges it: [b]'s note read 0.45-0.62 left on its own.
CLICK_SURE = 0.6
# A click held open for a surer one of its line's (below) is one zoomed under this: [b] once zoomed 0.61, its note read
# 0.40-0.50 left, the line ended there with the [20] that cites the birth boxed 0.58, and a note's text was copied
# (2026-10-19). Wikipedia links zoom 0.90-0.97.
HOLD_SURE = 0.7
# A zoom this sure of a click's target, though no surer after a look again, still places it: [b]'s zooms read 0.41-0.61
# in 10 runs, a search box under typing that went nowhere 0.41-0.48.
# shortcut: zooms on a spot that isn't the target are unmeasured; size them with zoomdone.py if doubted clicks miss.
DOUBT = 0.4
CLICKS = ("click", "double_click", "right_click")
# What a copy takes is asked with the request's later lines, which name it by what they paste: "copy the header" before
# "paste the date of birth reference in", on the citation a footnote jump marked, was boxed four rows deep alone and
# on that citation's row with them (15/15, the pointer off the page). Its presence doesn't tell it apart: 0.45-0.54
# there, 0.66-0.75 on the article's lead and Google's results, so nothing gates it: the lines before reach the screen.
COPY_PRESENT = ("Is the text that the first line of the `goal` copies visible in `screen`? A later line of the `goal` "
                "names that text by what it pastes.")
COPIED = ("The visible text in `screen` that the first line of the `goal` copies, boxed tightly around that text "
          "alone. A later line of the `goal` names that text by what it pastes.")
# A copy's triple click goes this far past the end of the text, still in its line: the box's middle was on the
# citation's link about half the time, and a click there follows it (a run pasted "Citations", 2026-10-19). Past
# the end it selected the citation 15/15, the article's title 10/10.
COPY_PAST = 0.005  # of the screen's width
# The look closer that places the copy's row: Jev's box on the citation just under Chrome's toolbar reached into the
# toolbar, and its middle row (0.09) selected nothing 6 times in 6; a zoom this wide on it placed the row 15/16, and
# 41/48 over that, its twin frame and an article's title (2026-10-19).
COPY_WINDOW = 0.25
CLIPBOARD = {"copy": "C", "paste": "V"}  # the key each clipboard step presses with Ctrl (Cmd on macOS)
# The agent joins words said mid-run on as "User correction: ..." lines (loop.py, app.py). Jev read "add hello to b1"
# so marked as a fix to "open up libre office" and waited on its Start Center (3/3); unmarked it clicked Calc (2/2).
CORRECTION = "User correction: "
# Each pause of 0.2 s ends a segment, sent as such a line: "Click on eras and ages" came as "Click on errors" and
# "ages.", and two "close this browser" runs closed it, then went on to a tail of their own, failing on it or
# opening Brave again (2026-10-20). Asked so, Jev read 7 such cut sentences one (4/4 each) and 5 of 7 requests two:
# 2 joined it "and"-led, the one the page sends whole anyway. Yet "Close this" + "browser." read two (0.8, 4/4), "Close
# this" whole on its own; told a line naming no action finishes the one before, 12/12 such cuts read one (0.89+) and
# 6/6 requests two, a correction ("no, firefox instead") one: joined it is read whole, never done twice (2026-10-20).
JOIN = ("Speech to text split the goal into two lines at a pause. Did the pause cut one sentence in two, or did the "
        "user say two separate requests? A line is a request of its own only if it says an action of its own to take; "
        "a line that names no action (only a thing, place or word) finishes the sentence on the line before, even when "
        "that line could stand alone.")
MAX_SPANS = 250  # Jev's choice budget (bench/interpreter_screen.py)
FAILED = "Jev can't take the next step"
Ask = tuple[bytes, str, list[tuple[dict, dict]]]  # an image, its trace label, and question sets with their state


class JevOnlyPlanner:
    def __init__(self, jev: JevVision, approved_run: Callable[[StepContext], bool] | None = None) -> None:
        self.jev = jev
        self.approved_run = approved_run
        self.run_id: str | None = None
        self.line = 0  # the goal's first line not yet done: the one Jev sees
        self.since = 0  # where that line's steps start in the run's history
        self.prev = 0  # where the line before it started: Jev sees its steps too ("delete hello" after typing it)
        self.routed = False  # Jev has seen the app in front can take the line's next step
        self.kind: str | None = None  # the kind of app the line names, routed to the browser as a website
        self.seen = False  # Jev planned the line's next step at an earlier step: it wasn't done then
        # History length and screen when an input changed nothing: its retry may repeat it.
        self.unmoved: tuple[int, bytes] | None = None
        # The goal's lines Jev saw done: a request continuing it starts at its next line. A failed run's too: its
        # restart with the words said since redid "close google" after "open up google again".
        self.finished = ""
        self.goal = ""  # the goal of the last step: a run whose goal goes on from it continues that request
        self.opened: str | None = None  # the app the request last opened, also in an earlier run it continues
        self.clicked: tuple[bytes, tuple[float, float], str] | None = None  # the last plan's click: screen, point, step
        self.doubted: str | None = None  # the last plan's click, as its step reads, when Jev wasn't sure of its target
        # Jev's last typing, as its step reads: Flash's went where its own focus check saw a field. Flash clicked
        # the address bar and typed "docs.new\n"; on the document it opened that text was nowhere (0.09), and it
        # was typed again into the document (2026-10-19).
        self.typed: str | None = None
        self.screen: bytes | None = None  # the last step's screen
        self.planned: tuple[int, list[str]] = (0, [])  # the history's length then, and the steps planned on it
        self.idle = False  # the last step looked again and the screen stayed as it was
        # Launches the skill refused (no launcher showed): never in the history, each was tried again, 2 s apiece
        # ("open google" three times on a desktop whose launcher key showed only the panel, 2026-10-20).
        self.refused: list[str] = []
        self.joined: set[int] = set()  # the goal's lines Jev read as the rest of the line before: each is joined on
        self.judged = 1  # the goal's lines Jev read so: the first line is no rest

    @property
    def current_line(self) -> str:
        """The line of the request Jev's last step was for."""
        lines = self._lines(self.goal)
        return lines[self.line] if self.line < len(lines) else self.goal

    def _lines(self, goal: str) -> list[str]:
        """The goal's lines, each joined on the line before it is the rest of, its own line left empty."""
        # The approval is no line to act on: read as one, "close the spreadsheet without saving" failed stuck on
        # the desktop its approved Don't Save left (2026-10-20).
        lines = ["" if line == APPROVED else line.removeprefix(CORRECTION) for line in goal.splitlines()]
        for i in sorted(self.joined):
            j = max((k for k in range(i) if lines[k]), default=0)
            lines[j], lines[i] = f"{lines[j]} {lines[i]}", ""
        return lines

    def acknowledge(self, executed: bool) -> None:
        if not executed:
            self.refused += [step for step in self.planned[1] if step.startswith("skill(open_app ")]

    def next_actions(self, image: bytes, ctx: StepContext) -> Plan:
        plan = self._plan(image, ctx)
        # A kind of app read as a website has its app to open: "open up a spreadsheet" with Google Docs in front read
        # a site (0.92), and on Chrome's new tab Jev had no words of the request to type, or no control (2026-10-20).
        launch = Action("skill", name="open_app", args={"name": self.kind})
        if plan.reasoning == FAILED and self.kind and launch.brief() not in ctx.history[self.since:] + self.refused:
            plan = Plan("Open the app the line names", [launch], [], blind=True)
        self.screen, self.planned = image, (len(ctx.history), [action.brief() for action in plan.actions])
        return plan

    def _plan(self, image: bytes, ctx: StepContext) -> Plan:
        if self.run_id != ctx.run_id:
            continued = bool(self.finished) and ctx.goal.startswith(self.finished + "\n")
            self.run_id, self.line, self.since = ctx.run_id, self.finished.count("\n") + 1 if continued else 0, 0
            self.prev, self.routed, self.clicked, self.seen, self.typed, self.unmoved = 0, False, None, False, None, None
            self.kind = None
            self.doubted = self.screen = None
            self.refused = []
            if not ctx.goal.startswith(self.goal + "\n"):
                self.opened = None  # a new request: what an earlier one opened isn't its app
                self.joined, self.judged = set(), 1
        # A run that failed starts again with the words said since joined on, its history empty: "make a new tab"
        # after closing Chrome's last tab found no app to reopen (v11-t4).
        self.goal, self.opened = ctx.goal, _last_opened(ctx.history) or self.opened
        clicked, self.clicked = self.clicked, None
        doubted, self.doubted = self.doubted, None
        self.idle = ctx.history[-1:] == [LOOK_AGAIN.brief()] and self.screen is not None and _same(self.screen, image)
        # Flash's click, the step on the last screen handed to it, is judged as Jev's sure ones are: the Docs
        # shortcut it clicked for "open up a google doc template" read 0.46-0.54 left on the gallery that opened, and
        # the paste after it clicked a blank document there (2026-10-19).
        taken = ctx.history[self.planned[0]:]
        if self.screen is not None and len(taken) == 1 and taken != self.planned[1] and (point := _POINT.match(taken[0])) \
                and point[1] == "click":
            clicked = self.screen, (int(point[2]) / 1000, int(point[3]) / 1000), taken[0]
        inputs = [done for done in ctx.history if done not in (LOOK_AGAIN.brief(), WAIT.brief())]
        if clicked and (inputs[-1:] != [clicked[2]] or "unchanged" in ctx.note):
            clicked = None  # not the last input, or one with no visible effect: nothing it did to judge
        self.clicked = clicked  # still judged after looks again: the page it opened may take a few to draw
        guard = self.approved_run is not None and not self.approved_run(ctx)
        opened = _just_opened(ctx.history)
        # One line at a time: "click the first email" as the last of 8 lines boxed Gmail's tab row (0.78 sure,
        # 3/3), and "close discord" with "open a terminal" after it was routed to the terminal (2/2).
        # shortcut: a correction said mid-line waits for that line to finish; show Jev the later lines too if
        # corrections need to cut in.
        # The next line is asked alongside, so its first step needs no look of its own once this line is done.
        # Lines new to Jev are asked too whether each is the rest of the line before; one that is asks again, joined.
        raw = ctx.goal.splitlines()
        while True:
            lines = self._lines(ctx.goal)
            if self.line in self.joined:
                # Its words went on a line an earlier run finished: that line is judged again with them. "In d 5"
                # clicked its cell and ended, and "type hello." said after it, joined on, ended untyped (2026-10-20).
                j = max((k for k in range(self.line) if lines[k]), default=0)
                self.line, self.since, self.finished = j, self.prev, "\n".join(raw[:j])
                continue
            if not lines[self.line]:  # an approval
                return self._next_line(image, ctx, lines, "Jev verified the completed goal")
            before = lines[self.line - 1] if self.line else ""
            asks = [(lines[self.line], ctx.history[self.since:], _seen(ctx.history, before, self.prev, self.since),
                     not self.routed)]
            if self.line + 1 < len(lines) and lines[self.line + 1]:
                asks.append((lines[self.line + 1], [], _seen(ctx.history, lines[self.line], self.since), True))
            typed = _typed_into_field(ctx.history[self.since:], self.typed)
            looks = [self._line_done(mark(*clicked[:2]), ctx, lines, LINE_LEFT_CLICKED) if clicked
                     else self._line_done(image, ctx, lines)]
            shown = [self._shown(image, ctx, lines[self.line], typed)] if typed else []
            rests = [i for i in range(self.judged, len(raw)) if raw[i].startswith(CORRECTION)]
            self.judged = len(raw)
            replies = self._evaluate_all(ctx, looks + shown + [self._ask(image, ctx, *ask, guard, opened) for ask in asks]
                                         + [self._rest(image, raw, i) for i in rests])
            joined = {i for i, (answers, _, _) in zip(rests, replies[len(replies) - len(rests):])
                      if answers["joined"].get("choice") == "one"}
            if not joined:
                break
            self.joined |= joined
        answered = replies[len(looks) + len(shown):len(replies) - len(rests)]
        missing = bool(shown) and probability(replies[len(looks)][0]["typed"].get("noul")) < MISSING
        left = max(1 - probability(answers["left"].get("noul")) for answers, _, _ in replies[:len(looks)])
        # An input that changed nothing finished nothing: a Brave Search result clicked 1.3 s after the results drew
        # stayed put, the unchanged page read 0.35 left, and the snippet under it was clicked as the next line's
        # "date of birth reference" (2026-10-19; 2 of 3 clicks that early did nothing, every one at 4 s went).
        unchanged = "unchanged" in ctx.note and bool(ctx.history[self.since:])
        retrying = self._retrying(ctx.history)
        retried = unchanged and self.unmoved is not None and not retrying
        # Nor does the look after it while the screen stays as it was: the [20] after "Temüjin's birth" clicked on
        # its top edge stayed put, the look a second later read the line done (0.54), and the page itself was copied
        # (2026-10-19).
        still = retrying and not unchanged and _same(self.unmoved[1], image)
        if not unchanged and not retrying:
            self.unmoved = None
        finished = not missing and not unchanged and not still and _finished(answered[0][0], ctx.history[self.since:], left, self.seen)
        # A click Jev doubted doesn't end its line while Jev has another for it that it is sure of: after the lead's
        # [b] the screen read done (0.38-0.55) with the note that cites the birth showing, its [20] boxed and zoomed
        # 0.70 (2026-10-19). Without such a click the line is done.
        # Nor while an app launcher is open with an entry Jev is sure to click: "open the calculator" read done (0.72)
        # on a phone's app drawer listing Calculator, which Samsung opens only on a tap (2026-10-19).
        first = answered[0][0]
        listed = "launcher" in first and min(probability(first["launcher"].get("noul")),
                                             probability(first["present"].get("noul"))) >= GATE
        held = finished and first["next"].get("choice") in CLICKS and (ctx.history[-1:] == [doubted] or listed)
        fresh = opened and first["next"].get("choice") == "new_tab" and probability(first["fresh"].get("noul")) >= GATE \
            and any(s.startswith("skill(open_app ") for s in ctx.history[self.since:])
        finished = (finished or fresh) and not held
        if finished and len(answered) > 1:
            self._advance(ctx)
            answered, finished = answered[1:], False  # the next line's first step
        self.seen = True
        answers, size, padded = answered[0]
        line, steps = lines[self.line], ctx.history[self.since:]
        seen = _seen(ctx.history, lines[self.line - 1] if self.line else "", self.prev, self.since)
        texts = spans(line, MAX_SPANS)
        named = {str(i) for i in range(len(texts))}
        routing = "fits" in answers
        if probability(answers["question"].get("noul")) >= GATE \
                and (finished or probability(answers["complete"].get("noul")) >= GATE):
            # Jev called "what is this page about?" complete and said "Done." with no answer. Only then: "search up
            # genghis khan" scored a question 0.50-0.63 (3/3) with Google's search box boxed to click.
            return _fail("the request asks for an answer, not an action")
        if probability(answers["composed"].get("noul")) >= GATE and answers["next"].get("choice") in ("type", "fill"):
            # Only a step that types needs the text: "open camera" on a phone (a tap) was refused here, and so was
            # "go to google docs" (0.55) when the address bar takes a site's name as the request gives it.
            # "work out 12 times 9": Jev pressed keypad keys one per step and missed one (41.7 s; Flash typed
            # "12*9=" in one step, 12.2 s).
            return _fail("the rest needs text the request doesn't give word for word", planner=True)
        if finished:
            return self._next_line(image, ctx, lines, "Jev verified the completed goal")
        if not steps and probability(answers["complete"].get("noul")) >= ALREADY:
            return self._next_line(image, ctx, lines, "Jev sees the line already done")
        # A paste that changed nothing went where no key lands: Ctrl+V into a Docs template just inserted, its caret
        # drawn, pasted nothing and the line read done on Docs' "Saving..." (2026-10-19). Its retry clicks first.
        refocus = unchanged and not retried and bool(_PASTE.fullmatch(steps[-1]))
        # A copy that changed nothing selected nothing (a selection shows): its select landed on Chrome's toolbar
        # edge, the look after it read the copy done (paste next, 0.22 complete) and Flash went on with nothing
        # copied (2026-10-19). It copies again from a fresh look.
        recopy = unchanged and not retried and bool(_COPY.fullmatch(steps[-1]))
        if unchanged:
            # Once more after a look: a page just drawn may not take input yet. Twice is Jev choosing it again.
            self.unmoved = len(ctx.history), image
            look = None if retried or refocus or recopy else self._look_again(ctx)
            if not (refocus or recopy):
                return look or _fail("the last step had no visible effect")
        # Text sent with Enter leaves its box while the page it asked for loads: Google's results drew 2.7 s after
        # Enter on Chrome's new tab, its emptied box read not typed (0.08), and the space in the text typed again
        # scrolled those results past the Wikipedia link (2026-10-19).
        sent = [s for s in steps if s not in (LOOK_AGAIN.brief(), WAIT.brief())][-1:]
        if missing and sent and _ENTER.fullmatch(sent[0]) and (look := self._look_again(ctx)):
            return look
        # Only when nothing can be done yet: YouTube's placeholder page already has a search box to click. An app
        # just opened is waited for before its first step: its own controls may not be there yet.
        # A key for a tab or window needs nothing drawn: Chrome's drawn new tab page read as drawing 0.51-0.58, and
        # "close google" waited 12 s for it (2026-10-19). What is still drawing is drawn within two looks.
        keys = answers["next"].get("choice") in (*TAB_KEYS, "close_app")
        drawing = opened and not keys and probability(answers["drawing"].get("noul")) >= GATE
        pending = not keys and probability(answers["pending"].get("noul")) >= GATE
        loading = pending or probability(answers["loading"].get("noul")) >= GATE
        wait = self._look_again(ctx, MAX_LOADING_WAITS if loading else MAX_WAITS, anytime=True) if loading or drawing \
            else None
        if (opened or pending) and wait:
            return wait
        if routing:
            # A website that names no browser is no app to search the launcher for: "youtube" found nothing there.
            site = probability(answers["site"].get("noul")) >= GATE \
                and probability(answers["browser_named"].get("noul")) < GATE
            choice = answers["app_name"].get("choice")
            name = BROWSER if site else texts[int(choice)] if choice in named else None
            # Not a site's own name, which no launcher lists ("youtube" found nothing on Cinnamon).
            self.kind = texts[int(choice)] if site and choice in named \
                and probability(answers["dotcom"].get("noul")) < GATE else None
            # "make a new tab" after closing Chrome's last tab: a line that names no app, with no app window open,
            # goes on in the app the request last opened, by name, since a visible app suited to the line can be
            # another browser.
            closed = probability(answers["window"].get("noul")) < GATE
            # Nor does another app in front: Calc behind the closed Chrome fitted "make a new tab" 0.97 (its sheet
            # tabs), and the step went to Calc, as Ctrl+T and as a click on its add-sheet button (2026-10-19).
            elsewhere = "opened_front" in answers and probability(answers["opened_front"].get("noul")) < GATE
            again = self.opened if (closed or elsewhere) and not name else None
            # The open_app skill knows each launcher's keys (a phone's drawer needs Ctrl+F before typing). Once per
            # app in a clause: one line can name two ("open up discord, close discord, and open a terminal").
            launch = Action("skill", name="open_app", args={"name": name or again})
            # Before the fits check: Jev can judge the open launcher itself as fitting and type the whole goal.
            search = self._search(answers, ctx, name, steps)
            if search:
                return search
            # Nothing is in front with only the desktop showing: "open a terminal" there fitted 0.53, Jev boxed VS Code
            # beside the terminal's panel icon, its zooms the terminal, and the line failed (2026-10-20).
            fits = (probability(answers["fits"].get("noul")) >= GATE
                    or site and probability(answers["browser_front"].get("noul")) >= GATE) and not (closed and name)
            # The demo phone ignored taps; its keyboard launcher works. Not for the app in front: Brave was launched
            # again for each line of "search for horses and click the third video" on a DeX phone (2026-10-19).
            if ctx.profile.input == "touch" and name and not fits and launch.brief() not in steps + self.refused:
                return Plan("Open the app", [launch], [], blind=True)
            if (not fits or elsewhere) and again and launch.brief() not in steps + self.refused:
                return wait or Plan("Open the app again", [launch], [], blind=True)
            # Only a request that names an app or a site is routed: gibberish ("asdkjh qwe zxcv") opened the launcher.
            if not fits and name:
                # An app still opening is the one just launched: clicking its panel icon minimised it.
                route = wait or self._route(image, line, answers, size, padded, ctx, launch, steps)
                if route:
                    return route
            else:
                self.routed = True
        step = "copy" if recopy else answers["next"].get("choice")
        choice = answers["text"].get("choice")
        text = texts[int(choice)] if choice in named else None
        if step == "address" and text is not None and _is_the_request(text, line):
            # A site's name is part of the line: "go to google docs" on Chrome's Google page chose the whole line
            # 0.37, "google docs" 0.35, and the bare click on the address bar it became cost 4 steps (2026-10-20).
            odds = {k: probability(v) for k, v in (answers["text"].get("probabilities") or {}).items()}
            site = max((k for k in named if not _is_the_request(texts[int(k)], line)), key=lambda k: odds.get(k, 0.0))
            if odds.get(site, 0.0) > odds.get("none", 0.0):
                text = texts[int(site)]
        if step == "erase" and text is None:
            text = _last_typed(ctx.history[self.prev:])  # "delete hello" right after typing it: Jev picked no words (2/2)
        if missing:
            step, text = "fill", typed  # into the field this time, clicked first
        mod = "GUI" if ctx.profile.os in ("macos", "ios") else "CTRL"
        actions: list[Action] = []
        if step == "address" and probability(answers["lookup"].get("noul")) >= GATE \
                and probability(answers["submit"].get("noul")) < GATE:
            return _fail("the request describes a page to look for, and names no website", planner=True)
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
        if step == "paste" and ctx.history and ctx.history[-1].startswith("click(") \
                and probability(answers["focused"].get("noul")) < GATE and (look := self._look_again(ctx)):
            # What the click opened may still be opening: a Docs template clicked in its gallery took the paste
            # before its document showed (focused 0.25), and the line read done (2026-10-19).
            return look
        # As with typing: a paste right after a click, or into the place Jev sees focused, needs no click first.
        focus = step == "paste" and (refocus or not (ctx.history and ctx.history[-1].startswith("click("))
                                     and probability(answers["focused"].get("noul")) < GATE)
        if focus and not refocus and (box_center(answers["target"], size, padded) is None
                                      or probability(answers["present"].get("noul")) < GATE):
            # Where Jev can't see the place to click, the keys go where the page put its caret: a Docs template
            # just opened was 0.34-0.39 sure of a body line, waited twice, and Flash, handed it after 3.3 s, pressed
            # only Ctrl+V, which pasted (2026-10-19). A paste that changes nothing clicks on its retry.
            look = wait or self._look_again(ctx)
            if look:
                return look
            focus = False
        certain = settled = True  # of the click's target: only a doubtful zoom says otherwise
        if step in ("click", "double_click", "right_click", "fill", "copy") or focus:
            point = box_center(answers["target"], size, padded)
            opens = probability(answers["opens"].get("noul"))
            sure = max(probability(answers["present"].get("noul")), opens if opens >= GATE else 0.0)
            control = OPENER_CONTROL if opens >= GATE else STEP_CONTROL
            if held and (point is None or sure < GATE):
                return self._next_line(image, ctx, lines, "Jev verified the completed goal")
            # Not a copy's: on the citation a footnote jump marked, "copy the header" was 0.22-0.35 sure of it, waited,
            # and Flash took the hatnote at the article's top (0.53 there, 2026-10-19). Its own look places it.
            if step != "copy" and (point is None or sure < GATE):
                return wait or self._look_again(ctx) or _fail("its target isn't clearly visible", planner=True)
            width, height = box_size(answers["target"], size, padded)
            if step == "copy":
                point = self._copy_spot(image, ctx, lines, seen)
                if point is None:
                    return wait or self._look_again(ctx) or _fail("its target isn't clearly visible", planner=True)
            elif min(width, height) >= BIG_TARGET and sure >= self.jev.min_presence \
                    and probability(answers["target"].get("confidence")) < BIG_SURE:
                return wait or self._look_again(ctx) or _fail("its target isn't clearly visible", planner=True)
            elif min(width, height) < BIG_TARGET or sure < self.jev.min_presence:
                # Two more looks at an enlarged crop: the first box can land a row off (a missed search box), but
                # one zoom also strays to a neighbouring row about 1 in 6 times (Chrome's new tab: search box vs
                # shortcuts). Keeping the zoom nearer the first box is a majority of the three looks. No tighter
                # look after it: on cell B1 it lost the row and column headers and chose the selected A1 (5 of 8).
                # typing that went nowhere left out: told the text was typed, both zooms doubted the search box (0.41-0.48)
                state = {"goal": line, "executed_steps": seen[:self.since - self.prev] if missing else seen}
                # the crop's side is a fraction of the screen's longer side, and its height 0.66 of its width
                window = max(ZOOM_WINDOW, ZOOM_MARGIN * max(width * size[0], height * size[1] / 0.66) / max(size))
                with ThreadPoolExecutor(2) as pool:
                    zooms = list(pool.map(lambda _: zoom(self.jev, image, state, control, point, window,
                                                          ctx.run_id, ctx.step), range(2)))
                # A zoom boxing the other's box with room around it boxed the region it sits in: "search up genghis
                # khan" on Chrome's new tab had one zoom on the search box, one on it and the logo above, nearer the
                # first look's address bar, and the click on the logo typed into nothing (2026-10-19).
                zooms = [z for z in zooms if not any(encloses(z[2], other[2]) for other in zooms)]
                looks = [(p, b) for p, z, b in zooms if p and z >= GATE]
                certain = max(z for _, z, _ in zooms) >= CLICK_SURE
                settled = max(z for _, z, _ in zooms) >= HOLD_SURE
                again = ctx.history[-1:] == [LOOK_AGAIN.brief()]
                # Three looks on one spot agree: both zooms on cell B1 read 0.46-0.49, inside Jev's box, and the look
                # again they cost (2.2 s) placed the same click (2026-10-20).
                agree = len(zooms) == 2 and all(p and box_contains(answers["target"], size, padded, p) for p, _, _ in zooms)
                if not looks and (again or agree):
                    # A page that drew no closer view in a look again won't: both zooms on Wikipedia's [b] read
                    # 0.41-0.53 twice, and Flash, handed it after 7 s, clicked [b] (2026-10-19). As a doubted click.
                    looks = [(p, b) for p, z, b in zooms if p and z >= DOUBT]
                # Jev's doubt about which step comes next also lowers its sureness that the target is visible (a
                # small YouTube search box: 0.77-0.84 on 5 tries, each box on it; a spreadsheet cell 0.79-0.81), so
                # a zoom that lands in its box stands in for it: on cell B1 one zoom in 3 drifted to the selected
                # A1 beside it. Zooms elsewhere don't: both left Chrome's Gmail link for the address bar. Once a look
                # again saw the same, zooms beside the box do: the step's box on Wikipedia's birth date ended
                # 0.007-0.016 left of the [b] both zooms boxed, three looks running, and Flash took it (2026-10-19).
                if sure < self.jev.min_presence:
                    looks = [(p, b) for p, b in looks if box_contains(answers["target"], size, padded, p)
                             or again and beside(answers["target"], size, padded, b)]
                if held and not (looks and certain):
                    return self._next_line(image, ctx, lines, "Jev verified the completed goal")
                if not looks:
                    # On a run's first screen too: "close the spreadsheet without saving" boxed Calc's close button,
                    # both zooms the window's 0.41-0.47, and the run failed with nothing tried (2026-10-20). A look
                    # again takes the zooms over DOUBT.
                    return wait or self._look_again(ctx, anytime=True) \
                        or _fail("its target isn't clearly visible", planner=True)
                first = _edges(answers["target"], size, padded)
                point = min((_narrowed(p, b, first) for p, b in looks), key=lambda p: math.dist(p, point))
            if step == "copy":
                actions.append(Action("skill", name="select", args={"x": round(point[0], 3), "y": round(point[1], 3)}))
            else:
                actions.append(Action(step if step in CLICKS else "click", x=point[0], y=point[1]))
        if step in ("copy", "paste"):
            actions.append(Action("hotkey", keys=(mod, CLIPBOARD[step])))
        in_place = False
        if step in ("type", "fill", "address", "erase"):
            if text is None:
                # "open up a google doc template" on a new tab: Jev chose the address bar with no words to type
                return _fail("the text to type isn't in the request", planner=True)
            if _is_the_request(text, line):
                # "write a short note about my day" was typed as the note (Diego, agent/jev-only-fixes).
                return _fail("the request asks for text to be written, not typed word for word", planner=True)
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
                self.typed = actions[-1].brief()
                # Enter in the same batch: a separate step for it cost one more Jev call (0.5-0.7 s).
                if step == "address" and ctx.profile.os in DOTCOM_OS and not re.search(r"\s", text) \
                        and probability(answers["dotcom"].get("noul")) >= GATE:
                    actions.append(Action("hotkey", keys=("CTRL", "ENTER")))
                elif step == "address" or probability(answers["submit"].get("noul")) >= GATE:
                    actions.append(Action("hotkey", keys=("ENTER",)))
                elif probability(answers["commit"].get("noul")) >= self.jev.min_presence:
                    # Calc's Enter after "hello" moved too little to count as a reaction, so the settle held the full
                    # 1 s a page load may take to start, and the run's last look came 1.1 s after it (2026-10-20).
                    actions.append(Action("hotkey", keys=("ENTER",)))
                    in_place = True
        if step in TAB_KEYS:
            actions.append(Action("hotkey", keys=(mod, TAB_KEYS[step])))
        if step == "close_app":
            # Samsung DeX closes the window in front on Alt+F4 too (the calculator, 2026-10-19).
            # shortcut: a phone outside DeX has no quit key, so the step fails; close from the app switcher if needed.
            actions.append(Action("hotkey", keys=("GUI", "Q") if ctx.profile.os == "macos" else ("ALT", "F4")))
        if step == "enter":
            actions.append(Action("hotkey", keys=("ENTER",)))
        if step in ("scroll_down", "scroll_up"):
            # A line begun on Enter sees the page Enter left: "click the third video" after a YouTube search scrolled
            # YouTube's home page under its suggestions twice while the results loaded (2.7 s, 2026-10-20).
            if not steps and ctx.history and _ENTER.fullmatch(ctx.history[-1]) and (look := self._look_again(ctx)):
                return look
            if ctx.profile.input == "mouse":
                # The wheel scrolls what is under the pointer: a toolbar scrolled nothing on a Mac, and the screen's
                # middle was the desktop beside a half-screen Chrome, so its apps panel never scrolled (3 runs).
                # Jev's box for a scroll is what it reveals or the panel holding it.
                sure = probability(answers["present"].get("noul")) >= GATE
                x, y = (box_center(answers["target"], size, padded) if sure else None) or (0.5, 0.5)
                actions.append(Action("skill", name="hover", args={"x": round(x, 3), "y": round(y, 3)}))
            actions.append(Action("scroll", dy=5 if step == "scroll_down" else -5))
        if not actions:
            # Discord's update splash has nothing to act on for 4 s after its icon is clicked. A finished request
            # of three clauses scored 0.63-0.65 complete, with nothing left to do.
            if not wait and probability(answers["complete"].get("noul")) >= GATE:
                return self._next_line(image, ctx, lines, "Jev sees nothing left to do")
            # Jev's answers vary between looks: "open up youtube" on a browser's new tab was stuck 1 time in 3.
            return wait or self._look_again(ctx) or _fail("it needs reasoning or content Jev can't supply")
        if _repeats(actions, steps) and not self._retrying(ctx.history):
            # Clicking "1" twice typed 11 for "12 times eight", and Enter twice submitted twice: never repeat
            # input to find out whether it worked (Diego, agent/jev-only-fixes). Only the line's own: Docs' Blank
            # document tile was where the Google Docs search result just clicked had been (v11-t4). A look again isn't
            # a repeat: Enter after Flash's "https://www.google.com\n" was the line's screen not yet read done, and
            # failing there restarted the request (2026-10-19).
            return self._look_again(ctx) or _fail("it would repeat the step it just took")
        if guard and probability(answers["irreversible"].get("noul")) >= GATE:
            return Plan("Irreversible step needs approval", [], [], held=f'the next step of "{line}"')
        if step in CLICKS and certain:
            self.clicked = image, point, actions[0].brief()
        if step in CLICKS and not settled:
            self.doubted = actions[0].brief()
        return Plan(f"Jev next step: {step}", actions, [], blind=step in ("address", "new_tab"),
                    in_place=in_place)

    def _look_again(self, ctx: StepContext, limit: int = MAX_WAITS, anytime: bool = False) -> Plan | None:
        """A target not visible yet is often a page still loading: right after a launch, YouTube's search box
        wasn't there and Flash took the rest (30 s on the Pi). Wait briefly instead, up to `limit` in a row
        (nrebolloso, origin/live-speech cfb822b); on a run's first screen only when Jev sees it still `anytime`
        loading."""
        # A page a look again left as it was is drawn: Chrome's new tab page waited twice for a Google Docs
        # template 2.4 s apart, unchanged, before Flash clicked its Docs shortcut (2026-10-19). Loading isn't: an
        # app launching or a placeholder page can sit still a second.
        if self.idle and limit <= MAX_WAITS:
            return None
        waits = 0
        for done in reversed(ctx.history):
            if done != LOOK_AGAIN.brief():
                break
            waits += 1
        return Plan("Jev waits for the page", [LOOK_AGAIN], []) if (ctx.history or anytime) and waits < limit else None

    def _ask(self, image: bytes, ctx: StepContext, line: str, steps: list[str], context: list[str], route: bool,
             guard: bool, opened: bool) -> Ask:
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
                   "browser_front": {"type": "noul", "instructions": BROWSER_FRONT},
                   "launcher": {"type": "noul", "instructions": LAUNCHER_OPEN},
                   "window": {"type": "noul", "instructions": WINDOW}} \
            | ({"opened_front": {"type": "noul", "instructions": OPENED_FRONT.format(app=json.dumps(self.opened))}}
               if self.opened else {}) if route and ctx.profile.os in OPEN_APP_KEYS else {}  # firmware has no apps
        irreversible = {"irreversible": {"type": "noul", "instructions": NEXT_IRREVERSIBLE}} if guard else {}
        drawing = {"drawing": {"type": "noul", "instructions": DRAWING},
                   "fresh": {"type": "noul", "instructions": FRESH}} if opened else {}
        stepping = irreversible | drawing | {
            "complete": {"type": "noul", "instructions": COMPLETION_QUESTION},
            "question": {"type": "noul", "instructions": QUESTION},
            "composed": {"type": "noul", "instructions": COMPOSED},
            "next": {"type": "choice", "instructions": NEXT_QUESTION, "criteria": NEXT_CRITERIA},
            # "or text": what a copy selects is no control. A Wikipedia article's title as "the title" was seen
            # 0.77-0.90 so, 0.33-0.49 as a control; on 11 frames of past clicks (3 tries each) it boxed the same
            # targets as sure or surer, but for Calc on LibreOffice's Start Center (0.60-0.70, from 0.79-0.86).
            "present": {"type": "noul", "instructions":
                        "Is the one control or text that the next step of the `goal` acts on visible in `screen`, "
                        f"with no ambiguity? {OPENER}"},
            "target": {"type": "box", "instructions":
                       f"The single visible control or text in `screen` that the next step of the `goal` acts on. "
                       f"{OPENER}"},
            "text": {"type": "choice", "instructions": TEXT_QUESTION,
                     "criteria": {"none": "No text to type."} | options},
            "replace": {"type": "noul", "instructions": REPLACE_QUESTION},
            "submit": {"type": "noul", "instructions": SUBMIT_QUESTION},
            "commit": {"type": "noul", "instructions": COMMIT},
            "input": {"type": "noul", "instructions": INPUT},
            "dotcom": {"type": "noul", "instructions": DOTCOM},
            "lookup": {"type": "noul", "instructions": LOOKUP},
            "focused": {"type": "noul", "instructions": FOCUSED},
            "loading": {"type": "noul", "instructions": LOADING},
            "pending": {"type": "noul", "instructions": PENDING},
            "opens": {"type": "noul", "instructions": OPENS},
        }
        return image, "jev", ([(routing, {"goal": line, "executed_steps": steps})] if routing else []) \
            + [(stepping, {"goal": line, "executed_steps": context})]

    def _line_done(self, image: bytes, ctx: StepContext, lines: list[str], question: str = LINE_LEFT) -> Ask:
        """Whether the line Jev sees is left to do ("left"), asked with the line before it in view."""
        shown = "\n".join(filter(None, lines[max(self.line - 1, 0):self.line + 1]))
        return image, "line-done", [({"left": {"type": "noul", "instructions": question}},
                                     {"goal": shown, "executed_steps": ctx.history[self.prev:]})]

    def _rest(self, image: bytes, raw: list[str], i: int) -> Ask:
        """Whether the goal's line `i` is the rest of the sentence on the line before it ("joined": one)."""
        first, rest = (line.removeprefix(CORRECTION) for line in raw[i - 1:i + 1])
        said = f"{first} {rest.rstrip('.')}"
        readings = {"one": f"One request: {json.dumps(said, ensure_ascii=False)}",
                    "two": f"Two requests: first {json.dumps(first, ensure_ascii=False)}, "
                           f"then {json.dumps(rest, ensure_ascii=False)}"}
        return image, "join", [({"joined": {"type": "choice", "instructions": JOIN, "criteria": readings}},
                                {"goal": f"{first}\n{rest}"})]

    def _shown(self, image: bytes, ctx: StepContext, line: str, text: str) -> Ask:
        """Whether the text the line typed shows where it was typed ("typed")."""
        question = TYPED.format(text=json.dumps(text.removesuffix("…"), ensure_ascii=False))
        return image, "typed", [({"typed": {"type": "noul", "instructions": question}},
                                 {"goal": line, "executed_steps": ctx.history[self.since:]})]

    def _copy_spot(self, image: bytes, ctx: StepContext, lines: list[str], context: list[str]) -> tuple[float, float] | None:
        """Where a triple click selects the text the line copies: just past the end of Jev's box around it, on the
        box's middle row, so it lands on no link in that text. Of two boxes the one reaching further: one look's box
        ended inside the citation, on its link or before it, 3 times in 15."""
        questions = {"copy_present": {"type": "noul", "instructions": COPY_PRESENT},
                     "copied": {"type": "box", "instructions": COPIED}}
        state = {"goal": "\n".join(lines[self.line:]), "executed_steps": context}
        with ThreadPoolExecutor(2) as pool:
            replies = list(pool.map(lambda _: self.jev.evaluate(image, questions, run_id=ctx.run_id, step=ctx.step,
                                                                state=state, label="copy"), range(2)))
        ends = [(center[0] + box_size(answers["copied"], size, padded)[0] / 2, center, answers["copied"], size, padded)
                for answers, size, padded in replies if (center := box_center(answers["copied"], size, padded))]
        if not ends:
            return None
        # shortcut: a paragraph wrapping to its column's edge puts this point past the column; aim inside its first
        # line if copies of long paragraphs need it.
        x, center, box, size, padded = max(ends, key=lambda end: end[0])
        point, _, _ = zoom(self.jev, image, state, COPY_CONTROL, center, COPY_WINDOW, ctx.run_id, ctx.step)
        # Only its row: a zoom off the box went to the address bar above it (1 in 16).
        y = point[1] if point and box_contains(box, size, padded, point) else center[1]
        return min(1.0, x + COPY_PAST), y

    def _evaluate_all(self, ctx: StepContext, asks: list[Ask]) -> list[tuple[dict, tuple[int, int], tuple[int, int]]]:
        """Each ask's answers, its question sets asked in parallel, those on the same image in the same state in one
        call: each call uploads the screen and a step's calls queue on those uploads (a 1080p screen sent at 1280 px
        from the Pi: one call alone 0.30 s, 2 at once 0.35 s, 3 0.42 s, 5 0.58 s)."""
        calls: list[tuple[bytes, dict, dict, list[str]]] = []  # image, questions, state, labels
        parts: list[list[tuple[int, dict]]] = []  # each ask's question sets, by the call they went in
        for image, label, sets in asks:
            parts.append([])
            for questions, state in sets:
                i = next((i for i, (shot, asked, said, _) in enumerate(calls) if shot is image and said == state
                          and all(asked.get(k, q) == q for k, q in questions.items())), len(calls))
                if i == len(calls):
                    calls.append((image, {}, state, []))
                calls[i][1].update(questions)
                calls[i][3].append(label)
                parts[-1].append((i, questions))
        with ThreadPoolExecutor(len(calls)) as pool:
            replies = list(pool.map(lambda c: self.jev.evaluate(c[0], c[1], run_id=ctx.run_id, step=ctx.step, state=c[2],
                                                                label="+".join(dict.fromkeys(c[3]))), calls))
        return [({k: v for i, questions in part for k, v in replies[i][0].items() if k in questions},
                 *replies[part[0][0]][1:]) for part in parts]

    def line_done(self, image: bytes, ctx: StepContext) -> Plan:
        """Another planner finished the line Jev is on: Jev's step for the next one, or the request is done."""
        return self._next_line(image, ctx, ctx.goal.splitlines(), "Flash verified the line done")

    def _next_line(self, image: bytes, ctx: StepContext, lines: list[str], why: str) -> Plan:
        """The line Jev sees is done: go on to the next one, or the request is done."""
        while self.line + 1 in self.joined:  # judged done with the line they joined
            self.line += 1
        if self.line + 1 < len(lines):
            self._advance(ctx)
            self.seen = False
            return self.next_actions(image, ctx)
        self.finished = ctx.goal
        return Plan(why, [Action("done", summary="Done.")], [])

    def _retrying(self, history: list[str]) -> bool:
        """An input changed nothing and only looks have run since: the next step is its one retry."""
        return self.unmoved is not None and all(done == LOOK_AGAIN.brief() for done in history[self.unmoved[0]:])

    def _advance(self, ctx: StepContext) -> None:
        self.line, self.prev, self.since, self.routed, self.kind = self.line + 1, self.since, len(ctx.history), False, None
        self.finished = "\n".join(ctx.goal.splitlines()[:self.line])

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
                return self._look_again(ctx, MAX_LOADING_WAITS) or _fail("the app the last click opened never appeared")
            # shortcut: one click opens taskbar, dock and launcher entries but not desktop icons; double-click
            # when the app box is a desktop icon if those get missed.
            return Plan("Switch to the app for the request", [switch], [])
        # Blind: a launch goes by name, so a Short playing behind it discarded "open up discord" twice (2026-10-19).
        if launch.brief() not in steps + self.refused:
            return Plan("Open the app", [launch], [], blind=True)
        # The app just launched is still coming: LibreOffice's splash ended the launch's wait and its window drew
        # after a look at the bare desktop, which pressed the launcher key over it (2026-10-20).
        if _just_opened(ctx.history) and (look := self._look_again(ctx)):
            return look
        if launcher.brief() not in steps and not self.refused:  # its key just showed no launcher
            return Plan("Open the app launcher", [launcher], [], blind=True)
        if ctx.history[-1:] != [WAIT.brief()]:  # the launcher, its results or the chosen app still appearing
            return Plan("Waiting for the app", [WAIT], [])
        return _fail("no app suited to the request is visible")


def _same(before: bytes, after: bytes) -> bool:
    """Whether two screens look alike; one that won't decode is not."""
    a, b = (cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) for jpeg in (before, after))
    return a is not None and b is not None and not visibly_changed(a, b)


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


def _typed_into_field(steps: list[str], own: str | None) -> str | None:
    """The text these steps' last input typed, Enter after it or not, when it was the `own` step. None after other
    input, or for text typed into the address bar: the page it opens replaces it."""
    inputs = [s for s in steps if s not in (LOOK_AGAIN.brief(), WAIT.brief())]
    while inputs and _ENTER.fullmatch(inputs[-1]):
        inputs.pop()
    if inputs[-2:-1] in (["hotkey(CTRL+L)"], ["hotkey(GUI+L)"]):
        return None
    return _last_typed(inputs[-1:]) if inputs[-1:] == [own] else None


def _last_opened(steps: list[str]) -> str | None:
    """The name of the app the last open_app step among these opened."""
    opened = [s for s in steps if s.startswith("skill(open_app ")]
    return ast.literal_eval(opened[-1][len("skill(open_app "):-1])["name"] if opened else None


def _finished(answers: dict, steps: list[str], done: float, seen: bool, gate: float = GATE) -> bool:
    """Jev verified the line done. Its own next step is Enter right after typing: the text isn't submitted,
    whatever the line-done score (0.94 for "add hello to b1" with hello typed into B1, not entered)."""
    unsent = bool(steps) and steps[-1].startswith("type_text") and answers["next"].get("choice") == "enter"
    # A line already seen not done that took no input since isn't done by itself: "close discord" scored done
    # between Discord's splash and its window (v7-t2), and Alt+F4 never ran; on the desktop before Discord's
    # window showed, its one plan discarded as the screen changed (v8-t2).
    # shortcut: a line the screen finishes on its own ("wait for the video to end") can't be done; let the
    # line-done check see why the screen changed if such lines matter.
    idle = seen and all(s in (LOOK_AGAIN.brief(), WAIT.brief()) for s in steps)
    # The screen doesn't show the clipboard: a copy or paste Jev still has next, on a line that pressed neither, is
    # untaken. "copy the header" read done (0.52) after Flash scrolled to the title, and a paste line after a
    # template's Insert click: neither copied nor pasted. Either key counts: once copied, Jev's next for "copy the
    # header" was paste (2026-10-19).
    # shortcut: a copy made through a menu still reads untaken while Jev picks copy; count menu copies if it matters.
    untaken = answers["next"].get("choice") in CLIPBOARD \
        and not any(re.fullmatch(rf"hotkey\(.*\+[{''.join(CLIPBOARD.values())}]\)", s, re.I) for s in steps)
    # A line that took no step of its own is held to ALREADY: "open up chrome" with Brave in front read done 0.58
    # and the request went on in the user's Brave (2026-10-19).
    return done >= (gate if steps else max(gate, ALREADY)) and not unsent and not idle and not untaken


def _edges(answer: dict, size: tuple[int, int], padded: tuple[int, int]) -> Box | None:
    """A box answer as (left, top, right, bottom, 0-1 of the image), as a zoom gives its box."""
    box = answer.get("box") or {}
    try:
        left, right, top, bottom = (probability(box.get(k)) for k in ("left", "right", "top", "bottom"))
    except ValueError:
        return None
    (w, h), (pw, ph) = size, padded
    return left * pw / w, top * ph / h, right * pw / w, bottom * ph / h


def _narrowed(point: tuple[float, float], box: Box | None, first: Box | None) -> tuple[float, float]:
    """A zoom's `point`, along each axis on which its `box` is over SPAN times as long as the `first` look's, in the
    middle of where the two overlap: inside the zoom's box still."""
    if box is None or first is None:
        return point
    moved = list(point)
    for axis in (0, 1):
        low, high = max(box[axis], first[axis]), min(box[axis + 2], first[axis + 2])
        if box[axis + 2] - box[axis] > SPAN * (first[axis + 2] - first[axis]) and low < high:
            moved[axis] = (low + high) / 2
    return moved[0], moved[1]


def _is_the_request(text: str, line: str) -> bool:
    """True when the text to type is the whole request line: an instruction, not dictated words. One word short of
    it is the dictated words of a command ("type Nicholas Rebelloza" was refused so); text the request only
    describes is Jev's composed answer to refuse."""
    request = _words(line)
    return len(request) >= 3 and len(_words(text)) >= len(request)


_POINT = re.compile(r"^(click|double_click|right_click)\((\d+),(\d+)\)$")
_ENTER = re.compile(r"hotkey\((\w+\+)?ENTER\)")
_PASTE = re.compile(r"hotkey\((CTRL|GUI)\+V\)", re.IGNORECASE)
_COPY = re.compile(r"hotkey\((CTRL|GUI)\+C\)", re.IGNORECASE)
SAME_SPOT = 20  # history coordinates are 0-1000: a click this close to the last one is the same click


def _repeats(actions: list[Action], history: list[str]) -> bool:
    """True when these actions are the same input as the last ones that ran (a click within SAME_SPOT, the
    same text or the same keys, Enter after text typed with its own). Scrolls, waits and pointer moves may repeat."""
    inputs = [a.brief() for a in actions if a.type not in ("scroll", "skill")]
    recent = [h for h in history if not h.startswith(("scroll(", "skill("))][-len(inputs):] if inputs else []
    if not inputs or len(recent) < len(inputs):
        return False
    for new, old in zip(inputs, recent):
        a, b = _POINT.match(new), _POINT.match(old)
        if a and b:
            if a[1] != b[1] or abs(int(a[2]) - int(b[2])) > SAME_SPOT or abs(int(a[3]) - int(b[3])) > SAME_SPOT:
                return False
        # Text typed with its newline was sent: Enter after Flash's "docs.new\n" went into the blank document it
        # opened, twice (2026-10-19).
        elif new != old and not (new == "hotkey(ENTER)" and old.startswith("type_text(") and old.endswith("\\n')")):
            return False
    return True


def _fail(why: str, planner: bool = False) -> Plan:
    """`planner`: Flash takes this one step even without hand-off: text written for it, which only a language
    model can supply, or a target Jev still can't make out after looking again ("copy the header" after a
    footnote jump: Jev 0.40-0.52 on the hatnote, the request ended; Flash copied a heading 4 of 6, 5-8 s)."""
    return Plan(FAILED, [Action("fail", reason=f"Jev alone can't take the next step: {why}.")], [], needs_planner=planner)
