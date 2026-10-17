"""Jev solo skills (experimental, JEV_SOLO_SKILLS=1): easy tasks with no LLM call.

Triage offers six more intents in its one Jev call: open an app, go to a website, type
into a field, press a key, scroll, or find words on the page. Jev also picks the value:
which contiguous words of the utterance to use (code types exactly those words, never
model-written text, computah's span idea on pixels + keyboard), which key from a closed
list, or which scroll direction. The only per-OS knowledge is the key tables below.
No phrase, app or site rules.
"""

from __future__ import annotations

import re

from brain.skills import OPEN_APP_KEYS
from brain.types import Action

SOLO_CRITERIA = {
    "open_app": "The utterance asks to open, launch or start an application or program by name "
                "(e.g. a browser, a text editor, settings, a music app), and nothing else.",
    "go_to_site": "The utterance asks to go to, open or visit a website or web address by name, and nothing else.",
    "type_into": "The utterance dictates exact words to type into one specific visible text field "
                 "(e.g. 'type hello in the search box'), and nothing else. Not writing or composing content.",
    "press_key": "The utterance asks to press one key or a standard shortcut (enter, escape, tab, undo, "
                 "go back, refresh, an arrow key...), and nothing else.",
    "scroll": "The utterance asks to scroll the page or view up or down, and nothing else.",
    "find_on_page": "The utterance asks to find or search for words on the current page or document, "
                    "and nothing else.",
}
VALUE_QUESTION = ("Which exact contiguous words of the `utterance` are the value it acts on: the application or "
                  "website to open, the text to type, or the words to find? Leave out request words such as "
                  "open, go to, type, find, search for, in the, the, my, please.")
BROWSER_QUESTION = "Is a web browser window open and in front in `screen`, ready to type an address?"
KEY_CRITERIA = {
    "enter": "Enter or return, submit, confirm", "escape": "Escape, cancel, close a menu or dialog",
    "tab": "Tab, next field", "backspace": "Backspace, delete the previous character",
    "delete": "Delete key", "space": "Space bar, play or pause",
    "up": "Up arrow", "down": "Down arrow", "left": "Left arrow", "right": "Right arrow",
    "page_up": "Page up", "page_down": "Page down", "home": "Home, go to the top", "end": "End, go to the bottom",
    "undo": "Undo", "redo": "Redo", "back": "Go back to the previous page", "refresh": "Refresh or reload the page",
}
KEY_QUESTION = "Which one key or standard shortcut does the `utterance` ask to press?"
SCROLL_QUESTION = "Which way does the `utterance` ask to scroll?"
SCROLL_CRITERIA = {"up": "Scroll up, toward the top", "down": "Scroll down, toward the bottom"}

INTENT_GATE = 0.7  # probability of the chosen solo intent
VALUE_GATE = 0.6  # probability of the chosen words, key or direction
MAX_WORDS = 30  # longer utterances aren't easy tasks; also keeps spans under Jev's ~254-option limit
MAX_OPTIONS = 250  # value_span options, plus none_here
MAX_SPAN_WORDS = 8  # long enough for a short dictated phrase
MAX_TEXT = 120
SCROLL_NOTCHES = 5
_EDGE_PUNCTUATION = ".,!?;:\"'()[]"

COMPUTERS = ("macos", "windows", "linux", "chromeos")
ADDRESS_BAR_KEYS = {os: ["GUI", "L"] if os == "macos" else ["CTRL", "L"] for os in COMPUTERS}
FIND_KEYS = {os: ["GUI", "F"] if os == "macos" else ["CTRL", "F"] for os in COMPUTERS}
PLAIN_KEYS = {"enter": "ENTER", "escape": "ESC", "tab": "TAB", "backspace": "BACKSPACE", "delete": "DELETE",
              "space": "SPACE", "up": "UP", "down": "DOWN", "left": "LEFT", "right": "RIGHT",
              "page_up": "PAGEUP", "page_down": "PAGEDOWN", "home": "HOME", "end": "END"}
SHORTCUTS: dict[str, dict[str, list[str]]] = {  # per OS; phones and BIOS have none
    "macos": {"undo": ["GUI", "Z"], "redo": ["GUI", "SHIFT", "Z"], "back": ["GUI", "LEFT"], "refresh": ["GUI", "R"]},
    "windows": {"undo": ["CTRL", "Z"], "redo": ["CTRL", "Y"], "back": ["ALT", "LEFT"], "refresh": ["F5"]},
    "linux": {"undo": ["CTRL", "Z"], "redo": ["CTRL", "SHIFT", "Z"], "back": ["ALT", "LEFT"], "refresh": ["F5"]},
    "chromeos": {"undo": ["CTRL", "Z"], "redo": ["CTRL", "SHIFT", "Z"], "back": ["ALT", "LEFT"],
                 "refresh": ["CTRL", "R"]},
}

Spans = dict[str, str]  # option id -> the exact utterance text it stands for
SPAN_SKILLS = frozenset({"open_app", "go_to_site", "type_into", "find_on_page"})


def spans(utterance: str) -> Spans:
    """Every contiguous run of up to MAX_SPAN_WORDS words, as exact slices of the utterance.
    Punctuation at word edges is trimmed ("youtube." -> "youtube"), inner dots kept ("google.com")."""
    words = []
    for match in re.finditer(r"\S+", utterance):
        start, end = match.start(), match.end()
        while start < end and utterance[start] in _EDGE_PUNCTUATION:
            start += 1
        while end > start and utterance[end - 1] in _EDGE_PUNCTUATION:
            end -= 1
        if start < end:
            words.append((start, end))
    if not words or len(words) > MAX_WORDS:
        return {}
    options = {f"s{i}_{j}": utterance[words[i][0]:words[j][1]]
               for i in range(len(words)) for j in range(i, min(i + MAX_SPAN_WORDS, len(words)))}
    return options if len(options) <= MAX_OPTIONS else {}


def solo_questions(options: Spans) -> dict[str, dict]:
    """The extra questions triage adds to its single Jev call when solo skills are on."""
    return {
        "value_span": {"type": "choice", "instructions": VALUE_QUESTION,
                       "criteria": {**options, "none_here": "The utterance names no such value."}},
        "browser_front": {"type": "noul", "instructions": BROWSER_QUESTION},
        "key": {"type": "choice", "instructions": KEY_QUESTION, "criteria": KEY_CRITERIA},
        "scroll_direction": {"type": "choice", "instructions": SCROLL_QUESTION, "criteria": SCROLL_CRITERIA},
    }


def solo_actions(skill: str, value: str, os: str, point: tuple[float, float] | None = None) -> list[Action] | None:
    """Actions for a solo skill on this OS, or None when it can't run here. `value` is the copied
    words (open_app, go_to_site, type_into, find_on_page), a KEY_CRITERIA id, or a scroll direction."""
    if skill in SPAN_SKILLS and not _typeable(value):
        return None
    if skill == "open_app" and os in OPEN_APP_KEYS:
        return [Action("skill", name="open_app", args={"name": value})]
    if skill == "go_to_site" and os in ADDRESS_BAR_KEYS:
        return [Action("hotkey", keys=tuple(ADDRESS_BAR_KEYS[os])), Action("type_text", text=value),
                Action("hotkey", keys=("ENTER",))]
    if skill == "find_on_page" and os in FIND_KEYS:
        return [Action("hotkey", keys=tuple(FIND_KEYS[os])), Action("type_text", text=value),
                Action("hotkey", keys=("ENTER",))]
    if skill == "type_into" and point is not None and os != "bios":
        # The controller turns the click into a tap on touch targets.
        return [Action("click", x=point[0], y=point[1]), Action("type_text", text=value)]
    if skill == "press_key":
        keys = [PLAIN_KEYS[value]] if value in PLAIN_KEYS else SHORTCUTS.get(os, {}).get(value)
        return [Action("hotkey", keys=tuple(keys))] if keys else None
    if skill == "scroll" and value in SCROLL_CRITERIA and os in COMPUTERS:
        return [Action("scroll", dy=SCROLL_NOTCHES if value == "down" else -SCROLL_NOTCHES)]
    return None


def _typeable(text: str) -> bool:
    return bool(text) and len(text) <= MAX_TEXT and not any(ord(char) < 32 or ord(char) == 127 for char in text)
