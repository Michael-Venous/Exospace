"""Jev solo skills (experimental, JEV_SOLO_SKILLS=1): easy keyboard tasks with no LLM call.

Triage offers two more intents in its one Jev call: open an app, or go to a website.
Jev also picks which contiguous words of the utterance name the app or site, and code
types exactly those words (never model-written text), computah's span idea on pixels +
keyboard. The only per-OS knowledge is a key table. No phrase, app or site rules.
"""

from __future__ import annotations

import re

from brain.skills import OPEN_APP_KEYS
from brain.types import Action

SOLO_CRITERIA = {
    "open_app": "The utterance asks to open, launch or start an application or program by name "
                "(e.g. a browser, a text editor, settings, a music app), and nothing else.",
    "go_to_site": "The utterance asks to go to, open or visit a website or web address by name, and nothing else.",
}
SPAN_QUESTION = ("Which exact contiguous words of the `utterance` name the application or website to open? "
                 "Leave out request words such as open, launch, start, go to, visit, the, my, please.")
BROWSER_QUESTION = "Is a web browser window open and in front in `screen`, ready to type an address?"

INTENT_GATE = 0.7  # probability of the chosen solo intent
SPAN_GATE = 0.6  # probability of the chosen word span
MAX_WORDS = 40  # longer utterances aren't easy tasks; they go to the planner
MAX_SPAN_WORDS = 5
MAX_TEXT = 60
_EDGE_PUNCTUATION = ".,!?;:\"'()[]"

# The address bar shortcut per OS; touch targets have none, so they use the planner.
ADDRESS_BAR_KEYS: dict[str, list[str]] = {
    "macos": ["GUI", "L"], "windows": ["CTRL", "L"], "linux": ["CTRL", "L"], "chromeos": ["CTRL", "L"],
}

Spans = dict[str, str]  # option id -> the exact utterance text it stands for


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
    return {f"s{i}_{j}": utterance[words[i][0]:words[j][1]]
            for i in range(len(words)) for j in range(i, min(i + MAX_SPAN_WORDS, len(words)))}


def solo_questions(options: Spans) -> dict[str, dict]:
    """The extra questions triage adds to its single Jev call when solo skills are on."""
    return {
        "name_span": {"type": "choice", "instructions": SPAN_QUESTION,
                      "criteria": {**options, "none_here": "No application or website is named."}},
        "browser_front": {"type": "noul", "instructions": BROWSER_QUESTION},
    }


def solo_actions(skill: str, text: str, os: str) -> list[Action] | None:
    """Keyboard actions for a solo skill on this OS, or None when it can't run here."""
    if not text or len(text) > MAX_TEXT or any(ord(char) < 32 or ord(char) == 127 for char in text):
        return None
    if skill == "open_app" and os in OPEN_APP_KEYS:
        return [Action("skill", name="open_app", args={"name": text})]
    if skill == "go_to_site" and os in ADDRESS_BAR_KEYS:
        return [Action("hotkey", keys=tuple(ADDRESS_BAR_KEYS[os])),
                Action("type_text", text=text),
                Action("hotkey", keys=("ENTER",))]
    return None
