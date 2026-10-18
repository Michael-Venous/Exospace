"""Model-free start: a goal made of unambiguous commands ("press ctrl c", 'type "hi"'),
or one that starts by opening an app or scrolling ("open calculator, then ...", "scroll
down and ..."), acts at once, with no model call. Commands after an app launch, including
"go to <address>" and "open a new tab", run blind once its window opens. The wrapped
executor then checks the result and does the rest from the next step."""

from __future__ import annotations

import re
from typing import Any

from brain.agent.executor import Executor, Plan, StepContext
from brain.keys import MODIFIERS, canonical
from brain.skills import OPEN_APP_KEYS
from brain.types import Action, OSName

_END = r"(?=\s*(?:$|[,;!?]|\.(?:\s|$)|\s(?:and|then)\b))"
_OPEN = re.compile(r"(?:open|launch)\s+(?:up\s+)?(?:the\s+)?([a-z0-9][a-z0-9 ]*?)(?:\s+app(?:lication)?)?" + _END, re.I)
_PRESS = re.compile(r"press\s+(?:the\s+)?([a-z0-9+ -]+?)(?:\s+keys?)?" + _END, re.I)
_TYPE = re.compile(r'type\s+["“]([^"“”\n]+)["”]' + _END, re.I)
_SCROLL = re.compile(r"scroll\s+(up|down)" + _END, re.I)
_NEW_TAB = re.compile(r"open\s+(?:a\s+)?new\s+tab" + _END, re.I)
_ADDRESS = re.compile(r"(?:go\s+to|visit)\s+([a-z0-9-]+(?:\.[a-z0-9-]+)+(?:/[^\s,;!?]*)?)" + _END, re.I)
# Browser address-bar and tab keys; phones have none without a keyboard.
_BROWSER_OS = frozenset({"windows", "macos", "linux", "chromeos"})
_SEPARATOR = re.compile(r"\s*(?:[,;.!?]\s*)?(?:(?:and|then)\b\s*)*", re.I)
# "open X" names something inside an app, or is vague, when X contains any of these.
# shortcut: a word list misses other on-screen things ("open the invoice"); ask the model first if misfires show up.
_NOT_APP = frozenset(
    "a an my new up it its this that these those some first second third last next previous top "
    "latest one them his her their your our another from with in on to for of at by menu tab window "
    "folder file page dialog link email message document url website site result results "
    "attachment history profile".split())
# "open a/an X" names a kind of app only for these; "open a new tab" stays with the model.
_APP_KINDS = frozenset({"spreadsheet", "text editor", "word processor", "calculator", "camera", "browser",
                        "web browser", "terminal", "file manager"})


def quick_actions(goal: str, os: OSName, launched: bool = False) -> list[Action]:
    """Actions for a goal whose every clause is an unambiguous command, else [], except
    that leading scrolls still act. An app launch acts even when the rest needs the model;
    `launched` allows the browser commands, which need the launched app in front."""
    actions: list[Action] = []
    rest = goal.strip()
    if "\n" in rest:  # a correction ("...\nUser correction: ...") can change the first command
        return []
    while rest:
        clause, end = _clause(rest, os, launched)
        if not clause:
            # The rest may qualify what came before ("press delete, but only ..."); a scroll undoes easily.
            return actions if all(a.type == "scroll" for a in actions) else []
        actions += clause
        rest = _SEPARATOR.sub("", rest[end:], count=1)
        if clause[0].type == "skill":
            return actions + quick_actions(rest, os, launched=True)
        if rest and clause[0].type == "hotkey" and set(clause[0].keys) <= MODIFIERS:
            return []  # half a combination: "press shift and press enter"
    return actions


def _clause(text: str, os: OSName, launched: bool) -> tuple[list[Action], int]:
    modifier = "GUI" if os == "macos" else "CTRL"
    if launched and os in _BROWSER_OS and (match := _NEW_TAB.match(text)):
        return [Action("hotkey", keys=(modifier, "T"))], match.end()
    if launched and os in _BROWSER_OS and (match := _ADDRESS.match(text)):
        return [Action("hotkey", keys=(modifier, "L")), Action("type_text", text=match[1]),
                Action("hotkey", keys=("ENTER",))], match.end()
    if match := _OPEN.match(text):
        words = match[1].split()
        if words[0].lower() in ("a", "an") and " ".join(words[1:]).lower() in _APP_KINDS:
            words = words[1:]
        if os in OPEN_APP_KEYS and len(words) <= 3 and not _NOT_APP & {w.lower() for w in words}:
            return [Action("skill", name="open_app", args={"name": " ".join(words)})], match.end()
    elif match := _PRESS.match(text):
        if keys := _combo(match[1]):
            if keys == ["HOME"] and os in ("android", "ios"):
                keys = ["GUI", "H"]  # the phone's home screen, not the cursor key (OS_HINTS)
            return [Action("hotkey", keys=tuple(keys))], match.end()
    elif match := _TYPE.match(text):
        return [Action("type_text", text=match[1])], match.end()
    elif (match := _SCROLL.match(text)) and os != "bios":  # firmware has no wheel
        return [Action("scroll", dy=-5 if match[1].lower() == "up" else 5)], match.end()
    return [], 0


def _combo(phrase: str) -> list[str]:
    """One key combination ("ctrl c", "ctrl+shift+t", "page down"), else []."""
    try:
        return [canonical(phrase)]
    except ValueError:
        pass
    try:
        keys = [canonical(part) for part in re.split(r"[\s+-]+", phrase) if part]
    except ValueError:
        return []
    return keys if sum(key not in MODIFIERS for key in keys) <= 1 else []  # a sequence needs the model


class QuickStart:
    """Executor wrapper that answers step 1 itself when `quick_actions` applies."""

    def __init__(self, inner: Executor) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def next_actions(self, image_jpeg: bytes, ctx: StepContext) -> Plan:
        if ctx.step == 1 and (actions := quick_actions(ctx.goal, ctx.profile.os)):
            from brain.agent.jev_only import scroll_actions  # the pointer moves onto the page first

            actions = [step for a in actions
                       for step in (scroll_actions(a.dy, ctx.profile.input) if a.type == "scroll" else [a])]
            return Plan("Quick command: " + ", ".join(a.brief() for a in actions), actions, [])
        return self.inner.next_actions(image_jpeg, ctx)

    def classify_os(self, image_jpeg: bytes) -> OSName:
        return self.inner.classify_os(image_jpeg)
