"""Model-free start: a goal that opens with unambiguous commands ("open calculator",
"press ctrl c", 'type "hi"', "scroll down") acts at once, with no model call. The
wrapped executor then checks the result and does the rest from the next step."""

from __future__ import annotations

import re
from typing import Any

from brain.agent.executor import Executor, Plan, StepContext
from brain.keys import MODIFIERS, canonical
from brain.skills import OPEN_APP_KEYS
from brain.types import Action, OSName

_END = r"(?=\s*(?:$|[,;!?]|\.(?:\s|$)|\s(?:and|then)\b))"
_OPEN = re.compile(r"(?:open|launch)\s+(?:the\s+)?([a-z0-9][a-z0-9 ]*?)(?:\s+app(?:lication)?)?" + _END, re.I)
_PRESS = re.compile(r"press\s+(?:the\s+)?([a-z0-9+ -]+?)(?:\s+keys?)?" + _END, re.I)
_TYPE = re.compile(r'type\s+["“]([^"“”\n]+)["”]' + _END, re.I)
_SCROLL = re.compile(r"scroll\s+(up|down)" + _END, re.I)
_SEPARATOR = re.compile(r"\s*(?:[,;.!?]\s*)?(?:(?:and|then)\b\s*)*", re.I)
# "open X" names something inside an app, or is vague, when X contains any of these.
_NOT_APP = frozenset(
    "a an my new up it this that these those some first last next previous another from with "
    "in on to for of at by menu tab window folder file page dialog link email message document "
    "url website site".split())


def quick_actions(goal: str, os: OSName) -> list[Action]:
    """Actions for the goal's leading clauses while each is an unambiguous command.
    An app launch ends the run: what comes next depends on what the app shows."""
    actions: list[Action] = []
    rest = goal.strip()
    if "\n" in rest:  # a correction ("...\nUser correction: ...") can change the first command
        return []
    while rest:
        action, end = _clause(rest, os)
        if action is None:
            break
        actions.append(action)
        if action.type == "skill":
            break
        rest = _SEPARATOR.sub("", rest[end:], count=1)
    return actions


def _clause(text: str, os: OSName) -> tuple[Action | None, int]:
    if match := _OPEN.match(text):
        words = match[1].split()
        if os in OPEN_APP_KEYS and len(words) <= 3 and not _NOT_APP & {w.lower() for w in words}:
            return Action("skill", name="open_app", args={"name": " ".join(words)}), match.end()
    elif match := _PRESS.match(text):
        if keys := _combo(match[1]):
            return Action("hotkey", keys=tuple(keys)), match.end()
    elif match := _TYPE.match(text):
        return Action("type_text", text=match[1]), match.end()
    elif match := _SCROLL.match(text):
        return Action("scroll", dy=-5 if match[1].lower() == "up" else 5), match.end()
    return None, 0


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
            return Plan("Quick command: " + ", ".join(a.brief() for a in actions), actions, [])
        return self.inner.next_actions(image_jpeg, ctx)

    def classify_os(self, image_jpeg: bytes) -> OSName:
        return self.inner.classify_os(image_jpeg)
