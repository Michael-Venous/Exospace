"""Model-free start: a goal made of unambiguous commands ("press ctrl c", 'type "hi"'),
or one that starts by opening an app or scrolling ("open calculator, then ...", "scroll
down and ..."), acts at once, with no model call. "open a new tab" is Ctrl+T on a computer;
"go to <address>" and a last "search <words>" run blind only after a browser's launch or a new tab. The wrapped
executor then checks the result and does the rest from the next step. `spoken_commands`
finds the commands at the start of speech still coming in, so they can act before it ends."""

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
_NEW_TAB = re.compile(r"open\s+(?:up\s+)?(?:a\s+)?new\s+tab" + _END, re.I)
_SEARCH = re.compile(r"search\s+(?:up\s+|for\s+)?([^,;!?]+?)" + _END, re.I)
# shortcut: a search naming where to search ("cats on youtube") goes to the model; widen if others slip through.
_WHERE = frozenset("on in at with using inside within youtube amazon wikipedia".split())
_TAB_KEYS = {("CTRL", "T"), ("GUI", "T")}  # a new tab: the browser is in front, so its address bar takes text
_ADDRESS = re.compile(r"(?:go\s+to|visit)\s+([a-z0-9-]+(?:\.[a-z0-9-]+)+(?:/[^\s,;!?]*)?)" + _END, re.I)
_GO = re.compile(r"(?:go|switch)\s+(?:back\s+)?to\s+(?:the\s+)?([a-z0-9][a-z0-9 ]*?)" + _END, re.I)
# Names of the apps on the target, lowercased (skills.installed_apps), when the target is this machine; else "go to X"
# can't tell an app from a site or a place in an app, and goes to the model.
INSTALLED: frozenset[str] = frozenset()
BROWSERS: frozenset[str] = frozenset()  # those that are web browsers (skills.installed_apps("WebBrowser"))
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
# "open a/an X" names a kind of app only for these.
_APP_KINDS = frozenset({"spreadsheet", "text editor", "word processor", "calculator", "camera", "browser",
                        "web browser", "terminal", "file manager"})
JUST_DID = "Just did: "  # brain/app.py's note on a request that follows on from the last one
CORRECTION = "User correction: "  # and on words that correct or answer the last one


def quick_actions(goal: str, os: OSName, launched: bool = False) -> list[Action]:
    """Actions for a goal whose every clause is an unambiguous command, else [], except
    that leading scrolls still act. An app launch acts even when the rest needs the model;
    `launched` allows the browser commands, which need a browser in front."""
    actions: list[Action] = []
    rest = goal.strip()
    if "\n" in rest:  # a correction ("...\nUser correction: ...") can change the first command
        note, _, rest = rest.rpartition("\n")
        if not note.startswith(JUST_DID) or "\n" in note:
            return []
        # Its last request already ran: a browser it launched is in front ("open chrome" ... "and go to x.com").
        launched = _fresh(quick_actions(note[len(JUST_DID):], os), launched)
    while rest:
        clause, end = _clause(rest, os, launched)
        if not clause:
            # The rest may qualify what came before ("press delete, but only ..."); a scroll undoes easily.
            return actions if all(a.type == "scroll" for a in actions) else []
        actions += clause
        rest = _SEPARATOR.sub("", rest[end:], count=1)
        # A launch, a new tab or a page load (several actions) acts even when the rest needs the model.
        if clause[0].type == "skill" or clause[0].keys in _TAB_KEYS or len(clause) > 1:
            return actions + quick_actions(rest, os, launched=_fresh(clause, launched))
        launched = _fresh(clause, launched)
        if rest and clause[0].type == "hotkey" and set(clause[0].keys) <= MODIFIERS:
            return []  # half a combination: "press shift and press enter"
    return actions


def _fresh(actions: list[Action], launched: bool) -> bool:
    """Whether a browser just launched, or a new tab, is still in front after these actions: another app's
    launch puts that app in front ("open discord, go to x" typed x into Discord's message box and sent it);
    Enter loads a page, whose own search box then means "search"."""
    for a in actions:
        if a.type == "skill":
            launched = _named(a.args["name"], BROWSERS)
        else:
            launched = (launched or a.keys in _TAB_KEYS) and a.keys != ("ENTER",)
    return launched


def _named(name: str, apps: frozenset[str]) -> bool:
    """Whether `name` is whole words of one of `apps`: "chrome" is "Google Chrome", "firefox" "Firefox Web Browser"."""
    return any(f" {name.lower()} " in f" {app} " for app in apps)


def _clause(text: str, os: OSName, launched: bool) -> tuple[list[Action], int]:
    # In the address bar, Delete drops Chrome's inline completion ("Genghis Khan" -> "Genghis Khan videos").
    modifier = "GUI" if os == "macos" else "CTRL"
    # shortcut: Ctrl+T goes to whatever is in front, not only a browser; check the app first if that misfires.
    if os in _BROWSER_OS and (match := _NEW_TAB.match(text)):
        return [Action("hotkey", keys=(modifier, "T"))], match.end()
    if launched and os in _BROWSER_OS and (match := _ADDRESS.match(text)):
        return [Action("hotkey", keys=(modifier, "L")), Action("type_text", text=match[1]),
                Action("hotkey", keys=("DELETE",)), Action("hotkey", keys=("ENTER",))], match.end()
    if (match := _GO.match(text)) and len(match[1].split()) <= 3 and not _NOT_APP & set(match[1].lower().split()):
        if os in OPEN_APP_KEYS and _named(match[1], INSTALLED):
            return [Action("skill", name="open_app", args={"name": match[1]})], match.end()
        if launched and os in _BROWSER_OS and match[0][:2].lower() == "go":
            # No Delete: the browser's own completion ("youtube" -> youtube.com) picks a site it knows; else it searches.
            return [Action("hotkey", keys=(modifier, "L")), Action("type_text", text=match[1]),
                    Action("hotkey", keys=("ENTER",))], match.end()
    # Only as the last words: "search youtube and search cats" means searching inside YouTube.
    if launched and os in _BROWSER_OS and (match := _SEARCH.match(text)) and not text[match.end():].strip(" .!?") \
            and not _WHERE & {w.lower() for w in match[1].split()}:
        return [Action("hotkey", keys=(modifier, "L")), Action("type_text", text=match[1]),
                Action("hotkey", keys=("DELETE",)), Action("hotkey", keys=("ENTER",))], match.end()
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
    elif (match := _SCROLL.match(text)):
        return [Action("scroll", dy=-5 if match[1].lower() == "up" else 5)], match.end()
    return [], 0


def spoken_commands(text: str) -> list[tuple[str, str]]:
    """The commands at the start of speech still coming in that later words can no longer
    change, each as (its words, the speech after it): a keyboard-only quick command with
    more speech after it. Model work waits for the pause, which can change its meaning."""
    found, rest = [], text.strip()
    while True:
        # Any computer: which commands count only decides what runs at once; the agent knows the real OS.
        clause, end = _clause(rest, "linux", launched=False)
        if not clause or not rest[end:].strip() or clause[0].type == "scroll" or \
                (clause[0].type == "hotkey" and set(clause[0].keys) <= MODIFIERS):
            return found  # a scroll lands under the pointer, which is still on the talk button
        after = _SEPARATOR.sub("", rest[end:], count=1)
        found.append((rest[:end], after))
        rest = after


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
