"""Keyboard first: a request that is one common command ("close this app", "new tab", "go back") is
pressed as that OS's shortcut at once, with no model aiming at the screen. Clicking a 12-pixel close
button is slower (a Jev look, a zoom, sometimes a hover check) and can miss; the keys can't. Jev's
completion check still confirms the result, and anything it doesn't confirm goes to the planner.

Only whole requests match: "close this app and open Safari" is planned as usual.
"""

from __future__ import annotations

import re

from brain.types import OSName

_MAC = ("macos", "ios")
_PHONES = ("ios", "android")
_POLITE = re.compile(r"^(?:(?:hey|ok|okay)\s+exo\w*,?\s+|please\s+|can you\s+|could you\s+|now\s+)+")
_TRAILING = re.compile(r"[\s,.!?]*(?:\s+please)?[\s,.!?]*$")


def _keys(mac: tuple[str, ...] | None, other: tuple[str, ...] | None, *, phones: tuple[str, ...] | None = None,
          only: tuple[str, ...] = ()) -> dict[str, tuple[str, ...]]:
    table: dict[str, tuple[str, ...]] = {}
    for os in ("macos", "ios", "windows", "linux", "chromeos", "android"):
        keys = phones if os in _PHONES and phones is not None else mac if os in _MAC else other
        if keys and (not only or os in only):
            table[os] = keys
    return table


# (label, whole-request pattern, keys per OS). A missing OS means no safe shortcut there: plan as usual.
COMMANDS: list[tuple[str, re.Pattern[str], dict[str, tuple[str, ...]]]] = [
    ("close the app", re.compile(r"(?:close|quit|exit)(?: out of)?(?: (?:this|the|current))? (?:app|application|program)"
                                 r"|quit(?: (?:this|it))?"),
     _keys(("GUI", "Q"), ("ALT", "F4"), phones=("GUI", "H"), only=("macos", "windows", "linux", "ios"))),
    ("close the tab", re.compile(r"close(?: (?:this|the|current))? tab"), _keys(("GUI", "W"), ("CTRL", "W"),
                                                                                   only=("macos", "windows", "linux", "chromeos"))),
    ("close the window", re.compile(r"close(?: (?:this|the|current))? window"),
     _keys(("GUI", "W"), ("ALT", "F4"), only=("macos", "windows", "linux"))),
    ("open a new tab", re.compile(r"(?:open )?(?:a )?new tab"), _keys(("GUI", "T"), ("CTRL", "T"),
                                                                    only=("macos", "windows", "linux", "chromeos"))),
    ("reopen the closed tab", re.compile(r"(?:re-?open|bring back)(?: the)?(?: last)?(?: closed)? tab"),
     _keys(("GUI", "SHIFT", "T"), ("CTRL", "SHIFT", "T"), only=("macos", "windows", "linux", "chromeos"))),
    ("go back", re.compile(r"(?:go )?back"), _keys(("GUI", "LEFT"), ("ALT", "LEFT"), phones=("ESC",),
                                                     only=("macos", "windows", "linux", "chromeos", "android"))),
    ("refresh the page", re.compile(r"(?:refresh|reload)(?: (?:this|the))?(?: page)?"),
     _keys(("GUI", "R"), ("F5",), only=("macos", "windows", "linux", "chromeos"))),
    ("undo", re.compile(r"undo(?: (?:that|it|this))?"), _keys(("GUI", "Z"), ("CTRL", "Z"))),
    ("redo", re.compile(r"redo(?: (?:that|it|this))?"), _keys(("GUI", "SHIFT", "Z"), ("CTRL", "Y"))),
    ("save", re.compile(r"save(?: (?:this|it|the file|the document))?"), _keys(("GUI", "S"), ("CTRL", "S"),
                                                                                 only=("macos", "windows", "linux", "chromeos"))),
    ("select all", re.compile(r"select (?:all|everything)"), _keys(("GUI", "A"), ("CTRL", "A"))),
    ("find on the page", re.compile(r"find (?:on|in) (?:the |this )?page"), _keys(("GUI", "F"), ("CTRL", "F"),
                                                                                 only=("macos", "windows", "linux", "chromeos"))),
    ("start something new", re.compile(r"(?:start |create |make |open )?(?:a )?new (?:message|text|note|document|email|mail)"),
     _keys(("GUI", "N"), ("CTRL", "N"))),
    ("switch apps", re.compile(r"switch (?:apps|app|windows|to the (?:last|previous) (?:app|window))"),
     _keys(("GUI", "TAB"), ("ALT", "TAB"), only=("macos", "windows", "linux", "chromeos"))),
    ("minimize the window", re.compile(r"minimi[sz]e(?: (?:this|the|it))?(?: (?:window|app))?"),
     _keys(("GUI", "M"), ("GUI", "DOWN"), only=("macos", "windows"))),
    ("go to the home screen", re.compile(r"(?:go )?(?:to )?(?:the )?home(?: screen)?|go home"),
     _keys(None, None, phones=("GUI", "H"), only=_PHONES)),
]


def normalize(utterance: str) -> str:
    text = re.sub(r"\s+", " ", utterance.strip().lower())
    return _TRAILING.sub("", _POLITE.sub("", text))


def shortcut_for(utterance: str, os: OSName) -> tuple[str, tuple[str, ...]] | None:
    """(label, keys) when the whole request is one common command with a shortcut on this OS."""
    text = normalize(utterance)
    for label, pattern, table in COMMANDS:
        if os in table and pattern.fullmatch(text):
            return label, table[os]
    return None
