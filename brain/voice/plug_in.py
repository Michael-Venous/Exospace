"""Which computer is this? Asked aloud when the box starts on a screen its profiles don't place for sure
(brain.voice.profiles): "Is this a new computer?" for a screen like none of them, "Is this Work laptop or Home PC?"
when it could be either. A new computer is then named; a known one has its fingerprint refreshed from this screen.
"Repeat that" asks again, and "never mind" leaves everything as it was. Nothing said here reaches the agent.
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum, auto
from pathlib import Path

from brain.types import Frame
from brain.voice import profiles
from brain.voice.commands import Repeat, normalize, parse
from brain.voice.settings import Settings
from brain.voice.setup import Turn, computer_name

_YES = {"yes", "yeah", "yep", "yes it is", "it is", "correct", "right", "sure", "new", "it s new", "a new one",
        "yes it s new"}
_NO = {"no", "nope", "no it s not", "no it isn t", "it isn t", "not new", "no it s not new"}
_NEITHER = {"neither", "neither of them", "none", "none of them", "new", "a new one", "a new computer", "another one",
            "something else", "no"}
_LEAVE = {"never mind", "nevermind", "cancel", "stop", "skip", "skip it", "forget it", "not now"}
NAME_QUESTION = "What should I call it?"


class Ask(Enum):
    NEW = auto()  # is this a new computer?
    WHICH = auto()  # is this A or B?
    NAME = auto()  # what should I call it?


def either(names: list[str]) -> str:
    """Names joined for a question: A, A or B, A, B or C."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} or {names[-1]}"


class WhichComputer:
    """One round of asking which computer the box is plugged into; `frame` is the screen it is showing."""

    def __init__(self, found: profiles.Match, frame: Frame, path: Path = profiles.PATH) -> None:
        self.frame, self.path = frame, path
        self.store = profiles.load(path)
        self.choices = list(found.candidates) or list(self.store.profiles)
        self.ask = Ask.WHICH if found.kind == "ambiguous" else Ask.NEW

    def question(self) -> str:
        if self.ask == Ask.NEW:
            return "Is this a new computer?"
        if self.ask == Ask.WHICH:
            return f"Is this {either([p.name for p in self.choices])}?"
        return NAME_QUESTION

    def handle(self, words: str, current: Settings) -> Turn:
        text = normalize(words)
        if text in _LEAVE:
            return Turn("OK, I'll leave it.", current, over=True)
        if isinstance(parse(words), Repeat):
            return Turn(self.question(), current)
        if self.ask == Ask.NEW:
            return self._new(text, current)
        if self.ask == Ask.WHICH:
            return self._which(text, current)
        name = computer_name(words)
        if not name:
            return Turn(NAME_QUESTION, current)
        return self._choose(name, f"I'll call this computer {name}.", current)

    def _new(self, text: str, current: Settings) -> Turn:
        if text in _YES:
            self.ask = Ask.NAME
            return Turn(NAME_QUESTION, current)
        if text not in _NO:
            return Turn("Say yes or no. Is this a new computer?", current)
        if not self.choices:
            self.ask = Ask.NAME
            return Turn(f"I don't know any other computers yet. {NAME_QUESTION}", current)
        self.ask = Ask.WHICH
        return Turn(self.question(), current)

    def _which(self, text: str, current: Settings) -> Turn:
        named = [p for p in self.choices if f" {normalize(p.name)} " in f" {text} "]
        if named:
            name = max(named, key=lambda p: len(p.name)).name  # "Work laptop two" over "Work laptop"
            return self._choose(name, f"OK, this is {name}.", current)
        if text in _NEITHER:
            self.ask = Ask.NAME
            return Turn(f"Then it's a new one. {NAME_QUESTION}", current)
        return Turn(f"Say {either([p.name for p in self.choices] + ['new'])}.", current)

    def _choose(self, name: str, said: str, current: Settings) -> Turn:
        """`name` is this computer: it becomes the active profile, with this screen as its fingerprint."""
        profiles.save(profiles.remember(self.store, name, self.frame), self.path)
        return Turn(said, replace(current, computer=name), over=True)
