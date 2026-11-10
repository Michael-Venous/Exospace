"""What each finished utterance does, shared by every voice front end: a box command is answered here, out loud,
and anything else goes to the agent as its goal. Also remembers the last line spoken, for "repeat that".

In dictation mode ("start dictating" ... "stop dictating") each utterance becomes a goal that has the agent type
exactly those words at the caret, and the words are read back (a blind user can't see what was typed), unless echo
is turned off.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from brain.voice import settings
from brain.voice.commands import Dictate, Echo, Repeat, Speed, parse, speed_line

# Deepgram's dictation mode turns spoken "comma" and "period" into punctuation; it needs punctuate on.
DEEPGRAM_DICTATION = "&punctuate=true&dictation=true"
START_LINE = "Dictating. Say stop dictating when you're done."


@dataclass(frozen=True)
class Reply:
    """`goal` goes to the agent (None: nothing does); `say` is spoken by the box itself."""

    goal: str | None = None
    say: str | None = None


def dictation_goal(words: str, first: bool) -> str:
    """The agent's goal for one dictated utterance. Later utterances start with a space, so sentences don't run
    together."""
    space = "" if first else "a space, then "
    return (f"Dictation: type {space}exactly the words after the colon at the caret, in the field already in front, "
            f"and nothing else; no clicks and no other keys: {words}")


class Controls:
    def __init__(self, path: Path = settings.PATH) -> None:
        self.path = path
        self.settings = settings.load(path)
        self.last = ""  # the last line spoken
        self.dictating = False
        self.dictated = 0  # utterances typed since dictation started

    @property
    def speed(self) -> float:
        return self.settings.speed

    def deepgram(self, url: str) -> str:
        """The Deepgram URL for the next utterance: with spoken punctuation while dictating."""
        return url + DEEPGRAM_DICTATION if self.dictating else url

    def said(self, line: str) -> None:
        """A line the front end speaks: what "repeat that" says next."""
        self.last = line

    def handle(self, words: str) -> Reply:
        command = parse(words)
        if isinstance(command, Speed):
            before = self.settings.speed
            self._save(speed=command.apply(before))
            return Reply(say=speed_line(self.settings.speed, before))
        if isinstance(command, Repeat):
            return Reply(say=self.last or "I haven't said anything yet.")
        if isinstance(command, Dictate):
            was, self.dictating, self.dictated = self.dictating, command.on, 0
            if command.on:
                return Reply(say=START_LINE)
            return Reply(say="Stopped dictating." if was else "I wasn't dictating.")
        if isinstance(command, Echo):
            self._save(echo=command.on)
            return Reply(say="I'll read back what you dictate." if command.on
                         else "I won't read back what you dictate.")
        if self.dictating:
            self.dictated += 1
            return Reply(goal=dictation_goal(words, self.dictated == 1), say=words if self.settings.echo else None)
        return Reply(goal=words)

    def _save(self, **changes: float | bool) -> None:
        self.settings = replace(self.settings, **changes)
        settings.save(self.settings, self.path)
