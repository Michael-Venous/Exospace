"""What each finished utterance does, shared by every voice front end: a box command is answered here, out loud,
and anything else goes to the agent as its goal. Also remembers the last line spoken, for "repeat that".

In dictation mode ("start dictating" ... "stop dictating") each utterance becomes a goal that has the agent type
exactly those words at the caret, and the words are read back (a blind user can't see what was typed), unless echo
is turned off.

While the spoken setup runs ("set up", or by itself on a first run: brain.voice.setup) every utterance is an answer
to it, and none reaches the agent.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from brain.voice import settings
from brain.voice.commands import Dictate, Echo, Repeat, SetUp, Speed, parse, speed_line
from brain.voice.setup import Setup

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
    def __init__(self, path: Path = settings.PATH, first_run_setup: bool = False) -> None:
        """With `first_run_setup`, a missing settings file starts the spoken setup (front ends pass
        settings.FIRST_RUN_SETUP; tests leave it off)."""
        self.path = path
        self.setup = Setup() if first_run_setup and not path.exists() else None
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

    def greeting(self) -> str | None:
        """What a front end says as it starts: setup's first question on a first run, else nothing."""
        return self.setup.question() if self.setup else None

    def handle(self, words: str) -> Reply:
        if self.setup:
            return self._answer(self.setup, words)
        command = parse(words)
        if isinstance(command, SetUp):
            self.setup, self.dictating = Setup(), False
            return Reply(say=self.setup.question())
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

    def _answer(self, setup: Setup, words: str) -> Reply:
        """One answer to the spoken setup. Its end is saved even with nothing changed, so a first run is over."""
        turn = setup.handle(words, self.settings)
        if turn.settings != self.settings or turn.over:
            self.settings = turn.settings
            settings.save(self.settings, self.path)
        if turn.over:
            self.setup = None
        return Reply(say=turn.say)

    def _save(self, **changes: float | bool) -> None:
        self.settings = replace(self.settings, **changes)
        settings.save(self.settings, self.path)
