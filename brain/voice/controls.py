"""What each finished utterance does, shared by every voice front end: a box command is answered here, out loud,
and anything else goes to the agent as its goal. Also remembers the last line spoken, for "repeat that".

In dictation mode ("start dictating" ... "stop dictating") each utterance becomes a goal that has the agent type
exactly those words at the caret, and the words are read back (a blind user can't see what was typed), unless echo
is turned off.

While the spoken setup runs ("set up", or by itself on a first run: brain.voice.setup) every utterance is an answer
to it, and none reaches the agent; so too while the box asks which computer it is plugged into (brain.voice.plug_in,
when it starts on a screen its computer profiles don't place for sure).

Private mode ("I'll type my password", issue #27): the agent never sees or types a password, so the user types it
and the box stops the agent and ignores every utterance until "done" (the agent continues the request it was on)
or "cancel" (nothing is sent). Those utterances never reach the agent, and front ends print `shown(words)`, never
the words themselves.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from brain.types import Frame
from brain.voice import profiles, settings
from brain.voice.commands import Dictate, Echo, Private, Repeat, SetUp, Speed, leave_private, parse, speed_line
from brain.voice.plug_in import WhichComputer
from brain.voice.setup import Setup, Turn

# Deepgram's dictation mode turns spoken "comma" and "period" into punctuation; it needs punctuate on.
DEEPGRAM_DICTATION = "&punctuate=true&dictation=true"
START_LINE = "Dictating. Say stop dictating when you're done."
PRIVATE_LINE = "I'm not looking or listening. Type it, then press the button and say done."
STILL_PRIVATE_LINE = "I'm still not looking. Say done when you've typed it, or cancel."
CANCEL_PRIVATE_LINE = "Okay, I'm listening again."
RESUME_GOAL = ("The user finished typing their password themselves. Continue the previous request from the current "
               "screen; don't read or repeat what's in the password field.")
PRIVATE_SHOWN = "(private)"


@dataclass(frozen=True)
class Reply:
    """`goal` goes to the agent (None: nothing does); `say` is spoken by the box itself; `stop` stops what the
    agent is doing now (front ends whose "nothing heard" path already stops it need not look)."""

    goal: str | None = None
    say: str | None = None
    stop: bool = False


def dictation_goal(words: str, first: bool) -> str:
    """The agent's goal for one dictated utterance. Later utterances start with a space, so sentences don't run
    together."""
    space = "" if first else "a space, then "
    return (f"Dictation: type {space}exactly the words after the colon at the caret, in the field already in front, "
            f"and nothing else; no clicks and no other keys: {words}")


class Dialog(Protocol):
    """A spoken exchange that takes every utterance until it is over: the setup, or which computer this is."""

    def question(self) -> str: ...

    def handle(self, words: str, current: settings.Settings) -> Turn: ...


class Controls:
    def __init__(self, path: Path = settings.PATH, first_run_setup: bool = False,
                 frame: Callable[[], Frame | None] = lambda: None) -> None:
        """With `first_run_setup`, a missing settings file starts the spoken setup (front ends pass
        settings.FIRST_RUN_SETUP; tests leave it off). `frame()` is the target's screen now, or None: it
        fingerprints the computer profiles, kept next to the settings file."""
        self.path, self.frame = path, frame
        self.profiles = path.with_name(profiles.PATH.name)
        self.dialog: Dialog | None = self._setup() if first_run_setup and not path.exists() else None
        self.settings = settings.load(path)
        self.last = ""  # the last line spoken
        self.dictating = False
        self.dictated = 0  # utterances typed since dictation started
        self.private = False  # the user is typing a password: nothing reaches the agent, nothing is shown

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
        """What a front end says as it starts: setup's first question on a first run, else which computer this is
        when its screen matches no profile for sure; nothing for a known computer."""
        self.dialog = self.dialog or self._which_computer()
        return self.dialog.question() if self.dialog else None

    def shown(self, words: str) -> str:
        """The words as a front end may print or log them: in private mode, only "done" or "cancel"."""
        return PRIVATE_SHOWN if self.private and words and leave_private(words) is None else words

    def handle(self, words: str) -> Reply:
        if self.private:
            return self._private(words)
        if self.dialog:
            return self._answer(self.dialog, words)
        command = parse(words)
        if isinstance(command, SetUp):
            setup = self._setup()
            self.dialog, self.dictating = setup, False
            return Reply(say=setup.question())
        if isinstance(command, Private):
            self.private = True
            return Reply(say=PRIVATE_LINE, stop=True)
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

    def _private(self, words: str) -> Reply:
        """One utterance in private mode: "done" hands the request back to the agent, "cancel" sends nothing, and
        anything else is dropped unheard."""
        leave = leave_private(words)
        if leave is None:
            return Reply(say=STILL_PRIVATE_LINE)
        self.private = False
        return Reply(goal=RESUME_GOAL) if leave == "done" else Reply(say=CANCEL_PRIVATE_LINE)

    def _answer(self, dialog: Dialog, words: str) -> Reply:
        """One answer to a dialog. Its end is saved even with nothing changed, so a first run is over."""
        turn = dialog.handle(words, self.settings)
        if turn.settings != self.settings or turn.over:
            self.settings = turn.settings
            settings.save(self.settings, self.path)
        if turn.over:
            self.dialog = None
        return Reply(say=turn.say)

    def _setup(self) -> Setup:
        return Setup(self._name_computer)

    def _name_computer(self, name: str) -> None:
        """The name setup heard becomes this computer's profile, with its screen's fingerprint when there is one."""
        profiles.save(profiles.remember(profiles.load(self.profiles), name, self.frame()), self.profiles)

    def _which_computer(self) -> WhichComputer | None:
        """The computer on the screen now: a sure match quietly becomes the active profile (its fingerprint
        refreshed); an unsure one is asked about. With no profiles or no frame there is nothing to ask."""
        store = profiles.load(self.profiles)
        frame = self.frame() if store.profiles else None
        if frame is None:
            return None
        found = profiles.match(store.profiles, frame)
        if found.kind != "match" or found.profile is None:
            return WhichComputer(found, frame, self.profiles)
        profiles.save(profiles.remember(store, found.profile.name, frame), self.profiles)
        self._save(computer=found.profile.name)
        return None

    def _save(self, **changes: float | bool | str) -> None:
        self.settings = replace(self.settings, **changes)
        settings.save(self.settings, self.path)
