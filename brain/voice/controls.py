"""What each finished utterance does, shared by every voice front end: a box command is answered here, out loud,
and anything else goes to the agent as its goal. Also remembers the last line spoken, for "repeat that"."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from brain.voice import settings
from brain.voice.commands import Repeat, Speed, parse, speed_line


@dataclass(frozen=True)
class Reply:
    """`goal` goes to the agent (None: nothing does); `say` is spoken by the box itself."""

    goal: str | None = None
    say: str | None = None


class Controls:
    def __init__(self, path: Path = settings.PATH) -> None:
        self.path = path
        self.settings = settings.load(path)
        self.last = ""  # the last line spoken

    @property
    def speed(self) -> float:
        return self.settings.speed

    def said(self, line: str) -> None:
        """A line the front end speaks: what "repeat that" says next."""
        self.last = line

    def handle(self, words: str) -> Reply:
        command = parse(words)
        if isinstance(command, Speed):
            before = self.settings.speed
            self.settings = replace(self.settings, speed=command.apply(before))
            settings.save(self.settings, self.path)
            return Reply(say=speed_line(self.settings.speed, before))
        if isinstance(command, Repeat):
            return Reply(say=self.last or "I haven't said anything yet.")
        return Reply(goal=words)
