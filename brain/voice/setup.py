"""The spoken first-time setup (issue #30): a few short questions, all answered by voice.

It starts on "set up" (brain.voice.commands), and by itself on a first run (brain.voice.settings). The steps:
a welcome that teaches holding the button, the talking speed, a name for this computer, and how much the box talks;
then a one-line summary. Anywhere in it, "repeat that" asks the step's question again, "go back" returns to the step
before, "skip" leaves the step as it is, and "stop setup" ends it keeping what was already chosen. Each choice is
saved as it is made, and nothing said during setup reaches the agent.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import IntEnum

from brain.voice.commands import MAX_SPEED, MIN_SPEED, STEP, Repeat, Speed, normalize, parse
from brain.voice.settings import Settings, Talk


class Step(IntEnum):
    WELCOME = 0
    SPEED = 1
    NAME = 2
    TALK = 3


QUESTIONS = {
    Step.WELCOME: "Hi, I'm Exospace. Let's set me up, one short question at a time. To talk to me, hold the button, "
                  "speak, then let go. Try it now: hold the button and say anything.",
    Step.SPEED: "How fast should I talk? Say faster, slower, or that's good.",
    Step.NAME: "What should I call this computer? For example, work laptop.",
    Step.TALK: "How much should I talk while I work? Say every step, just results, or only when done.",
}
CHANGE_HINT = "You can change any of this by asking, like talk faster."
STOPPED = "Setup stopped. I kept what you chose. Say set up any time to start again."
TALK_WORDS: dict[Talk, str] = {"steps": "every step", "results": "just the results", "done": "only when I'm done"}

_BACK = {"go back", "back", "previous", "previous question", "go back one"}
_SKIP = {"skip", "skip that", "skip this", "skip it", "skip this one", "skip this step", "next", "next question"}
_STOP = {f"{verb} {setup}" for verb in ("stop", "cancel", "end", "exit", "quit", "leave")
         for setup in ("setup", "set up", "the setup")}
_GOOD = {"that s good", "that is good", "good", "that s fine", "fine", "that s great", "great", "perfect", "ok",
         "okay", "yes", "sounds good", "that s right", "keep it", "keep it there", "just right"}
_BARE_SPEED = {**{word: Speed(factor=STEP) for word in ("faster", "quicker", "a bit faster", "a little faster")},
               **{word: Speed(factor=1 / STEP) for word in ("slower", "more slowly", "a bit slower",
                                                            "a little slower")}}
_NAME_LEAD = re.compile(r"^(?:call it|call this|name it|it[’']?s|it is|this is)\s+", re.IGNORECASE)


@dataclass(frozen=True)
class Turn:
    """What setup says back, the settings after this answer, and whether setup is over."""

    say: str
    settings: Settings
    over: bool = False


def speed_words(speed: float) -> str:
    return "normal speed" if speed == 1.0 else "half speed" if speed == 0.5 else f"{speed:g} times speed"


def summary(settings: Settings) -> str:
    """The last line of setup: what was chosen, and how to change it later."""
    computer = f", this computer is {settings.computer}" if settings.computer else ""
    return (f"All set. I talk at {speed_words(settings.speed)}{computer}, and I'll tell you "
            f"{TALK_WORDS[settings.talk]}. {CHANGE_HINT}")


def computer_name(words: str) -> str:
    """A spoken name tidied for keeping: "Call it work laptop." becomes "Work laptop"."""
    name = _NAME_LEAD.sub("", " ".join(words.split()).strip(" .,!?"))[:40].strip()
    return name[:1].upper() + name[1:]


def talk_level(text: str) -> Talk | None:
    """The talk level a normalized answer picks, or None. Answers are short, so a key word is enough."""
    if "result" in text:
        return "results"
    if re.search(r"\b(?:every|each|all|step|steps|everything)\b", text):
        return "steps"
    if re.search(r"\b(?:done|finished|finish|end|only)\b", text):
        return "done"
    return None


class Setup:
    """One run through the setup steps; `handle` takes each answer in turn. `name_computer` keeps the name given
    for this computer (its profile, brain.voice.profiles)."""

    def __init__(self, name_computer: Callable[[str], None]) -> None:
        self.step = Step.WELCOME
        self.name_computer = name_computer

    def question(self) -> str:
        return QUESTIONS[self.step]

    def handle(self, words: str, current: Settings) -> Turn:
        text = normalize(words)
        if text in _STOP:
            return Turn(STOPPED, current, over=True)
        if isinstance(parse(words), Repeat):
            return Turn(self.question(), current)
        if text in _BACK:
            self.step = Step(max(Step.WELCOME, self.step - 1))
            return Turn(self.question(), current)
        if text in _SKIP:
            return self._next("", current)
        if self.step == Step.WELCOME:
            return self._next("I heard you.", current)
        if self.step == Step.SPEED:
            return self._speed(words, text, current)
        if self.step == Step.NAME:
            name = computer_name(words)
            if not name:
                return Turn(self.question(), current)
            self.name_computer(name)
            return self._next(f"I'll call this computer {name}.", replace(current, computer=name))
        level = talk_level(text)
        if level is None:
            return Turn("Say every step, just results, or only when done.", current)
        return self._next(f"I'll tell you {TALK_WORDS[level]}.", replace(current, talk=level))

    def _speed(self, words: str, text: str, current: Settings) -> Turn:
        if text in _GOOD:
            return self._next("", current)
        command = _BARE_SPEED.get(text) or parse(words)
        if not isinstance(command, Speed):
            return Turn("Say faster, slower, or that's good.", current)
        speed = command.apply(current.speed)
        if speed == current.speed and speed in (MIN_SPEED, MAX_SPEED):
            said = f"That's already the {'fastest' if speed == MAX_SPEED else 'slowest'} I can talk."
        else:
            said = f"Now I talk at {speed_words(speed)}."
        return Turn(f"{said} Faster, slower, or that's good?", replace(current, speed=speed))

    def _next(self, said: str, settings: Settings) -> Turn:
        """`said`, then the next step's question, or the summary after the last step."""
        if self.step == Step.TALK:
            return Turn(" ".join(filter(None, [said, summary(settings)])), settings, over=True)
        self.step = Step(self.step + 1)
        return Turn(" ".join(filter(None, [said, self.question()])), settings)
