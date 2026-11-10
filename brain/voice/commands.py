"""Spoken commands meant for the box itself, not the agent: how fast it talks, "repeat that", dictation on and off,
whether dictated words are read back, and "set up" (the spoken setup, brain.voice.setup).

Only a whole utterance counts ("talk faster", "Talk faster, please."), so a request that merely contains the words
("type talk faster in the doc") still goes to the agent. Case and punctuation don't matter: Deepgram adds both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MIN_SPEED, MAX_SPEED = 0.5, 5.0
STEP = 1.25  # "faster" and "slower" multiply or divide the speed by this

_VERB = r"(?:talk|speak|read)"
_NUMBER_WORDS = {"one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0}
_NUMBER = r"(\d+(?:\.\d+)?|one|two|three|four|five)( and a half)?"
_TIMES = r"(?: ?x| times)"


@dataclass(frozen=True)
class Speed:
    """A new talking speed: an absolute `rate` (2.0 is twice normal), or a `factor` on the current one."""

    rate: float | None = None
    factor: float | None = None

    def apply(self, current: float) -> float:
        speed = self.rate if self.rate is not None else current * (self.factor or 1.0)
        return round(min(MAX_SPEED, max(MIN_SPEED, speed)), 2)


@dataclass(frozen=True)
class Repeat:
    """Say the last line again."""


@dataclass(frozen=True)
class Dictate:
    """Dictation mode on or off: while on, each utterance is typed where the caret is."""

    on: bool


@dataclass(frozen=True)
class Echo:
    """Read dictated words back aloud, or not."""

    on: bool


@dataclass(frozen=True)
class SetUp:
    """Start the spoken setup."""


Command = Speed | Repeat | Dictate | Echo | SetUp


def normalize(words: str) -> str:
    """Lower case, punctuation dropped (a decimal point between digits stays), a polite "please" or "can you"
    around the command dropped too."""
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", words.lower())
    text = " ".join(re.sub(r"[^a-z0-9. ]", " ", text).split())
    text = re.sub(r"^(?:please |(?:can|could) you (?:please )?)", "", text)
    return text.removesuffix(" please")


def _number(digits: str, half: str | None) -> float:
    return (_NUMBER_WORDS.get(digits) or float(digits)) + (0.5 if half else 0.0)


_FIXED: dict[str, Command] = {
    **{f"{verb} {word}": Speed(factor=STEP) for verb in ("talk", "speak", "read")
       for word in ("faster", "quicker", "a bit faster", "a little faster")},
    **{f"{verb} {word}": Speed(factor=1 / STEP) for verb in ("talk", "speak", "read")
       for word in ("slower", "more slowly", "a bit slower", "a little slower")},
    **{phrase: Speed(rate=1.0) for phrase in ("normal speed", "regular speed", "reset speed", "reset the speed",
                                              "talk normally", "speak normally", "talk at normal speed",
                                              "speak at normal speed", "read at normal speed")},
    **{phrase: Speed(rate=0.5) for phrase in ("half speed", "talk at half speed", "speak at half speed",
                                              "read at half speed")},
    **{phrase: Repeat() for phrase in ("repeat", "repeat that", "repeat it", "say that again", "say it again",
                                       "what did you say", "what did you just say", "come again")},
    **{phrase: Dictate(True) for phrase in ("start dictating", "start dictation", "begin dictating",
                                            "begin dictation")},
    **{phrase: Dictate(False) for phrase in ("stop dictating", "stop dictation", "end dictation",
                                             "done dictating", "i m done dictating", "finish dictating")},
    **{phrase: Echo(True) for phrase in ("echo on", "turn echo on", "turn on echo")},
    **{phrase: Echo(False) for phrase in ("echo off", "turn echo off", "turn off echo")},
    **{f"{verb}{space}up": SetUp() for verb in ("set", "start set", "run set") for space in (" ", "")},
}
_PATTERNS: list[tuple[re.Pattern[str], float]] = [  # (pattern, exponent on the number): slower means 1/n
    (re.compile(rf"{_VERB} at {_NUMBER}{_TIMES}(?: (?:the )?speed)?"), 1.0),
    (re.compile(rf"{_VERB} {_NUMBER}{_TIMES} (?:faster|as fast|speed)"), 1.0),
    (re.compile(rf"{_VERB} {_NUMBER}{_TIMES} slower"), -1.0),
    (re.compile(rf"(?:set )?(?:speech|voice|talking|speaking) speed (?:to )?{_NUMBER}{_TIMES}?"), 1.0),
]


def parse(words: str) -> Command | None:
    """The box command these words are, or None when they are a request for the agent."""
    text = normalize(words)
    if text in _FIXED:
        return _FIXED[text]
    for pattern, exponent in _PATTERNS:
        if match := pattern.fullmatch(text):
            number = _number(*match.groups())
            return Speed(rate=number ** exponent) if number > 0 else None
    return None


def speed_line(speed: float, before: float) -> str:
    """What the box says after a speed change."""
    if speed == before and speed in (MIN_SPEED, MAX_SPEED):
        return f"That's already the {'fastest' if speed == MAX_SPEED else 'slowest'} I can talk."
    if speed == 1.0:
        return "Speaking at normal speed."
    if speed == 0.5:
        return "Speaking at half speed."
    return f"Speaking at {speed:g} times speed."
