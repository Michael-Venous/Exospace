"""Speech to text on the box itself, opt in with EXO_STT=local: no Deepgram key, no audio leaving the box.

Deepgram stays the default (EXO_STT=deepgram, or unset). A LocalUtterance has the same shape as the Deepgram one
(bench/arch/talk.py's Utterance): it starts on a press, keeps the microphone's chunks while the key is held, and
`finish()` on release gives the words. The difference is when they come: faster-whisper (tiny.en, int8, on the CPU)
hears the whole utterance at once on release, so the agent gets no words while the user is still speaking.

faster-whisper is an optional extra (`pip install -e '.[local-speech]'`), imported only when this path is chosen.
Its model is loaded once per process and shared by every utterance. Latency and accuracy per model:
bench/arch/stt_local.py.
"""

from __future__ import annotations

import os
import queue
import re
from collections.abc import Callable
from functools import cache
from typing import Protocol, TypeVar

import numpy as np

RATE = 16000  # the microphone's 16 kHz, 16-bit mono (bench/arch/talk.py's Mic)
MIN_S = 0.3  # shorter than this is a tap of the key, not speech
MODEL = "tiny.en"
ENGINES = ("deepgram", "local")

# Spoken punctuation, for dictation: Deepgram's dictation mode does this itself, Whisper doesn't. Whisper adds its
# own commas and periods around the spoken word ("Dear Janet, comma."), so those go with it.
MARKS = {"comma": ",", "period": ".", "full stop": ".", "question mark": "?", "exclamation point": "!",
         "exclamation mark": "!",
         # A line break is left out: typing Enter in a field can send it, and the agent's confirm gate would stop
         # to ask about every dictated line. The words "new line" are dropped instead of typed.
         "new line": ""}
SPOKEN = re.compile(r"[\s,.!?]*\b(" + "|".join(name.replace(" ", r"\s+") for name in MARKS) + r")\b[,.!?]*",
                    re.IGNORECASE)


class Mic(Protocol):
    """The always-open microphone: `listen()` gives its chunks from the pre-roll on, `stop()` ends them with None."""

    def listen(self) -> queue.Queue: ...

    def stop(self) -> None: ...


class Model(Protocol):
    def transcribe(self, samples: np.ndarray) -> str:
        """The words in `samples` (16 kHz mono floats in [-1, 1])."""
        ...


class Whisper:
    """faster-whisper on the CPU: int8, greedy (beam 1), English only."""

    def __init__(self, name: str = MODEL) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError("EXO_STT=local needs faster-whisper: pip install -e '.[local-speech]'") from None
        self.model = WhisperModel(name, device="cpu", compute_type="int8")

    def transcribe(self, samples: np.ndarray) -> str:
        # The voice filter drops silence and noise first: Whisper is known to make up words ("Thank you.") from them.
        # It cost no accuracy or time on bench/arch/stt_local.py's clips.
        segments, _ = self.model.transcribe(samples, beam_size=1, language="en", vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments).strip()


@cache
def whisper() -> Model:
    """The shared Whisper model, loaded on first use (front ends call this at start, so the first press is quick)."""
    return Whisper()


def chosen() -> bool:
    """Whether EXO_STT asks for speech to text on the box; anything but its two values is a mistake, said so."""
    engine = os.environ.get("EXO_STT", "deepgram")
    if engine not in ENGINES:
        raise SystemExit(f"EXO_STT must be one of {', '.join(ENGINES)}, not {engine!r}")
    return engine == "local"


def punctuate(words: str) -> str:
    """Dictation's spoken punctuation as marks: "Dear Janet comma" -> "Dear Janet,"."""
    marked = SPOKEN.sub(lambda match: MARKS[" ".join(match.group(1).lower().split())] + " ", words)
    return re.sub(r"\s+", " ", marked).strip()


class LocalUtterance:
    """One press of the talk key, heard on the box. `heard` is never called: there are no words until release, and
    `finish()` returns them, as the Deepgram utterance's caller already expects."""

    def __init__(self, mic: Mic, heard: Callable[[str, bool], None], dictating: bool = False,
                 model: Callable[[], Model] = whisper) -> None:
        self.mic, self.dictating, self.model = mic, dictating, model
        self.audio = mic.listen()

    def finish(self) -> str:
        """Let go of the key: everything said since the pre-roll, transcribed now; "" for a tap."""
        self.mic.stop()
        chunks = []
        while (chunk := self.audio.get()) is not None:
            chunks.append(chunk)
        sound = b"".join(chunks)
        if len(sound) < MIN_S * RATE * 2:
            return ""
        words = self.model().transcribe(np.frombuffer(sound, dtype=np.int16).astype(np.float32) / 32768)
        return punctuate(words) if self.dictating else words


Utterance = TypeVar("Utterance")


def utterance(mic: Mic, heard: Callable[[str, bool], None], dictating: bool,
              deepgram: Callable[[], Utterance]) -> Utterance | LocalUtterance:
    """The next press's utterance: on the box with EXO_STT=local, else Deepgram's, made by `deepgram()`."""
    return LocalUtterance(mic, heard, dictating) if chosen() else deepgram()
