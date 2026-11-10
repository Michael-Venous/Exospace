"""Speech to text on the box (EXO_STT=local): the held audio heard whole on release, chosen only when asked for."""

import queue
import sys

import numpy as np
import pytest

from bench.arch.stt_local import errors, words
from brain.voice import local_stt
from brain.voice.local_stt import RATE, LocalUtterance, punctuate

CHUNK = RATE // 10 * 2  # 100 ms of 16-bit samples, as the microphone reads them


class FakeMic:
    """Gives `chunks` from the pre-roll on, then ends them on `stop()`, as talk.py's Mic does."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks, self.stopped = chunks, False

    def listen(self) -> queue.Queue:
        self.audio: queue.Queue = queue.Queue()
        for chunk in self.chunks:
            self.audio.put(chunk)
        return self.audio

    def stop(self) -> None:
        self.stopped = True
        self.audio.put(None)


class FakeModel:
    def __init__(self, words: str) -> None:
        self.words, self.heard = words, []

    def transcribe(self, samples: np.ndarray) -> str:
        self.heard.append(samples)
        return self.words


def press(chunks: list[bytes], words: str = "Open Firefox.", dictating: bool = False) -> tuple[str, FakeModel, list]:
    """One press and release with `chunks` held and the model hearing `words`: what finish() gave, the model, and
    every call to `heard`."""
    model, heard = FakeModel(words), []
    utterance = LocalUtterance(FakeMic(chunks), lambda *args: heard.append(args), dictating, lambda: model)
    return utterance.finish(), model, heard


def test_the_held_audio_is_heard_whole_on_release():
    loud = np.full(CHUNK // 2, 16384, dtype=np.int16).tobytes()
    words, model, heard = press([loud] * 5)
    assert words == "Open Firefox."
    assert len(model.heard) == 1
    samples = model.heard[0]
    assert samples.dtype == np.float32 and len(samples) == RATE // 2
    assert samples.max() == pytest.approx(0.5)
    assert heard == []  # no words before release: the caller gets them from finish()


def test_a_tap_or_nothing_is_no_words_and_never_reaches_the_model():
    for chunks in ([], [bytes(CHUNK)] * 2):  # nothing, and 0.2 s
        words, model, _ = press(chunks)
        assert words == "" and model.heard == []


def test_finish_stops_the_microphone():
    mic = FakeMic([bytes(CHUNK)] * 5)
    LocalUtterance(mic, lambda *args: None, model=lambda: FakeModel("hi")).finish()
    assert mic.stopped


def test_spoken_punctuation_is_marked_only_while_dictating():
    chunks = [bytes(CHUNK)] * 5
    said = "Dear Janet, comma, thanks period."
    assert press(chunks, said, dictating=True)[0] == "Dear Janet, thanks."
    assert press(chunks, said)[0] == said


@pytest.mark.parametrize("words, typed", [
    ("Dear Janet comma", "Dear Janet,"),
    ("Dear Janet, comma.", "Dear Janet,"),
    ("thanks period see you", "thanks. see you"),
    ("It ended. Full stop.", "It ended."),
    ("Can you come question mark", "Can you come?"),
    ("Great exclamation point", "Great!"),
    ("first line new line second line", "first line second line"),  # never a typed Enter
    ("the periodic table", "the periodic table"),  # only whole words
    ("no punctuation here", "no punctuation here"),
])
def test_punctuate(words, typed):
    assert punctuate(words) == typed


def test_deepgram_is_the_default_and_local_is_opt_in(monkeypatch):
    mic = FakeMic([])
    deepgram = object()
    monkeypatch.delenv("EXO_STT", raising=False)
    assert local_stt.utterance(mic, lambda *args: None, False, lambda: deepgram) is deepgram
    monkeypatch.setenv("EXO_STT", "deepgram")
    assert local_stt.utterance(mic, lambda *args: None, False, lambda: deepgram) is deepgram
    monkeypatch.setenv("EXO_STT", "local")
    local = local_stt.utterance(mic, lambda *args: None, True, lambda: deepgram)
    assert isinstance(local, LocalUtterance) and local.dictating
    monkeypatch.setenv("EXO_STT", "whisper")
    with pytest.raises(SystemExit):
        local_stt.chosen()


def test_without_faster_whisper_the_local_path_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # as if not installed
    with pytest.raises(RuntimeError, match="local-speech"):
        local_stt.Whisper()


def test_the_benchmark_counts_word_errors_without_case_or_punctuation():
    assert words("Search YouTube for lo-fi music.") == ["search", "youtube", "for", "lo", "fi", "music"]
    assert errors(words("pay my power bill"), words("Pay my parable.")) == 2  # one swapped, one dropped
    assert errors(["talk", "faster"], ["took", "faster", "took", "faster"]) == 3
    assert errors([], ["hi"]) == 1
