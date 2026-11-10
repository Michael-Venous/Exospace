"""Speech to text on the box (EXO_STT=local, brain/voice/local_stt.py): word error rate and latency per model.

Give it a folder of 16 kHz, 16-bit mono WAVs and what each one says, either as a `<clip>.txt` beside each WAV or as
a phrases file with one request per line and clips named `<anything>_<line>.wav` (1-based). Each model is loaded and
warmed up on the first clip, then times every clip on its own, as one release of the talk key would:

  PYTHONPATH=. .venv/bin/python -m bench.arch.stt_local recordings/ --phrases phrases.txt --show

Models: faster-whisper tiny.en and base.en (int8, CPU, beam 1, with the voice filter, as the box runs it), and
Moonshine tiny and base when `useful-moonshine-onnx` is installed. Words are compared lowercase, without
punctuation ("lo-fi" is "lo fi").

Measured so far only on a Mac (M5 Pro, 80 clips of synthetic macOS voices, issue #28): tiny.en 7.6% WER at 0.10 s
median, Moonshine tiny 12% at 0.035 s. The Pi 5's numbers, with real recordings from its microphone, still need
measuring with this script.
"""

from __future__ import annotations

import argparse
import re
import statistics
import time
import wave
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brain.voice.local_stt import RATE, Whisper

Transcribe = Callable[[np.ndarray], str]


@dataclass(frozen=True)
class Clip:
    name: str
    samples: np.ndarray
    reference: str


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", "", text.lower().replace("-", " ")).split()


def errors(reference: list[str], heard: list[str]) -> int:
    """Word substitutions, insertions and deletions turning `reference` into `heard` (edit distance)."""
    row = list(range(len(heard) + 1))
    for i, said in enumerate(reference, 1):
        previous, row[0] = row[0], i
        for j, got in enumerate(heard, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (said != got))
    return row[-1]


def read(path: Path) -> np.ndarray:
    with wave.open(str(path)) as sound:
        if (sound.getframerate(), sound.getsampwidth(), sound.getnchannels()) != (RATE, 2, 1):
            raise SystemExit(f"{path.name}: needs {RATE} Hz, 16-bit mono")
        return np.frombuffer(sound.readframes(sound.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def clips(folder: Path, phrases: Path | None) -> list[Clip]:
    lines = phrases.read_text().splitlines() if phrases else []
    found = []
    for path in sorted(folder.glob("*.wav")):
        reference = (lines[int(path.stem.rsplit("_", 1)[1]) - 1] if phrases
                     else path.with_suffix(".txt").read_text())
        found.append(Clip(path.name, read(path), reference.strip()))
    if not found:
        raise SystemExit(f"No WAVs in {folder}")
    return found


def models() -> Iterator[tuple[str, Callable[[], Transcribe]]]:
    """Each model's name and how to load it; Moonshine only when it is installed."""
    for name in ("tiny.en", "base.en"):
        yield f"faster-whisper {name}", lambda name=name: Whisper(name).transcribe
    try:
        import moonshine_onnx
    except ImportError:
        return
    for name in ("moonshine/tiny", "moonshine/base"):
        def load(name: str = name) -> Transcribe:
            model = moonshine_onnx.MoonshineOnnxModel(model_name=name)
            return lambda samples: " ".join(moonshine_onnx.transcribe(samples, model))
        yield name, load


def measure(name: str, load: Callable[[], Transcribe], found: list[Clip], show: bool) -> None:
    start = time.perf_counter()
    transcribe = load()
    transcribe(found[0].samples)  # the first call sets up the runtime: not part of any clip's time
    loaded = time.perf_counter() - start
    wrong = total = 0
    seconds = []
    for clip in found:
        start = time.perf_counter()
        heard = transcribe(clip.samples)
        seconds.append(time.perf_counter() - start)
        missed = errors(words(clip.reference), words(heard))
        wrong, total = wrong + missed, total + len(words(clip.reference))
        if show and missed:
            print(f"    {clip.name}: {clip.reference!r} -> {heard!r}")
    p90 = statistics.quantiles(seconds, n=10)[-1] if len(seconds) > 1 else seconds[0]
    print(f"{name}: {len(found)} clips, WER {100 * wrong / total:.1f}%, median {statistics.median(seconds):.3f} s, "
          f"p90 {p90:.3f} s (load and warm-up {loaded:.1f} s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="16 kHz mono WAVs")
    parser.add_argument("--phrases", type=Path, help="one request per line; clips named <anything>_<line>.wav")
    parser.add_argument("--show", action="store_true", help="print each clip a model got wrong")
    args = parser.parse_args()
    found = clips(args.folder, args.phrases)
    for name, load in models():
        measure(name, load, found, args.show)


if __name__ == "__main__":
    main()
