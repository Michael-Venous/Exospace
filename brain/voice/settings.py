"""The box's voice settings, kept across restarts in a small JSON file ($EXO_VOICE_SETTINGS, by default
~/.config/exospace/voice.json). A missing or broken file means the defaults; a file that can't be written is
reported, never fatal (the voice must keep working). No file at all is a first run: the front ends then start the
spoken setup (brain.voice.setup), unless EXO_FIRST_RUN_SETUP=0."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, get_args

from brain.voice.commands import MAX_SPEED, MIN_SPEED

PATH = Path(os.environ.get("EXO_VOICE_SETTINGS") or Path.home() / ".config/exospace/voice.json")
FIRST_RUN_SETUP = os.environ.get("EXO_FIRST_RUN_SETUP", "1") != "0"

# How much the box talks while it works: every step, just the results, or only when a request is done.
# Chosen in setup and saved, but nothing reads it yet: narration still speaks every step.
Talk = Literal["steps", "results", "done"]


@dataclass(frozen=True)
class Settings:
    speed: float = 1.0  # how fast the box talks: 2.0 is twice normal
    echo: bool = True  # dictated words are read back aloud, as a screen reader echoes typing
    computer: str = ""  # what the user calls the computer plugged in, e.g. "Work laptop"
    talk: Talk = "steps"


def load(path: Path = PATH) -> Settings:
    try:
        saved = json.loads(path.read_text())
        speed = float(saved.get("speed", Settings.speed))
        echo, computer, talk = saved.get("echo"), saved.get("computer"), saved.get("talk")
    except (OSError, ValueError, TypeError, AttributeError):
        return Settings()
    return Settings(speed=min(MAX_SPEED, max(MIN_SPEED, speed)), echo=echo if isinstance(echo, bool) else True,
                    computer=computer if isinstance(computer, str) else "",
                    talk=talk if talk in get_args(Talk) else Settings.talk)


def save(settings: Settings, path: Path = PATH) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(settings)) + "\n")
    except OSError as exc:
        print(f"  (voice settings not saved to {path}: {exc})", flush=True)
