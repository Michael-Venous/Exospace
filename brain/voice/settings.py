"""The box's voice settings, kept across restarts in a small JSON file ($EXO_VOICE_SETTINGS, by default
~/.config/exospace/voice.json). A missing or broken file means the defaults; a file that can't be written is
reported, never fatal (the voice must keep working)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from brain.voice.commands import MAX_SPEED, MIN_SPEED

PATH = Path(os.environ.get("EXO_VOICE_SETTINGS") or Path.home() / ".config/exospace/voice.json")


@dataclass(frozen=True)
class Settings:
    speed: float = 1.0  # how fast the box talks: 2.0 is twice normal


def load(path: Path = PATH) -> Settings:
    try:
        saved = json.loads(path.read_text())
        speed = float(saved.get("speed", Settings.speed))
    except (OSError, ValueError, TypeError, AttributeError):
        return Settings()
    return Settings(speed=min(MAX_SPEED, max(MIN_SPEED, speed)))


def save(settings: Settings, path: Path = PATH) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(settings)) + "\n")
    except OSError as exc:
        print(f"  (voice settings not saved to {path}: {exc})", flush=True)
