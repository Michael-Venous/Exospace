"""Session memory (PLAN.md §5.5). Lives on the brain, so it survives every
target swap, and is saved to disk after each change."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TRANSCRIPT_LIMIT = 50


@dataclass
class Memory:
    path: Path
    transcript: list[dict[str, str]] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    task: dict[str, Any] = field(default_factory=dict)
    prefs: dict[str, Any] = field(default_factory=dict)
    # The race executor's speculative planner thread also writes (set_subtasks).
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    @classmethod
    def load(cls, path: Path) -> Memory:
        if not path.exists():
            return cls(path=path)
        data = json.loads(path.read_text())
        return cls(
            path=path,
            transcript=data.get("transcript", []),
            facts=data.get("facts", {}),
            task=data.get("task", {}),
            prefs=data.get("prefs", {}),
        )

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {"transcript": self.transcript, "facts": self.facts, "task": self.task, "prefs": self.prefs}
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name, suffix=".tmp")
            with os.fdopen(fd, "w") as out:
                out.write(json.dumps(data, indent=2))
            os.replace(tmp, self.path)

    def add_turn(self, role: str, text: str) -> None:
        with self._lock:
            self.transcript.append({"role": role, "text": text})
            del self.transcript[:-TRANSCRIPT_LIMIT]
            self.save()

    def remember(self, key: str, value: Any) -> None:
        with self._lock:
            self.facts[key] = _maybe_json(value)
            self.save()

    def set_task(self, goal: str, status: str) -> None:
        with self._lock:
            if self.task.get("goal") != goal:
                self.task = {"goal": goal, "subtasks": [], "done": []}
            self.task.update(status=status)
            self.task.setdefault("subtasks", [])
            self.task.setdefault("done", [])
            self.save()

    def set_subtasks(self, subtasks: list[str]) -> None:
        with self._lock:
            self.task["subtasks"] = subtasks
            self.save()

    def recent_turns(self, n: int = 6) -> list[dict[str, str]]:
        return self.transcript[-n:]


def _maybe_json(value: Any) -> Any:
    """Models often hand us structured facts as JSON text; store them as data."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value
