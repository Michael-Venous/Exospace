"""Replay cache: a request that succeeded before, from the same-looking screen, runs again
with no model calls.

Recorded per request: the screen before each step and the actions that ran on it. Replay
continues only while every screen matches the recording; the end is confirmed by one Jev
completion check; any mismatch hands the request to the wrapped executor on the current
screen. Runs that needed an approval (sends, deletes) or asked a question are never recorded,
so a replay can never skip a confirmation.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from brain.agent.executor import Executor, Plan, StepContext
from brain.types import Action, OSName

MATCH = 6.0  # mean grayscale difference (0-255) between screens that count as the same
SIZE = (160, 100)  # finer than the settle thumbnail: a moved window must not match
GATE = 0.5  # completion score that confirms a replay (as the planner's completion gate)


def fingerprint(jpeg: bytes) -> np.ndarray | None:
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    return None if image is None else cv2.resize(image, SIZE, interpolation=cv2.INTER_AREA)


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return float(np.mean(cv2.absdiff(a, b))) < MATCH


def _key(goal: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", goal.lower()).strip()


@dataclass
class _Recording:
    screens: list[np.ndarray] = field(default_factory=list)
    actions: list[list[Action]] = field(default_factory=list)
    summary: str = ""


class ReplayCache:
    """Executor wrapper; everything it does not handle goes to `inner`."""

    def __init__(self, inner: Executor, path: Path) -> None:
        self.inner, self.path = inner, path
        self._lock = threading.Lock()
        self._store: dict[str, dict[str, Any]] = self._load()
        self._run: tuple[str, str] | None = None  # (run_id, goal) being handled
        self._replaying: _Recording | None = None
        self._recording: _Recording | None = None
        self._pending: tuple[np.ndarray, list[Action]] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    @property
    def verifies_stages(self) -> bool:
        return self._replaying is not None or getattr(self.inner, "verifies_stages", False) is True

    def next_actions(self, image_jpeg: bytes, ctx: StepContext) -> Plan:
        screen = fingerprint(image_jpeg)
        if self._run != (ctx.run_id, ctx.goal):
            self._run, self._pending = (ctx.run_id, ctx.goal), None
            self._replaying = self._match(ctx.goal, screen)
            self._recording = None if self._replaying else _Recording()
        if self._replaying is not None:
            plan = self._replay_step(image_jpeg, screen, ctx)
            if plan is not None:
                return plan
            self._replaying = None  # the screen differs from the recording: plan from here
        plan = self.inner.next_actions(image_jpeg, ctx)
        self._observe(screen, plan, ctx)
        return plan

    def acknowledge(self, executed: bool) -> None:
        if self._recording is not None and self._pending is not None and executed:
            screen, actions = self._pending
            self._recording.screens.append(screen)
            self._recording.actions.append(actions)
        self._pending = None
        if self._replaying is None:
            acknowledge = getattr(self.inner, "acknowledge", None)
            if callable(acknowledge):
                acknowledge(executed)

    # -- replay ----------------------------------------------------------------

    def _match(self, goal: str, screen: np.ndarray | None) -> _Recording | None:
        entry = self._store.get(_key(goal))
        if entry is None or screen is None:
            return None
        recording = _decode(entry)
        return recording if recording.screens and _same(recording.screens[0], screen) else None

    def _replay_step(self, image_jpeg: bytes, screen: np.ndarray | None, ctx: StepContext) -> Plan | None:
        recording = self._replaying
        index = ctx.step - 1
        if index < len(recording.actions):
            if screen is None or not _same(recording.screens[index], screen):
                return None
            return Plan(f"Replay {index + 1}/{len(recording.actions)}", recording.actions[index], [])
        jev = getattr(self.inner, "jev", None)
        if jev is None:
            return None
        executed = [a.brief() for step in recording.actions for a in step]
        score = jev.completion(image_jpeg, ctx.goal, executed, evidence=recording.summary,
                               run_id=ctx.run_id, step=ctx.step)
        if score < GATE:
            return None
        return Plan("Replay verified", [Action("done", summary=recording.summary or "Done.")], [])

    # -- recording -------------------------------------------------------------

    def _observe(self, screen: np.ndarray | None, plan: Plan, ctx: StepContext) -> None:
        recording = self._recording
        if recording is None or screen is None:
            return
        terminal = next((a for a in plan.actions if a.type in ("done", "ask_user", "fail")), None)
        if terminal is None:
            steps = [a for a in plan.actions if a.type != "remember"]
            self._pending = (screen, steps) if steps else None
            return
        approved = "approved this request" in ctx.goal or "User correction" in ctx.goal
        if terminal.type == "done" and recording.actions and not approved:
            recording.summary = terminal.summary
            self._save(ctx.goal, recording)
        self._recording = None

    def _save(self, goal: str, recording: _Recording) -> None:
        with self._lock:
            self._store[_key(goal)] = {
                "screens": [s.tolist() for s in recording.screens],
                "actions": [[_action_json(a) for a in step] for step in recording.actions],
                "summary": recording.summary,
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._store))
            tmp.replace(self.path)

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def classify_os(self, image_jpeg: bytes) -> OSName:
        return self.inner.classify_os(image_jpeg)


def _action_json(action: Action) -> dict[str, Any]:
    data = asdict(action)
    data["keys"] = list(action.keys)
    return data


def _decode(entry: dict[str, Any]) -> _Recording:
    return _Recording(
        screens=[np.array(s, dtype=np.uint8) for s in entry.get("screens", [])],
        actions=[[Action(**{**a, "keys": tuple(a.get("keys", ()))}) for a in step] for step in entry.get("actions", [])],
        summary=str(entry.get("summary", "")),
    )
