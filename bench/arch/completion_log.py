"""Records every Jev completion call of an e2e run with the screen it judged and the fixture's state then.

The runtime's completion check (`JevVision.completion`: goal + executed steps + screen, gate 0.5) decides
when a request is done, and every "not verified" answer is a Flash replan. A trace keeps its scores but not
the screen or whether the goal was really met at that moment; this log keeps both, so each call can be
labelled by the scenario's own oracle and replayed (`completion_study.py`).
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

from brain.agent.jev import JevVision


class CompletionLog:
    def __init__(self, folder: Path, jev: JevVision, read_state: Callable[[], dict]) -> None:
        self.folder, self.read_state = folder, read_state
        self.started = time.monotonic()
        self.states: list[tuple[float, dict]] = [(0.0, read_state())]
        self.calls: list[dict] = []
        self._lock = threading.Lock()
        judge = jev.completion

        def recorded(jpeg: bytes, goal: str, executed: list[str], **kwargs) -> float:
            seen = len(self.states) - 1  # the state after the last input sent, the one this screen shows
            score = judge(jpeg, goal, executed, **kwargs)
            with self._lock:
                n = len(self.calls)
                (folder / f"completion-{n}.jpg").write_bytes(jpeg)
                self.calls.append({"n": n, "t": round(time.monotonic() - self.started, 3), "goal": goal,
                                   "executed": list(executed), "evidence": kwargs.get("evidence", ""),
                                   "score": score, "state": seen})
            return score

        jev.completion = recorded  # type: ignore[method-assign]

    def input_sent(self) -> None:
        """Call on the controller's thread after each input: Playwright reads the page only there."""
        self.states.append((round(time.monotonic() - self.started, 3), self.read_state()))

    def save(self) -> None:
        for call in self.calls:
            call["state"] = self.states[call["state"]][1]
        (self.folder / "completion.json").write_text(json.dumps(self.calls, indent=1))
