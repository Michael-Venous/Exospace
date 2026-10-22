"""Benchmark log (PLAN.md §5.7). Every speed claim comes from this file.

Timings per step, in seconds:
  t_capture  grabbing the fresh frame after the screen settled
  t_upload   cropping, resizing and JPEG-encoding the frame for the model
  t_model    the model call, including network round trip
  t_execute  sending the actions over HID
  t_settle   waiting for the screen to stop changing afterwards
"""

from __future__ import annotations

import csv
import uuid
from pathlib import Path

FIELDS = [
    "run_id", "config", "device", "task_id", "step",
    "t_capture", "t_upload", "t_model", "t_execute", "t_settle",
    "success", "total_s",
]


class Telemetry:
    def __init__(self, path: Path, config_name: str) -> None:
        self._path = path
        self._config = config_name

    def new_run(self) -> str:
        return uuid.uuid4().hex[:8]

    def step(
        self, run_id: str, device: str, task_id: str, step: int, *,
        t_capture: float, t_upload: float, t_model: float, t_execute: float, t_settle: float,
    ) -> None:
        print(f"timing run={run_id} step={step} model={t_model:.2f}s "
              f"execute={t_execute:.2f}s settle={t_settle:.2f}s "
              f"capture+prepare={t_capture + t_upload:.2f}s", flush=True)
        self._write({
            "run_id": run_id, "config": self._config, "device": device,
            "task_id": task_id, "step": step,
            "t_capture": f"{t_capture:.3f}", "t_upload": f"{t_upload:.3f}",
            "t_model": f"{t_model:.3f}", "t_execute": f"{t_execute:.3f}",
            "t_settle": f"{t_settle:.3f}", "success": "", "total_s": "",
        })

    def finish(
        self, run_id: str, device: str, task_id: str, steps: int, success: bool, total_s: float
    ) -> None:
        """Summary row for a whole task; `step` holds the step count."""
        self._write({
            "run_id": run_id, "config": self._config, "device": device,
            "task_id": task_id, "step": f"total:{steps}",
            "t_capture": "", "t_upload": "", "t_model": "", "t_execute": "", "t_settle": "",
            "success": int(success), "total_s": f"{total_s:.3f}",
        })

    def _write(self, row: dict[str, object]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self._path.exists()
        with self._path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")  # csv defaults to CRLF
            if new_file:
                writer.writeheader()
            writer.writerow(row)
