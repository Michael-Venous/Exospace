"""Cost of the agent loop's mid-batch guard: one thumbnail and change check per aimed
action after the first in a batch. `grab()` returns the reader thread's latest frame,
so the capture card adds no wait. Run on the Pi for the deployment number:

    python -m bench.guard_cost --device pi-microbenchmark
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
import uuid
from pathlib import Path

import numpy as np

from brain.capture import frame_change, thumbnail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="laptop-microbenchmark")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--csv", type=Path, default=Path("bench/results.csv"))
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    before = thumbnail(frame)
    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        frame_change(before, thumbnail(frame))
        times.append(time.perf_counter() - start)
    median, p95 = statistics.median(times), sorted(times)[int(0.95 * len(times))]
    print(f"guard check, 1080p: median {median * 1000:.2f} ms, p95 {p95 * 1000:.2f} ms over {args.runs}")
    with args.csv.open("a", newline="") as stream:
        csv.writer(stream).writerow([f"guard-{uuid.uuid4().hex[:8]}", "mid_batch_guard", args.device,
                                     "guard_check_1080p", 0, round(median, 6), 0, 0, 0, 0, "", round(median, 6)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
