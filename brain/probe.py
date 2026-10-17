"""Opt-in live model probe: synthetic image only, no capture or keyboard input."""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from brain.agent.model import language_model
from brain.cancel import CancelToken
from brain.config import Config
from brain.imaging import prepare
from brain.types import Rect


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.runs <= 20:
        parser.error("--runs must be between 1 and 20")
    cfg = Config.load()
    cancel = CancelToken()
    frame = np.full((120, 160, 3), (20, 80, 200), dtype=np.uint8)
    jpeg = prepare(frame, Rect(0, 0, 160, 120), grid=False)
    schema = {"type": "object", "properties": {"color": {"type": "string"}}, "required": ["color"]}
    try:
        for role in ("fast", "planner"):
            client = language_model(cfg, cancel, planner=role == "planner")
            try:
                elapsed = []
                for n in range(args.runs):
                    start = time.monotonic()
                    result = client.generate(jpeg, "Name the dominant color in one word.",
                                             "Describe only the supplied image.", schema,
                                             role="probe-" + role, run_id="probe", step=n + 1)
                    if not isinstance(result.get("color"), str) or not result["color"]:
                        raise ValueError("Missing color in probe result")
                    elapsed.append(time.monotonic() - start)
                print(f"{role} {client.model}: n={len(elapsed)}, median={statistics.median(elapsed):.3f}s, max={max(elapsed):.3f}s")
            finally:
                client.close()
    except Exception as exc:
        print(f"Probe failed ({type(exc).__name__}); check API key, model access and network.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
