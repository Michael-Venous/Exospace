"""A pointing model: for the targets Jev refuses, or in place of Jev (bench only).

Study 9's matrix: at the planner's gates Jev refused 31 of 100 described controls and never misclicked;
Perceptron Mk1.5 hit all 31 of those (the chain: 100/100, 0.69 s mean, derived from separate calls).
Study 17 (`click_check.py`): Qwen 3.8 27B on ModelRun hit 192 of 192 present targets with no misclick and
refused all 12 absent ones, 0.42 s p50; Jev 169 / 8 misclicks / 15 refused. The host matters: on Cerebras
the same model hit 15 of 192.
"""

from __future__ import annotations

import os
import time

from bench.arch.openrouter_study import OpenRouterJSON
from brain.agent.jev import JevVision
from brain.cancel import CancelToken
from brain.imaging import shrink

MODEL = "perceptron/perceptron-mk1.5"
PROMPT = ('Find {target}. Reply with JSON {{"found": true|false, "x": 0-1000, "y": 0-1000}}: the centre of that '
          "control, x from the left edge and y from the top edge of the image; found false if it isn't visible.")
SCHEMA = {"type": "object", "properties": {"found": {"type": "boolean"}, "x": {"type": "number"},
                                           "y": {"type": "number"}}, "required": ["found", "x", "y"]}
# Required x and y: with only "found" required, DeepSeek answered {"found": true} and nothing else (study 17).


class Pointer:
    def __init__(self, cancel: CancelToken, jev: JevVision, model: str = MODEL, only: str = "") -> None:
        """`only` pins one provider (e.g. "modelrun/fp4"); otherwise the lowest-latency one serves."""
        self.cancel, self.jev = cancel, jev  # its calls go in Jev's trace, beside Jev's own looks
        self.client = OpenRouterJSON(model, os.environ["OPENROUTER_API_KEY"], "", fastest=not only, only=only,
                                     keepalive=False)

    def point(self, image: bytes, target: str) -> tuple[float, float] | None:
        """Where `target` is (0-1 of the screen), or None if the model doesn't see it."""
        started = time.monotonic()
        data, _ = self.cancel.run(lambda: self.client.call(shrink(image), PROMPT.format(target=target),
                                                           "You locate controls in screenshots.", SCHEMA))
        self.jev.trace.append({"label": "pointer", "target": target, "reply": data,
                               "elapsed_s": round(time.monotonic() - started, 3)})
        if not data.get("found") or "x" not in data or "y" not in data:
            return None
        return min(max(float(data["x"]) / 1000, 0.0), 1.0), min(max(float(data["y"]) / 1000, 0.0), 1.0)

    def close(self) -> None:
        self.client.client.close()
