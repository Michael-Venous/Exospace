"""A pointing model: for the targets Jev refuses, or in place of Jev (bench only).

Study 9's matrix: at the planner's gates Jev refused 31 of 100 described controls and never misclicked;
Perceptron Mk1.5 hit all 31 of those (the chain: 100/100, 0.69 s mean, derived from separate calls).
Study 17 (`click_check.py`): Qwen 3.8 27B on ModelRun hit 192 of 192 present targets with no misclick and
refused all 12 absent ones, 0.42 s p50; Jev 169 / 8 misclicks / 15 refused. The host matters: on Cerebras
the same model hit 15 of 192. GLM 5.3 Flash answers in the image's pixels whatever it is asked (asked so:
189 hit / 3 misclicked of 192 on Together, 0.60 s p50), so it is asked in pixels.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import cv2
import numpy as np

from bench.arch.openrouter_study import OpenRouterJSON
from brain.agent.jev import JevVision
from brain.cancel import CancelToken
from brain.imaging import shrink

MODEL = "perceptron/perceptron-mk1.5"
READ_S = 3.0  # longest wait for a reply's next bytes
HEDGE_S = 1.5  # a pointing reply not back by then gets a second request beside it (usually 0.5 s)
TRIES = 3  # pointing requests at most, hedge included
GIVE_UP_S = 8.0  # no reply from either by then: the click fails
PROMPT = ('Find {target}. Reply with JSON {{"found": true|false, "x": 0-1000, "y": 0-1000}}: the centre of that '
          "control, x from the left edge and y from the top edge of the image; found false if it isn't visible. "
          "It is the thing to click (a link, button or control), not the same words as plain text, typed in a text "
          "field or shown in a title.")
PIXEL_PROMPT = ('Find {target}. Reply with JSON {{"found": true|false, "x": ..., "y": ...}}: the centre of that '
                "control in pixels of this {w}x{h} image, x from the left edge and y from the top edge; found false "
                "if it isn't visible. It is the thing to click (a link, button or control), not the same words as plain "
                "text, typed in a text field or shown in a title.")
SCHEMA = {"type": "object", "properties": {"found": {"type": "boolean"}, "x": {"type": "number"},
                                           "y": {"type": "number"}}, "required": ["found", "x", "y"]}
# Required x and y: with only "found" required, DeepSeek answered {"found": true} and nothing else (study 17).


class Pointer:
    def __init__(self, cancel: CancelToken, jev: JevVision, model: str = MODEL, only: str = "",
                 effort: str = "", pixels: bool = False) -> None:
        """`only` pins one provider (e.g. "modelrun/fp4"); otherwise the lowest-latency one serves. `effort` ""
        turns reasoning off (some models refuse that and take "minimal"). `pixels`: the model is asked for x and y
        in the image's pixels, not 0-1000."""
        self.cancel, self.jev, self.pixels = cancel, jev, pixels  # its calls go in Jev's trace, beside Jev's looks
        self.pool = ThreadPoolExecutor(4, thread_name_prefix="pointer")
        # A reply that stalls READ_S, or a dropped connection, is asked once more (a GLM pointing call once hung past
        # 8 s; three clicks in a row once failed on TLS errors while the link was flaky).
        self.client = OpenRouterJSON(model, os.environ["OPENROUTER_API_KEY"], effort, fastest=not only, only=only,
                                     keepalive=False, read_s=READ_S)

    def point(self, image: bytes, target: str) -> tuple[float, float] | None:
        """Where `target` is (0-1 of the screen), or None if the model doesn't see it."""
        started = time.monotonic()
        image = shrink(image)
        h, w = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_GRAYSCALE).shape[:2]
        scale = (w, h) if self.pixels else (1000, 1000)
        prompt = (PIXEL_PROMPT.format(target=target, w=w, h=h) if self.pixels else PROMPT.format(target=target))
        def ask() -> tuple[dict, dict]:
            return self.client.call(image, prompt, "You locate controls in screenshots.", SCHEMA)

        data = self._hedged(ask)
        self.jev.trace.append({"label": "pointer", "target": target, "reply": data,
                               "elapsed_s": round(time.monotonic() - started, 3)})
        if not data.get("found") or "x" not in data or "y" not in data:
            return None
        return min(max(float(data["x"]) / scale[0], 0.0), 1.0), min(max(float(data["y"]) / scale[1], 0.0), 1.0)

    def _hedged(self, ask) -> dict:
        """The reply of `ask`, with a second identical request sent if the first hasn't answered in HEDGE_S: the
        first good reply wins. A pointing call usually takes 0.5 s, but one stalled 8.7 s behind keep-alive lines
        (no read timeout fires), and three failed on TLS errors while the link was flaky."""
        self.cancel.check()
        tries = [self.pool.submit(ask)]
        done, _ = wait(tries, timeout=HEDGE_S)
        if not done or tries[0].exception() is not None:
            tries.append(self.pool.submit(ask))
        deadline = time.monotonic() + GIVE_UP_S
        pending = set(tries)
        while pending and time.monotonic() < deadline:
            done, pending = wait(pending, timeout=deadline - time.monotonic(), return_when=FIRST_COMPLETED)
            for future in done:
                if future.exception() is None:
                    self.cancel.check()
                    return future.result()[0]
            if not pending and len(tries) < TRIES:  # every try failed (both once got a stale TLS connection): again
                tries.append(self.pool.submit(ask))
                pending = {tries[-1]}
        raise TimeoutError(f"no pointing reply in {GIVE_UP_S} s") if pending else tries[-1].exception()

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.client.client.close()
