"""Screenshots sized by pixel budget, whatever the screen's aspect ratio (study 18, bench only).

Budgets are total pixels, not a long side: a long side of 1280 gives a 16:10 laptop 1.0 MP but a 21:9 ultrawide
0.69 MP, and the pointer missed 9 of 20 targets there. Each frame is resized once (INTER_AREA) and encoded once.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

# DeepSeek V4.1 Flash on Together and Morph resizes any image over about 1.6 MP itself, at every aspect ratio
# measured (16:10, 16:9, 21:9, 32:9, 9:16, 4:3): a bigger one adds upload bytes, not detail.
PLANNER_PIXELS = int(os.environ.get("EXO_OPT_MAX_PIXELS", "1600000"))
# Qwen 3.8 27B on ModelRun has no budget (its tokens grow with area). At 2.5 MP it hit 58/58 targets on a 150%
# laptop, a 4K screen at 100% and a 3440x1440 ultrawide at 100% (0.55 s p50); 1280 px long side 48/60, 1.6 MP
# 55/59, 4 MP and native no better and slower (0.8-1.7 s).
POINTER_PIXELS = int(os.environ.get("EXO_POINT_PIXELS", "2500000"))


def fit_scale(width: int, height: int, pixels: int) -> float:
    """The factor that brings width x height within `pixels` (never above 1: frames are not enlarged)."""
    return min(1.0, (pixels / (width * height)) ** .5)


def fit_jpeg(frame: np.ndarray, pixels: int, quality: int = 90) -> bytes:
    h, w = frame.shape[:2]
    scale = fit_scale(w, h, pixels)
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
