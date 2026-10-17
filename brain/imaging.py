"""Preparing frames for the model: crop to the active screen, shrink, add an
aiming grid, and JPEG-encode. Smaller images upload and process faster."""

from __future__ import annotations

import cv2
import numpy as np

from brain.types import Frame, Rect

MAX_SIDE = 1280
JPEG_QUALITY = 80
GRID_DIVISIONS = 10
GRID_COLOR = (0, 0, 255)  # BGR red


def prepare(frame: Frame, rect: Rect, grid: bool = True, max_side: int = MAX_SIDE) -> bytes:
    image = rect.crop(frame)
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        image = image.copy()
    if grid:
        _draw_grid(image)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def shrink(jpeg: bytes, max_side: int = MAX_SIDE) -> bytes:
    """A smaller copy of a prepared JPEG for the language model, which needs no more than MAX_SIDE."""
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if image is None or max(image.shape[:2]) <= max_side:
        return jpeg  # already small, or not an image this module can read: pass it on unchanged
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def _draw_grid(image: Frame) -> None:
    """Faint 10x10 grid labeled in the model's 0-1000 coordinate space."""
    h, w = image.shape[:2]
    overlay = image.copy()
    for i in range(1, GRID_DIVISIONS):
        x = round(w * i / GRID_DIVISIONS)
        y = round(h * i / GRID_DIVISIONS)
        cv2.line(overlay, (x, 0), (x, h), GRID_COLOR, 1)
        cv2.line(overlay, (0, y), (w, y), GRID_COLOR, 1)
    cv2.addWeighted(overlay, 0.35, image, 0.65, 0, dst=image)
    for i in range(1, GRID_DIVISIONS):
        label = str(i * 1000 // GRID_DIVISIONS)
        x = round(w * i / GRID_DIVISIONS)
        y = round(h * i / GRID_DIVISIONS)
        cv2.putText(image, label, (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, GRID_COLOR, 1)
        cv2.putText(image, label, (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, GRID_COLOR, 1)
