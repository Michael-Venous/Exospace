"""Where the target's screen sits in the capture frame, found with the pointer.

A mirrored screen rarely fills the capture frame: a 16:10 laptop mirrored into a 16:9 capture
has black bars on both sides, and on a black wallpaper they can't be told from the desktop, so
active_rect() returns the whole frame and every click lands a few percent off. The absolute
mouse doesn't care what the screen shows: point it at two known places, see where the cursor
appears, and the screen's rectangle follows. Touch targets have no cursor and keep active_rect().
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from brain.capture import ASPECT_RANGE, Capture
from brain.types import Rect

POINTS = ((0.1, 0.1), (0.9, 0.9))  # where the pointer is sent, as fractions of the target's screen
NUDGE = 0.04  # the second look at each point is this much further right and down
SETTLE_S = 0.35  # the cursor has moved on the captured frame by then (capture card latency included)
CURSOR_AREA = (8, 4000)  # changed pixels of one cursor image, in the capture frame
CURSOR_SIDE = 90  # a cursor's bounding box is never wider or taller than this
SLACK = 0.03  # a fitted rectangle may overshoot the frame by this fraction (cursor tip estimates)


def pointer_rect(capture: Capture, move, settle_s: float = SETTLE_S) -> Rect | None:
    """The target's screen inside the capture frame, or None if the cursor wasn't found.
    `move(x, y)` sends the absolute pointer to fractions of the target's screen."""
    tips = []
    for x, y in POINTS:
        before = _look(capture, move, x, y, settle_s)
        after = _look(capture, move, x + NUDGE, y + NUDGE, settle_s)
        if before is None or after is None:
            return None
        tip = _cursor_tip(before, after)
        if tip is None:
            return None
        tips.append(tip)
    move(0.5, 0.5)
    h, w = before.shape[:2]
    (ax, ay), (bx, by) = tips
    span = POINTS[1][0] - POINTS[0][0]
    width, height = (bx - ax) / span, (by - ay) / span
    x0, y0 = ax - POINTS[0][0] * width, ay - POINTS[0][1] * height
    rect = Rect(round(x0), round(y0), round(x0 + width), round(y0 + height))
    if not _plausible(rect, w, h):
        return None
    return Rect(max(0, rect.x0), max(0, rect.y0), min(w, rect.x1), min(h, rect.y1))


def _look(capture: Capture, move, x: float, y: float, settle_s: float) -> np.ndarray | None:
    move(x, y)
    time.sleep(settle_s)
    return capture.grab()


def _cursor_tip(before: np.ndarray, after: np.ndarray) -> tuple[float, float] | None:
    """The cursor's tip (top-left of its image) in `before`. The pointer moved right and down by the same
    fraction of the screen's width and height, so of the cursor-sized changes the pair offset that way, with
    alike sizes, is the cursor; other changes (a redraw, a clock) are ignored. On the Pi rig a frame held 6
    changes beside the cursor."""
    diff = cv2.cvtColor(cv2.absdiff(before, after), cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    blobs = [stats[i] for i in range(1, count)
             if CURSOR_AREA[0] <= stats[i][cv2.CC_STAT_AREA] <= CURSOR_AREA[1]
             and stats[i][cv2.CC_STAT_WIDTH] <= CURSOR_SIDE and stats[i][cv2.CC_STAT_HEIGHT] <= CURSOR_SIDE]
    h, w = before.shape[:2]
    pairs = []
    for a in blobs:
        for b in blobs:
            dx, dy = b[cv2.CC_STAT_LEFT] - a[cv2.CC_STAT_LEFT], b[cv2.CC_STAT_TOP] - a[cv2.CC_STAT_TOP]
            alike = abs(int(a[cv2.CC_STAT_HEIGHT]) - int(b[cv2.CC_STAT_HEIGHT])) <= 4
            # The nudge is NUDGE of the screen both ways: dx/w and dy/h agree for any screen inside the frame.
            if dx > 0 and dy > 0 and alike and 0.5 <= (dx / w) / (dy / h) <= 2 and NUDGE / 3 <= dy / h <= NUDGE * 1.5:
                pairs.append((a, b))
    if len(pairs) != 1:  # none, or ambiguous: don't guess
        return None
    first = pairs[0][0]
    return float(first[cv2.CC_STAT_LEFT]), float(first[cv2.CC_STAT_TOP])


def _plausible(rect: Rect, w: int, h: int) -> bool:
    width, height = rect.x1 - rect.x0, rect.y1 - rect.y0
    inside = (rect.x0 >= -SLACK * w and rect.y0 >= -SLACK * h
              and rect.x1 <= (1 + SLACK) * w and rect.y1 <= (1 + SLACK) * h)
    return inside and width >= w / 2 and height >= h / 2 and ASPECT_RANGE[0] <= width / height <= ASPECT_RANGE[1]
