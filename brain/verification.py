"""Local visual-change evidence, distinct from whole-screen settling.

Small field edits disappear in a global thumbnail average. Detect a modest
connected changed region at higher resolution; this is evidence of a visual
change, not semantic proof that the requested task succeeded.
"""
import cv2
import numpy as np


def visibly_changed(before, after) -> bool:
    if before.shape != after.shape:
        return True
    height, width = before.shape[:2]
    scale = min(1.0, 1280 / max(height, width))
    size = (max(1,round(width*scale)), max(1,round(height*scale)))
    a=cv2.cvtColor(cv2.resize(before,size,interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
    b=cv2.cvtColor(cv2.resize(after,size,interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
    mask=(cv2.absdiff(a,b)>20).astype(np.uint8)
    # Connect nearby letter strokes. Ignore isolated pixel noise and a thin caret.
    mask=cv2.dilate(mask,np.ones((3,3),np.uint8))
    count, labels, stats, centers=cv2.connectedComponentsWithStats(mask,8)
    return any(s[cv2.CC_STAT_AREA]>=40 and s[cv2.CC_STAT_WIDTH]>=6
               and s[cv2.CC_STAT_HEIGHT]>=6 for s in stats[1:])




def highlight_rows(before, after) -> tuple[int, int] | None:
    """Pixel rows (first, last) of the bar a key press moved a menu highlight onto, or None.

    Moving a solid highlight bar changes every pixel of the rows it leaves and lands on.
    The landing rows are the run that gained the most pixels unlike the background:
    letters cancel out between the two bars, a bar's blank margins do not. Text that
    merely changes is too thin to count."""
    if before.shape != after.shape:
        return None
    size = max(3, after.shape[0] // 160) * 2 + 1  # odd: thicker than letter strokes, thinner than a row
    changed = (cv2.absdiff(before, after).max(axis=2) > 20).astype(np.uint8)
    block = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((size, 2 * size + 1), np.uint8)).astype(bool)
    background = np.median(after[::4, ::4].reshape(-1, 3), axis=0).astype(np.int16)
    unlike = [np.abs(frame.astype(np.int16) - background).max(axis=2) > 60 for frame in (before, after)]
    gained = (block & unlike[1]).sum(axis=1) - (block & unlike[0]).sum(axis=1)
    score = np.where(block.any(axis=1), gained, -size)  # an unchanged row ends a bar
    best, total, start = (0, 0, 0), 0, 0
    for row, value in enumerate(score.tolist()):  # maximum-sum run of rows
        total, start = (total, start) if total > 0 else (0, row)
        total += value
        best = max(best, (total, start, row))
    return (best[1], best[2]) if best[0] >= size * size else None
