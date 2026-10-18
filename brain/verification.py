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
