"""Jev triage for one spoken segment: act now, launch an app, scroll, plan, or no action (bench/arch study 1, 6).

One Jev call classifies the utterance against the current screen, locates the target
it names, and flags irreversible requests. A second, zoomed call refines the target
point (fixed every near miss on small targets in the study). No regex or phrase rules.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from brain.agent.jev import IRREVERSIBLE_KINDS, JEV_SIDE, JevVision, probability
from brain.cancel import CancelToken

Mode = Literal["act", "plan", "none", "scroll", "launch"]
Gesture = Literal["click", "double_click", "right_click"]
Box = tuple[float, float, float, float]

GATE = 0.5  # presence/uniqueness, zoom presence, irreversible, approval: 0 unsafe acts at this gate
ZOOM_WINDOW = 0.30  # crop side as a fraction of the longer screen side
ZOOM_SCALE = 2.5
BOX_STEP = 0.01
FINE_WINDOW = 0.12  # the second, tighter look: Jev's boxes come in 0.005 steps of whatever it is shown
SCROLL = 5  # wheel notches for "scroll down a bit": about a screenful
MAX_SPANS = 40  # word runs of the utterance offered as the app's name
# Box width (fraction of the screen) below which a click is hover-checked first. Jev's boxes come in 0.01
# steps: KCalc's close X was boxed 0.97-0.99, minimize included.
SMALL_TARGET = 0.03
POINTED = ("This is a zoomed-in part of a screen. The mouse pointer rests on a control, which may be highlighted "
           "or show a tooltip naming it. Is that control the one that does what the `utterance` asks?")

INTENT_CRITERIA = {
    "act_now": "The utterance asks to click, tap, open or select ONE specific control that is visible on "
               "this screen right now, and that single pointer action fully satisfies the request.",
    "plan": "The utterance needs typing, keyboard keys, several steps, writing content, answering a question, "
            "an irreversible or destructive action, or its intended target is ambiguous or not visible.",
    "scroll": "The utterance only asks to scroll or page the current view up or down, not to find or open "
              "anything in particular.",
    "launch": "The next step is to open or start an app that is not the one in front on this screen "
              "(whatever the utterance asks for after that).",
    "no_action": "The utterance is a remark, background context, a feeling or an acknowledgement; "
                 "it does not ask for any action.",
}
GESTURE_CRITERIA = {
    "click": "A single click or tap: buttons, links, menus, tabs, list items in apps or web pages, "
             "and app icons on a phone.",
    "double_click": "Opening an icon on a computer desktop, or opening a file or folder shown as an "
                    "icon or row in a desktop file-manager window.",
    "right_click": "The user asks for a right-click, a context menu, or more options for an item.",
}
IRREVERSIBLE = "Would carrying out the `utterance` on this screen " + IRREVERSIBLE_KINDS
CORRECTION_SCOPE = (" The `utterance` corrects or continues the `earlier_request`: judge the whole request as "
                    "corrected, not the `utterance` alone.")
HELD_STEP = " The user was asked to approve this `held_step` of it."
NOT_APPROVAL = " A question, a hesitation or a change to it is not an approval."


@dataclass(frozen=True)
class Triage:
    mode: Mode
    point: tuple[float, float] | None  # normalized, zoom-refined; only for mode "act"
    irreversible: bool
    gesture: Gesture = "click"
    scroll: int = 0  # wheel notches for mode "scroll", positive = down
    app: str = ""  # for mode "launch": the utterance's own words naming the app
    small: bool = False  # for mode "act": a small target, worth a hover check before the click


def triage(jev: JevVision, jpeg: bytes, utterance: str, cancel: CancelToken, *,
           earlier: str = "", run_id: str = "", step: int = 0) -> Triage:
    names = spans(utterance)
    questions = {
        "intent": {"type": "choice", "criteria": INTENT_CRITERIA, "instructions":
                   "The user is talking to an AI assistant that operates the device shown in `screen` with a "
                   "mouse/touch and keyboard. Classify the user's latest `utterance`."},
        "present": {"type": "noul", "instructions":
                    "Is there a control visible in `screen` that the `utterance` asks to click, tap or open right now?"},
        "unique": {"type": "noul", "instructions":
                   "Does the `utterance` identify exactly one visible control in `screen`, with no ambiguity?"},
        "irreversible": {"type": "noul", "instructions": IRREVERSIBLE + (CORRECTION_SCOPE if earlier else "")},
        "direction": {"type": "choice", "criteria": {"down": "Down, further along, the next page.",
                                                     "up": "Up, back towards the top, the previous page."},
                      "instructions": "Which way does the `utterance` ask to scroll, if it asks to scroll?"},
        "gesture": {"type": "choice", "criteria": GESTURE_CRITERIA, "instructions":
                    "Which pointer gesture carries out the `utterance` on the control it names in `screen`?"},
        "app": {"type": "choice", "instructions": "Which words of the `utterance` name the app, or the kind of "
                "app, to open? Only the name, without \"the\", \"a\" or other words around it: it is typed into "
                "the app launcher's search.",
                "criteria": {"none": "The utterance names no app or kind of app."} | {str(i): json.dumps(text)
                                                                                    for i, text in enumerate(names)}},
        "target": {"type": "box", "instructions":
                   "The single visible control in `screen` that the `utterance` asks to click, tap or open "
                   "right now. For an item in a list or grid, locate the whole item."},
    }
    state = {"utterance": utterance, **({"earlier_request": earlier} if earlier else {})}
    answers, size, padded = jev.evaluate(jpeg, questions, run_id=run_id, step=step, state=state,
                                         label="triage", gates={"present": GATE, "unique": GATE, "irreversible": GATE})
    irreversible = probability(answers["irreversible"].get("noul")) >= GATE
    choice = answers["intent"].get("choice")
    if choice == "no_action":
        return Triage("none", None, irreversible)
    if choice == "scroll":
        return Triage("scroll", None, irreversible, scroll=-SCROLL if answers["direction"].get("choice") == "up" else SCROLL)
    app = answers["app"].get("choice")
    if choice == "launch" and app in {str(i) for i in range(len(names))}:
        # Launching needs no plan: Flash took 3-12 s to plan open_app (EXECUTOR=jev's speed came from this
        # step, done there by phrase rules); its plan for what follows keeps running meanwhile.
        return Triage("launch", None, irreversible, app=names[int(app)])
    present = min(probability(answers["present"].get("noul")), probability(answers["unique"].get("noul")))
    point = box_center(answers["target"], size, padded)
    if choice != "act_now" or present < GATE or point is None:
        return Triage("plan", None, irreversible)
    cancel.check()
    refined = refine(jev, jpeg, utterance, point, run_id=run_id, step=step)
    gesture = answers["gesture"].get("choice")
    gesture = gesture if gesture in GESTURE_CRITERIA else "click"
    small = box_size(answers["target"], size, padded)[0] < SMALL_TARGET
    named = names[int(app)] if app in {str(i) for i in range(len(names))} else ""
    return Triage("act", refined, irreversible, gesture, small=small, app=named) if refined \
        else Triage("plan", None, irreversible)


def pointed(jev: JevVision, jpeg: bytes, utterance: str, point: tuple[float, float], *,
            run_id: str = "", step: int = 0) -> bool:
    """With the pointer resting on `point`, is it on the control the utterance asks for? Its hover highlight
    and tooltip tell neighbours apart that look alike: KCalc's close X scored 0.97, the minimize chevron next
    to it (which Jev aimed at 3/3 for "close the calculator") 0.01-0.21."""
    crop, _ = _crop_around(jpeg, point, ZOOM_WINDOW)
    answers, _, _ = jev.evaluate(crop, {"on": {"type": "noul", "instructions": POINTED}}, run_id=run_id, step=step,
                                 state={"utterance": utterance})
    return probability(answers["on"].get("noul")) >= GATE


def spans(utterance: str, limit: int = MAX_SPANS) -> list[str]:
    """Each run of whole words in the utterance without wrapping quotes or punctuation, shortest first
    (from EXECUTOR=jev: whole runs bind a name better than separate words)."""
    words = list(re.finditer(r"\S+", utterance))
    runs = (utterance[words[a].start():words[a + n].end()].strip("\"'“”‘’,;.!?")
            for n in range(len(words)) for a in range(len(words) - n))
    return [run for run in dict.fromkeys(runs) if run][:limit]


UTTERANCE_CONTROL = "the control the `utterance` asks to click, tap or open"


def refine(jev: JevVision, jpeg: bytes, target: str, point: tuple[float, float], *, run_id: str = "",
           step: int = 0, state: dict | None = None, control: str = UTTERANCE_CONTROL,
           fine: bool = True, window: float = ZOOM_WINDOW) -> tuple[float, float] | None:
    """Re-locate `target` in an enlarged crop (`window`) around `point`, then (`fine`) in a tighter one around that;
    None if Jev no longer sees it. On 49 OCR-labelled targets (Blender, Wikipedia, KCalc, GitHub)
    the point landed in the target 4 times from the full screen, 28 after one zoom and 35 after
    both (a median of three zooms: 27)."""
    state = state or {"utterance": target}
    coarse, sure, _ = zoom(jev, jpeg, state, control, point, window, run_id, step)
    if coarse is None or sure < GATE:
        return None
    if not fine:
        return coarse
    closer, sure, _ = zoom(jev, jpeg, state, control, coarse, FINE_WINDOW, run_id, step)
    return closer if closer and sure >= GATE else coarse


def zoom(jev: JevVision, jpeg: bytes, state: dict, control: str, point: tuple[float, float], window: float,
         run_id: str = "", step: int = 0) -> tuple[tuple[float, float] | None, float, Box | None]:
    """One look at `control` in an enlarged crop (`window`) around `point`: where Jev boxed it, None with no box, how
    sure Jev is that it is there, and the box (left, top, right, bottom, 0-1 of the screen)."""
    crop, (x0, y0, x1, y1) = _crop_around(jpeg, point, window)
    questions = {
        "present": {"type": "noul", "instructions":
                    f"This is a zoomed-in part of a screen. Is {control} visible here?"},
        "target": {"type": "box", "instructions":
                   f"This is a zoomed-in part of a screen. Locate exactly {control}."},
    }
    answers, size, padded = jev.evaluate(crop, questions, run_id=run_id, step=step, state=state,
                                         label="zoom", gates={"present": GATE}, region=(x0, y0, x1, y1))
    local = box_center(answers["target"], size, padded)
    sure = probability(answers["present"].get("noul"))
    if local is None:
        return None, sure, None
    w, h = box_size(answers["target"], size, padded)
    left, top = x0 + (local[0] - w / 2) * (x1 - x0), y0 + (local[1] - h / 2) * (y1 - y0)
    box = (left, top, left + w * (x1 - x0), top + h * (y1 - y0))
    return (x0 + local[0] * (x1 - x0), y0 + local[1] * (y1 - y0)), sure, box


def encloses(outer: Box | None, inner: Box | None) -> bool:
    """Whether `outer` holds all of `inner` and is over twice its size: the region around it rather than the same
    control boxed again."""
    if outer is None or inner is None:
        return False
    holds = outer[0] - BOX_STEP <= inner[0] and outer[1] - BOX_STEP <= inner[1] \
        and inner[2] <= outer[2] + BOX_STEP and inner[3] <= outer[3] + BOX_STEP
    return holds and (outer[2] - outer[0]) * (outer[3] - outer[1]) > 2 * (inner[2] - inner[0]) * (inner[3] - inner[1])


Reply = Literal["approve", "decline", "other"]


APPROVED = "The user approved this request."  # the line an approved request's goal gains


def answer_to_pending(jev: JevVision, jpeg: bytes, pending: str, reply: str, *, held: str = "",
                      run_id: str = "", step: int = 0) -> Reply:
    """How the user's reply answers a pending irreversible request, or the `held` step of it the
    user was asked about: approve, decline, or neither (a new command or a correction, which is
    then handled as its own request)."""
    state = {"pending_request": pending, "reply": reply, **({"held_step": held} if held else {})}
    scope = HELD_STEP if held else ""
    answers, _, _ = jev.evaluate(jpeg, {
        "approve": {"type": "noul", "instructions": "Does the user's `reply` clearly approve carrying out the "
                    "`pending_request`?" + scope + (NOT_APPROVAL if held else "")},
        "decline": {"type": "noul", "instructions":
                    "Does the user's `reply` refuse, cancel or say no to the `pending_request`?" + scope},
    }, run_id=run_id, step=step, state=state, label="approval", gates={"approve": GATE, "decline": GATE})
    if probability(answers["approve"].get("noul")) >= GATE:
        return "approve"
    if probability(answers["decline"].get("noul")) >= GATE:
        return "decline"
    return "other"


def box_contains(answer: dict, size: tuple[int, int], padded: tuple[int, int], point: tuple[float, float]) -> bool:
    """Whether a box answer contains `point` (0-1 within the image)."""
    box = answer.get("box") or {}
    try:
        left, right, top, bottom = (probability(box.get(k)) for k in ("left", "right", "top", "bottom"))
    except ValueError:
        return False
    (w, h), (pw, ph) = size, padded
    # Jev's box edges come in 0.01 steps: a zoom on a footnote's [20] landed 0.002 right of its box (2026-10-19)
    return left * pw / w - BOX_STEP <= point[0] <= right * pw / w + BOX_STEP \
        and top * ph / h - BOX_STEP <= point[1] <= bottom * ph / h + BOX_STEP


def beside(answer: dict, size: tuple[int, int], padded: tuple[int, int], other: Box | None) -> bool:
    """Whether `other` (left, top, right, bottom, 0-1 of the image) lies within a box answer's own width and height
    of it."""
    box = answer.get("box") or {}
    try:
        left, right, top, bottom = (probability(box.get(k)) for k in ("left", "right", "top", "bottom"))
    except ValueError:
        return False
    if other is None:
        return False
    (w, h), (pw, ph) = size, padded
    left, right, top, bottom = left * pw / w, right * pw / w, top * ph / h, bottom * ph / h
    width, height = right - left, bottom - top
    return left - width <= other[2] and other[0] <= right + width and top - height <= other[3] \
        and other[1] <= bottom + height


def box_size(answer: dict, size: tuple[int, int], padded: tuple[int, int]) -> tuple[float, float]:
    """A box answer's width and height (0-1 of the image); a box Jev didn't give counts as the whole image."""
    box = answer.get("box") or {}
    try:
        left, right, top, bottom = (probability(box.get(k)) for k in ("left", "right", "top", "bottom"))
    except ValueError:
        return 1.0, 1.0
    return max(0.0, right - left) * padded[0] / size[0], max(0.0, bottom - top) * padded[1] / size[1]


def box_center(answer: dict, size: tuple[int, int], padded: tuple[int, int]) -> tuple[float, float] | None:
    box = answer.get("box") or {}
    try:
        left, right, top, bottom = (probability(box.get(k)) for k in ("left", "right", "top", "bottom"))
    except ValueError:
        return None
    if not left < right or not top < bottom:
        return None
    (w, h), (pw, ph) = size, padded
    x, y = (left + right) / 2 * pw / w, (top + bottom) / 2 * ph / h
    return (x, y) if 0 <= x <= 1 and 0 <= y <= 1 else None  # padding must never become a click


def mark(jpeg: bytes, point: tuple[float, float]) -> bytes:
    """`jpeg` with a red circle drawn around `point` (0-1 of its width and height)."""
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    cv2.circle(image, (int(point[0] * w), int(point[1] * h)), max(12, w // 60), (0, 0, 255), max(3, w // 400))
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("mark encoding failed")
    return buf.tobytes()


def grid_pitch(jpeg: bytes, point: tuple[float, float], axis: int, reach: float = 0.2, band: float = 0.03,
               line: float = 12) -> float | None:
    """The spacing of the grid lines around `point` across rows (`axis` 0) or columns (1), as a fraction of the
    screen: the usual gap between the lines that cross a band, the median over five bands around it. A band's median
    keeps those lines, which cross all of it, and drops the text in the cells; a band over the grid's labels reads
    off (Calc's columns from row 1: 34 px of 73), the others outvote it. On 6 Calc screens, 120 points each, every
    row (13.6 px) read within 4% and all but 6 columns (73 px), on row 1 of one; their repeats instead read 101 rows
    and 375 columns off (2026-10-20). None where no such lines show."""
    gray = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    gray = gray.astype(float) if axis == 0 else gray.astype(float).T
    n, m = gray.shape
    along, across = int(point[1 - axis] * n), int(point[axis] * m)
    rows = gray[max(0, along - int(reach * n)):along + int(reach * n)]
    gaps = [_line_gap(rows[:, max(0, c - int(band * m)):c + int(band * m)], line)
            for c in (across + int(k * 2 * band * m) for k in range(-2, 3)) if 0 <= c < m]
    gaps = [g for g in gaps if g is not None]
    return float(np.median(gaps)) / n if gaps else None


def _line_gap(strip: np.ndarray, line: float) -> float | None:
    """The usual gap in pixels between the lines darker than their cells that cross all of `strip`'s columns."""
    # shortcut: lines darker than their cells only (a light theme); a dark theme's lighter lines read none, and its
    # click steps by Jev's box. Accepting either read 26 columns off and 73 none on the light screens.
    profile = np.median(strip, 1)
    contrast = np.convolve(profile, np.ones(9) / 9, "same") - profile
    at = np.flatnonzero((contrast > line) & (contrast >= np.roll(contrast, 1)) & (contrast >= np.roll(contrast, -1)))
    gaps = np.diff(at)
    gaps = gaps[gaps >= 3]
    if len(gaps) < 2:
        return None
    usual = gaps[np.abs(gaps - np.median(gaps)) <= 0.15 * np.median(gaps)]
    return float(usual.mean()) if len(usual) >= 2 else None


def _crop_around(jpeg: bytes, point: tuple[float, float],
                 window: float = ZOOM_WINDOW) -> tuple[bytes, tuple[float, float, float, float]]:
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    side = window * max(w, h)
    half_w, half_h = min(side, w) / 2, min(side * 0.66, h) / 2
    cx = min(max(point[0] * w, half_w), w - half_w)
    cy = min(max(point[1] * h, half_h), h - half_h)
    x0, y0, x1, y1 = int(cx - half_w), int(cy - half_h), int(cx + half_w), int(cy + half_h)
    # No larger than Jev takes: a wide target's crop (Gmail's first email) blew up to 2560 px for Jev to shrink back,
    # 0.18 s for its two zooms on the Pi, 0.07 s at Jev's size (2026-10-20).
    scale = min(ZOOM_SCALE, JEV_SIDE / max(x1 - x0, y1 - y0))
    crop = cv2.resize(image[y0:y1, x0:x1], None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("crop encoding failed")
    return buf.tobytes(), (x0 / w, y0 / h, x1 / w, y1 / h)
