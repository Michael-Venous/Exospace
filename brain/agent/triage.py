"""Jev triage for one spoken segment: act now, plan, or no action (bench/arch study 1, 6).

One Jev call classifies the utterance against the current screen, locates the target
it names, and flags irreversible requests. A second, zoomed call refines the target
point (fixed every near miss on small targets in the study). No regex or phrase rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from brain.agent.jev import JevVision, probability
from brain.cancel import CancelToken

Mode = Literal["act", "plan", "none"]
Gesture = Literal["click", "double_click", "right_click"]

GATE = 0.5  # presence/uniqueness, zoom presence, irreversible, approval: 0 unsafe acts at this gate
ZOOM_WINDOW = 0.30  # crop side as a fraction of the longer screen side
ZOOM_SCALE = 2.5
UTTERANCE_CONTROL = "the control the `utterance` asks to click, tap or open"

INTENT_CRITERIA = {
    "act_now": "The utterance asks to click, tap, open or select ONE specific control that is visible on "
               "this screen right now, and that single pointer action fully satisfies the request.",
    "plan": "The utterance needs typing, keyboard keys, several steps, writing content, answering a question, "
            "an irreversible or destructive action, or its intended target is ambiguous or not visible.",
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
IRREVERSIBLE = ("Would carrying out the `utterance` on this screen do something irreversible or high-impact: "
                "sending or posting a message, deleting or emptying data, purchasing or paying, submitting a "
                "form, formatting, or changing and saving firmware/BIOS settings? Opening, navigating, "
                "selecting, typing a draft, adding to or removing from a cart, and creating something new are not.")


@dataclass(frozen=True)
class Triage:
    mode: Mode
    point: tuple[float, float] | None  # normalized, zoom-refined; only for mode "act"
    irreversible: bool
    gesture: Gesture = "click"


def triage(jev: JevVision, jpeg: bytes, utterance: str, cancel: CancelToken, *,
           run_id: str = "", step: int = 0) -> Triage:
    questions = {
        "intent": {"type": "choice", "criteria": INTENT_CRITERIA, "instructions":
                   "The user is talking to an AI assistant that operates the device shown in `screen` with a "
                   "mouse/touch and keyboard. Classify the user's latest `utterance`."},
        "present": {"type": "noul", "instructions":
                    "Is there a control visible in `screen` that the `utterance` asks to click, tap or open right now?"},
        "unique": {"type": "noul", "instructions":
                   "Does the `utterance` identify exactly one visible control in `screen`, with no ambiguity?"},
        "irreversible": {"type": "noul", "instructions": IRREVERSIBLE},
        "gesture": {"type": "choice", "criteria": GESTURE_CRITERIA, "instructions":
                    "Which pointer gesture carries out the `utterance` on the control it names in `screen`?"},
        "target": {"type": "box", "instructions":
                   "The single visible control in `screen` that the `utterance` asks to click, tap or open "
                   "right now. For an item in a list or grid, locate the whole item."},
    }
    answers, size, padded = jev.evaluate(jpeg, questions, run_id=run_id, step=step, state={"utterance": utterance},
                                         label="triage", gates={"present": GATE, "unique": GATE, "irreversible": GATE})
    irreversible = probability(answers["irreversible"].get("noul")) >= GATE
    choice = answers["intent"].get("choice")
    if choice == "no_action":
        return Triage("none", None, irreversible)
    present = min(probability(answers["present"].get("noul")), probability(answers["unique"].get("noul")))
    point = box_center(answers["target"], size, padded)
    if choice != "act_now" or present < GATE or point is None:
        return Triage("plan", None, irreversible)
    cancel.check()
    refined = refine(jev, jpeg, {"utterance": utterance}, point, run_id=run_id, step=step)
    gesture = answers["gesture"].get("choice")
    gesture = gesture if gesture in GESTURE_CRITERIA else "click"
    return Triage("act", refined, irreversible, gesture) if refined else Triage("plan", None, irreversible)


def refine(jev: JevVision, jpeg: bytes, state: dict, point: tuple[float, float], *, control: str = UTTERANCE_CONTROL,
           run_id: str = "", step: int = 0) -> tuple[float, float] | None:
    """Re-locate `control` (described in terms of `state`) in an enlarged crop around `point`; None if Jev
    no longer sees it."""
    crop, (x0, y0, x1, y1) = _crop_around(jpeg, point)
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
    if local is None or sure < GATE:
        return None
    # A zoomed box around the first point only confirms it: its center can be a gap inside a larger control
    # (a whole search result boxed, clicked between thumbnail and title). The zoom moves a first look that missed,
    # but only when sure: zooms at 0.65-0.80 moved "Search YouTube." off Chrome's search bar onto the Gmail
    # shortcut below it (8/10 replays); the zooms that found the new-tab "+" were 0.91.
    if sure < jev.min_presence or box_contains(answers["target"], size, padded,
                                               ((point[0] - x0) / (x1 - x0), (point[1] - y0) / (y1 - y0))):
        return point
    return x0 + local[0] * (x1 - x0), y0 + local[1] * (y1 - y0)


Reply = Literal["approve", "decline", "other"]


def answer_to_pending(jev: JevVision, jpeg: bytes, pending: str, reply: str, *,
                      run_id: str = "", step: int = 0) -> Reply:
    """How the user's reply answers a pending irreversible request: approve, decline, or
    neither (a new command or a correction, which is then handled as its own request)."""
    state = {"pending_request": pending, "reply": reply}
    answers, _, _ = jev.evaluate(jpeg, {
        "approve": {"type": "noul", "instructions":
                    "Does the user's `reply` clearly approve carrying out the `pending_request`?"},
        "decline": {"type": "noul", "instructions":
                    "Does the user's `reply` refuse, cancel or say no to the `pending_request`?"},
    }, run_id=run_id, step=step, state=state, label="approval", gates={"approve": GATE, "decline": GATE})
    if probability(answers["approve"].get("noul")) >= GATE:
        return "approve"
    if probability(answers["decline"].get("noul")) >= GATE:
        return "decline"
    return "other"


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


def box_contains(answer: dict, size: tuple[int, int], padded: tuple[int, int], point: tuple[float, float]) -> bool:
    """Whether a point (box_center's coordinates) falls in Jev's box answer."""
    box = answer.get("box") or {}
    (w, h), (pw, ph) = size, padded
    return (box.get("left", 1) * pw / w <= point[0] <= box.get("right", 0) * pw / w
            and box.get("top", 1) * ph / h <= point[1] <= box.get("bottom", 0) * ph / h)


def _crop_around(jpeg: bytes, point: tuple[float, float]) -> tuple[bytes, tuple[float, float, float, float]]:
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    side = ZOOM_WINDOW * max(w, h)
    half_w, half_h = min(side, w) / 2, min(side * 0.66, h) / 2
    cx = min(max(point[0] * w, half_w), w - half_w)
    cy = min(max(point[1] * h, half_h), h - half_h)
    x0, y0, x1, y1 = int(cx - half_w), int(cy - half_h), int(cx + half_w), int(cy + half_h)
    crop = cv2.resize(image[y0:y1, x0:x1], None, fx=ZOOM_SCALE, fy=ZOOM_SCALE, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("crop encoding failed")
    return buf.tobytes(), (x0 / w, y0 / h, x1 / w, y1 / h)
