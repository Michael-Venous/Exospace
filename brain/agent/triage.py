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

from brain.agent.jev import JevVision, probability
from brain.cancel import CancelToken

Mode = Literal["act", "plan", "none", "scroll", "launch"]
Gesture = Literal["click", "double_click", "right_click"]

GATE = 0.5  # presence/uniqueness, zoom presence, irreversible, approval: 0 unsafe acts at this gate
ZOOM_WINDOW = 0.30  # crop side as a fraction of the longer screen side
ZOOM_SCALE = 2.5
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
IRREVERSIBLE_KINDS = ("do something irreversible or high-impact: "
                "sending or posting a message, deleting or emptying data, purchasing or paying, submitting a "
                "form, formatting, changing and saving firmware/BIOS settings, or a public action on the user's "
                "account (starring, liking, following, subscribing, voting, sharing)? Opening, navigating, "
                "selecting, typing a draft, adding to or removing from a cart, and creating something new are not.")
IRREVERSIBLE = "Would carrying out the `utterance` on this screen " + IRREVERSIBLE_KINDS


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
           run_id: str = "", step: int = 0) -> Triage:
    names = spans(utterance)
    questions = {
        "intent": {"type": "choice", "criteria": INTENT_CRITERIA, "instructions":
                   "The user is talking to an AI assistant that operates the device shown in `screen` with a "
                   "mouse/touch and keyboard. Classify the user's latest `utterance`."},
        "present": {"type": "noul", "instructions":
                    "Is there a control visible in `screen` that the `utterance` asks to click, tap or open right now?"},
        "unique": {"type": "noul", "instructions":
                   "Does the `utterance` identify exactly one visible control in `screen`, with no ambiguity?"},
        "irreversible": {"type": "noul", "instructions": IRREVERSIBLE},
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
    answers, size, padded = jev.evaluate(jpeg, questions, run_id=run_id, step=step, state={"utterance": utterance})
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
    small = box_width(answers["target"], size, padded) < SMALL_TARGET
    return Triage("act", refined, irreversible, gesture, small=small) if refined else Triage("plan", None, irreversible)


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
           step: int = 0, state: dict | None = None, control: str = UTTERANCE_CONTROL) -> tuple[float, float] | None:
    """Re-locate `target` in an enlarged crop around `point`, then in a tighter one around that;
    None if Jev no longer sees it. On 49 OCR-labelled targets (Blender, Wikipedia, KCalc, GitHub)
    the point landed in the target 4 times from the full screen, 28 after one zoom and 35 after
    both (a median of three zooms: 27)."""
    state = state or {"utterance": target}
    coarse = _look(jev, jpeg, state, control, point, ZOOM_WINDOW, run_id, step)
    if coarse is None:
        return None
    return _look(jev, jpeg, state, control, coarse, FINE_WINDOW, run_id, step) or coarse


def _look(jev: JevVision, jpeg: bytes, state: dict, control: str, point: tuple[float, float], window: float,
          run_id: str, step: int) -> tuple[float, float] | None:
    crop, (x0, y0, x1, y1) = _crop_around(jpeg, point, window)
    questions = {
        "present": {"type": "noul", "instructions":
                    f"This is a zoomed-in part of a screen. Is {control} visible here?"},
        "target": {"type": "box", "instructions":
                   f"This is a zoomed-in part of a screen. Locate exactly {control}."},
    }
    answers, size, padded = jev.evaluate(crop, questions, run_id=run_id, step=step, state=state)
    local = box_center(answers["target"], size, padded)
    if local is None or probability(answers["present"].get("noul")) < GATE:
        return None
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
    }, run_id=run_id, step=step, state=state)
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
    return left * pw / w <= point[0] <= right * pw / w and top * ph / h <= point[1] <= bottom * ph / h


def box_width(answer: dict, size: tuple[int, int], padded: tuple[int, int]) -> float:
    box = answer.get("box") or {}
    try:
        left, right = probability(box.get("left")), probability(box.get("right"))
    except ValueError:
        return 1.0
    return max(0.0, right - left) * padded[0] / size[0]


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


def _crop_around(jpeg: bytes, point: tuple[float, float],
                 window: float = ZOOM_WINDOW) -> tuple[bytes, tuple[float, float, float, float]]:
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    side = window * max(w, h)
    half_w, half_h = min(side, w) / 2, min(side * 0.66, h) / 2
    cx = min(max(point[0] * w, half_w), w - half_w)
    cy = min(max(point[1] * h, half_h), h - half_h)
    x0, y0, x1, y1 = int(cx - half_w), int(cy - half_h), int(cx + half_w), int(cy + half_h)
    crop = cv2.resize(image[y0:y1, x0:x1], None, fx=ZOOM_SCALE, fy=ZOOM_SCALE, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("crop encoding failed")
    return buf.tobytes(), (x0 / w, y0 / h, x1 / w, y1 / h)
