"""Jev triage for one spoken segment: act now, plan, or no action (bench/arch study 1, 6).

One Jev call classifies the utterance against the current screen, locates the target
it names, and flags irreversible requests. A second, zoomed call refines the target
point (fixed every near miss on small targets in the study). No regex or phrase rules.
With `solo` (JEV_SOLO_SKILLS=1, experimental), the same call can also route easy tasks (open
an app or site, type into a field, press a key, scroll, find on page) to skills with no LLM
wait (brain/agent/solo.py).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import cv2
import numpy as np

from brain.agent.jev import JevVision, probability
from brain.agent.solo import (FIRST_ONLY, INTENT_GATE, KEY_CRITERIA, SCROLL_CRITERIA, SOLO_CRITERIA, SPAN_SKILLS,
                              VALUE_GATE, Ends, Spans, first_ends, rest_after, solo_questions, spans)
from brain.cancel import CancelToken

Mode = Literal["act", "keys", "plan", "none"]
Gesture = Literal["click", "double_click", "right_click"]

GATE = 0.5  # presence/uniqueness, zoom presence, irreversible, approval: 0 unsafe acts at this gate
ZOOM_WINDOW = 0.30  # crop side as a fraction of the longer screen side
ZOOM_SCALE = 2.5

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
                "form, formatting, changing and saving firmware/BIOS settings, or a public action on the user's "
                "account (starring, liking, following, subscribing, voting, sharing)? Opening, navigating, "
                "selecting, typing a draft, adding to or removing from a cart, and creating something new are not.")


@dataclass(frozen=True)
class Triage:
    mode: Mode
    point: tuple[float, float] | None  # normalized, zoom-refined; mode "act", or "keys" for type_into
    irreversible: bool
    gesture: Gesture = "click"
    skill: str = ""  # mode "keys": a SOLO_CRITERIA intent
    text: str = ""  # mode "keys": copied utterance words, a KEY_CRITERIA id, or a scroll direction
    rest: str = ""  # solo: the words after the first instruction, triaged again on the next screen


def triage(jev: JevVision, jpeg: bytes, utterance: str, cancel: CancelToken, *,
           solo: bool = False, run_id: str = "", step: int = 0) -> Triage:
    options, ends = (spans(utterance), first_ends(utterance)) if solo else ({}, {})
    answers, size, padded = jev.evaluate(jpeg, triage_questions(options, ends, utterance), run_id=run_id,
                                         step=step, state={"utterance": utterance})
    return decide(jev, jpeg, utterance, answers, size, padded, options, cancel, ends=ends, run_id=run_id, step=step)


def triage_questions(options: Spans, ends: Ends | None = None, utterance: str = "") -> dict[str, dict]:
    """Triage's questions. With solo span `options`, the solo intents and their questions join, and
    the action questions judge only the first instruction (the irreversible check reads it all)."""
    questions = {
        "intent": {"type": "choice", "criteria": {**INTENT_CRITERIA, **(SOLO_CRITERIA if options else {})},
                   "instructions":
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
    if not options:
        return questions
    for key in ("intent", "present", "unique", "gesture", "target"):
        questions[key] = {**questions[key], "instructions": FIRST_ONLY + questions[key]["instructions"]}
    return {**questions, **solo_questions(options, ends or {}, utterance)}


def decide(jev: JevVision, jpeg: bytes, utterance: str, answers: dict, size: tuple[int, int],
           padded: tuple[int, int], options: Spans, cancel: CancelToken, *, ends: Ends | None = None,
           run_id: str = "", step: int = 0) -> Triage:
    irreversible = probability(answers["irreversible"].get("noul")) >= GATE
    choice = answers["intent"].get("choice")
    if choice == "no_action":
        return Triage("none", None, irreversible)
    first, rest = utterance, ""
    if options and ends:
        split = _first_instruction(answers.get("first_end", {}), ends, utterance)
        if split is None:  # unsure where the first instruction ends: the planner takes it all
            return Triage("plan", None, irreversible)
        first, rest = split
    result = _decide_first(jev, jpeg, first, answers, size, padded, options, choice, irreversible, cancel,
                           run_id=run_id, step=step)
    if result.mode == "keys" and result.skill in SPAN_SKILLS and result.text not in first:
        return Triage("plan", None, irreversible)  # words picked from a later instruction
    return replace(result, rest=rest) if result.mode in ("act", "keys") else result


def _first_instruction(answer: dict, ends: Ends, utterance: str) -> tuple[str, str] | None:
    """(first instruction, rest). The whole utterance as one instruction needs no confidence; a
    split needs VALUE_GATE, since a wrong split would act on half an instruction."""
    last = max(ends, key=ends.get)
    chosen = answer.get("choice", last)
    if chosen not in ends:
        return None
    if chosen != last and _chosen_probability(answer) < VALUE_GATE:
        return None
    return utterance[:ends[chosen]], rest_after(utterance, ends[chosen])


def _decide_first(jev: JevVision, jpeg: bytes, utterance: str, answers: dict, size: tuple[int, int],
                  padded: tuple[int, int], options: Spans, choice: str, irreversible: bool, cancel: CancelToken, *,
                  run_id: str = "", step: int = 0) -> Triage:
    """Route the (first) instruction: a solo skill, a Jev click, or the planner."""
    present = min(probability(answers["present"].get("noul")), probability(answers["unique"].get("noul")))
    point = _box_center(answers["target"], size, padded)
    if options and choice in SOLO_CRITERIA:
        solo = _solo(answers, choice, options, irreversible)
        if solo.skill != "type_into":
            return solo
        if present < GATE or point is None:  # typing needs exactly one visible field
            return Triage("plan", None, irreversible)
        cancel.check()
        refined = refine(jev, jpeg, utterance, point, run_id=run_id, step=step)
        return replace(solo, point=refined) if refined else Triage("plan", None, irreversible)
    if choice != "act_now" or present < GATE or point is None:
        return Triage("plan", None, irreversible)
    cancel.check()
    refined = refine(jev, jpeg, utterance, point, run_id=run_id, step=step)
    gesture = answers["gesture"].get("choice")
    gesture = gesture if gesture in GESTURE_CRITERIA else "click"
    return Triage("act", refined, irreversible, gesture) if refined else Triage("plan", None, irreversible)


def _solo(answers: dict, choice: str, options: Spans, irreversible: bool) -> Triage:
    """A skill only when Jev is sure of both the intent and its value; else the planner.
    type_into's field point is added by the caller after zoom refinement."""
    plan = Triage("plan", None, irreversible)
    if _chosen_probability(answers["intent"]) < INTENT_GATE:
        return plan
    if choice in SPAN_SKILLS:
        value_answer, allowed = answers.get("value_span", {}), options
    elif choice == "press_key":
        value_answer, allowed = answers.get("key", {}), KEY_CRITERIA
    else:
        value_answer, allowed = answers.get("scroll_direction", {}), SCROLL_CRITERIA
    value = value_answer.get("choice")
    if value not in allowed or _chosen_probability(value_answer) < VALUE_GATE:
        return plan
    if choice == "go_to_site" and probability(answers.get("browser_front", {}).get("noul", 0.0)) < GATE:
        return plan  # no browser in front: the planner opens one first
    return Triage("keys", None, irreversible, skill=choice, text=options[value] if choice in SPAN_SKILLS else value)


def _chosen_probability(answer: dict) -> float:
    """The chosen option's probability (falls back to the answer's confidence)."""
    value = (answer.get("probabilities") or {}).get(answer.get("choice"), answer.get("confidence", 0.0))
    try:
        return probability(value)
    except ValueError:
        return 0.0


def refine(jev: JevVision, jpeg: bytes, target: str, point: tuple[float, float], *,
           run_id: str = "", step: int = 0) -> tuple[float, float] | None:
    """Re-locate `target` in an enlarged crop around `point`; None if Jev no longer sees it."""
    crop, (x0, y0, x1, y1) = _crop_around(jpeg, point)
    questions = {
        "present": {"type": "noul", "instructions":
                    "This is a zoomed-in part of a screen. Is the control the `utterance` asks to click, "
                    "tap or open visible here?"},
        "target": {"type": "box", "instructions":
                   "This is a zoomed-in part of a screen. Locate exactly the control the `utterance` asks "
                   "to click, tap or open."},
    }
    answers, size, padded = jev.evaluate(crop, questions, run_id=run_id, step=step, state={"utterance": target})
    local = _box_center(answers["target"], size, padded)
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


def _box_center(answer: dict, size: tuple[int, int], padded: tuple[int, int]) -> tuple[float, float] | None:
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
