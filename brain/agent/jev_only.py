"""Pure-Jev stand-in for race's planner (EXECUTOR=jev): no language model is called.

Race keeps everything Jev does alone: confident clicks, remembered remarks and the
irreversible-action gate. A request that needs a plan runs here one step per screen:
one batched Jev call judges completion, picks the next action and its target, and picks
the text to type from runs of the goal's own words (Jev selects; it never writes text).
A step Jev can't take, or one with no visible effect, fails and names EXECUTOR=race.
"""

from __future__ import annotations

import json
import re
from typing import cast

from brain.agent.executor import Plan, StepContext
from brain.agent.jev import COMPLETION_QUESTION, JevVision, probability
from brain.agent.jev_flash import COMPLETION_GATE
from brain.agent.prompts import OS_PROMPT
from brain.agent.triage import GATE, box_center
from brain.cancel import CancelToken
from brain.config import Config
from brain.types import Action, OSName

OS_CRITERIA = {
    "windows": "Microsoft Windows desktop or app.",
    "macos": "Apple macOS desktop or app.",
    "linux": "A Linux desktop (GNOME, KDE, Xfce and the like) or app.",
    "chromeos": "A Chromebook (ChromeOS shelf and launcher).",
    "android": "An Android phone or tablet.",
    "ios": "An iPhone or iPad.",
    "bios": "Firmware setup, a boot menu or another pre-OS screen.",
    "unknown": "None of the above can be told from this screen.",
}
NEXT_QUESTION = ("The user's `goal` is being carried out on the device in `screen`; `executed_steps` were already "
                 "done. Which single action comes next?")
NEXT_CRITERIA = {
    "click": "Click or tap one visible control: a button, link, tab, menu or list item, or an input field "
             "with nothing to type into it.",
    "fill": "Click an input field and type into it text that the goal gives word for word.",
    "double_click": "Open an icon on a computer desktop, or a file or folder in a desktop file manager.",
    "right_click": "Open the context menu of a visible item.",
    "type": "Type text that the goal gives word for word into the input field that is focused right now.",
    "enter": "Press Enter to submit or confirm the text that was just typed.",
    "scroll_down": "Scroll down to reveal something the goal needs that is below the visible area.",
    "scroll_up": "Scroll up to reveal something the goal needs that is above the visible area.",
    "stuck": "The next step needs reasoning, writing new text, information that isn't visible, a key other "
             "than Enter, or the goal is unclear.",
}
TEXT_QUESTION = ("Which text does the next step type? It must be copied word for word from the `goal`, without the "
                 "command words or field name around it. Choose none if the text must be computed, looked up, "
                 "translated or written rather than copied.")
REPLACE_QUESTION = ("Is the input the next step types into a single-line box, such as a search, address or form "
                    "field, that already holds text?")
MAX_SPANS = 250  # Jev's choice budget (bench/interpreter_screen.py)
DEADLINE_S = 5.0  # user decision: a request Jev alone can't finish in 5 s isn't reliable enough


class JevOnlyPlanner:
    def __init__(self, cfg: Config, cancel: CancelToken) -> None:
        self.jev = JevVision(cfg.typesafe_api_key, cancel, cfg.model_log, cfg.jev_model,
                             cfg.jev_timeout_s, cfg.jev_min_presence, cfg.jev_min_confidence)

    def acknowledge(self, executed: bool) -> None:
        pass  # no plan cursor: every step is chosen afresh from the screen and the history

    def next_actions(self, image: bytes, ctx: StepContext) -> Plan:
        texts = spans(ctx.goal)
        answers, size, padded = self.jev.evaluate(image, {
            "complete": {"type": "noul", "instructions": COMPLETION_QUESTION},
            "next": {"type": "choice", "instructions": NEXT_QUESTION, "criteria": NEXT_CRITERIA},
            "present": {"type": "noul", "instructions":
                        "Is the one control that the next step of the `goal` acts on visible in `screen`, "
                        "with no ambiguity?"},
            "target": {"type": "box", "instructions":
                       "The single visible control in `screen` that the next step of the `goal` acts on."},
            "text": {"type": "choice", "instructions": TEXT_QUESTION, "criteria": {"none": "No text to type."} |
                     {str(i): json.dumps(text, ensure_ascii=False) for i, text in enumerate(texts)}},
            "replace": {"type": "noul", "instructions": REPLACE_QUESTION},
        }, run_id=ctx.run_id, step=ctx.step, state={"goal": ctx.goal, "executed_steps": ctx.history})
        if probability(answers["complete"].get("noul")) >= COMPLETION_GATE:
            return Plan("Jev verified the completed goal", [Action("done", summary="Done.")], [])
        if "unchanged" in ctx.note:
            return _fail("the last step had no visible effect")  # Jev would choose it again
        step = answers["next"].get("choice")
        actions: list[Action] = []
        if step in ("click", "double_click", "right_click", "fill"):
            point = box_center(answers["target"], size, padded)
            # shortcut: no zoom refinement (triage's +0.2 s); add it if small targets get missed.
            if point is None or probability(answers["present"].get("noul")) < self.jev.min_presence:
                return _fail("its target isn't clearly visible")
            actions.append(Action("click" if step == "fill" else step, x=point[0], y=point[1]))
        if step in ("type", "fill"):
            choice = answers["text"].get("choice")
            if choice not in {str(i) for i in range(len(texts))}:
                return _fail("the text to type isn't in the request")
            if probability(answers["replace"].get("noul")) >= GATE and ctx.profile.os != "bios":
                # Select-all only in a filled single-line box: in a document it would replace everything.
                actions.append(Action("hotkey", keys=("GUI" if ctx.profile.os in ("macos", "ios") else "CTRL", "A")))
            actions.append(Action("type_text", text=texts[int(choice)]))
        if step == "enter":
            actions.append(Action("hotkey", keys=("ENTER",)))
        if step in ("scroll_down", "scroll_up") and ctx.profile.os != "bios":  # firmware has no wheel
            actions.append(Action("scroll", dy=5 if step == "scroll_down" else -5))
        if not actions:
            return _fail("it needs reasoning or content Jev can't supply")
        return Plan(f"Jev next step: {step}", actions, [])

    def classify_os(self, image: bytes) -> OSName:
        answers, _, _ = self.jev.evaluate(image, {"os": {"type": "choice", "instructions": OS_PROMPT,
                                                         "criteria": OS_CRITERIA}})
        os = answers["os"].get("choice")
        if os not in OS_CRITERIA:
            raise ValueError("invalid operating system classification")
        return cast(OSName, os)

    def close(self) -> None:
        self.jev.close()


def spans(goal: str) -> list[str]:
    """Each run of whole words in the goal without wrapping quotes or trailing punctuation,
    shortest first. Whole runs bind text better than separate first/last words (HANDOFF.md)."""
    words = list(re.finditer(r"\S+", goal))
    runs = (goal[words[a].start():words[a + n].end()].strip("\"'“”‘’,;.")
            for n in range(len(words)) for a in range(len(words) - n))
    # shortcut: a long goal loses its longest runs past the budget; split the goal if long literals get cut.
    return [run for run in dict.fromkeys(runs) if run][:MAX_SPANS]


def _fail(why: str) -> Plan:
    return Plan("Jev can't take the next step", [Action(
        "fail", reason=f"Jev alone can't take the next step: {why}. "
                       "The language-model planner (EXECUTOR=race) can.")], [])
