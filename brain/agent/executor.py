"""One-call-per-step screenshot executor (direct, and legacy-dual's fast layer);
routing metadata never reaches the HID controller."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast

from brain.agent.memory import Memory
from brain.agent.model import GeminiJSON, ZaiJSON, language_model
from brain.agent.prompts import (DONE_PROMPT, DONE_SCHEMA, DONE_TEMPLATE, OS_PROMPT, OS_SCHEMA,
                                 PLAN_SCHEMA, SKILLS, SYSTEM_PROMPT, USER_TEMPLATE)
from brain.cancel import CancelToken
from brain.config import Config
from brain.types import OS_NAMES, Action, ActionError, DeviceProfile, OSName


@dataclass(frozen=True)
class StepContext:
    goal: str
    profile: DeviceProfile
    memory: Memory
    history: list[str]
    step: int
    max_steps: int
    note: str = ""
    guidance: str = ""
    run_id: str = ""


@dataclass(frozen=True)
class Plan:
    reasoning: str
    actions: list[Action]
    rejected: list[str]
    needs_planner: bool = False
    message: str = ""
    refresh_screen: bool = False
    held: str = ""


class Executor(Protocol):
    def next_actions(self, image_jpeg: bytes, ctx: StepContext) -> Plan: ...

    def classify_os(self, image_jpeg: bytes) -> OSName: ...


ROUTING_PROMPT = """
You are the fast response and execution layer. Give a short user-facing message
about what you will do; never claim an action has happened before it has.
For a simple request, return the next actions directly with needs_planner=false.
If there is no planner guidance and the goal needs extended reasoning, synthesis,
multiple apps, or a long sequence, set needs_planner=true and actions=[] BEFORE
acting. Also delegate if the current guidance is insufficient or recovery requires
new reasoning. The planner will supply guidance; then YOU ground each step in a
fresh screenshot. Do not delegate repeatedly when the existing guidance suffices.
Only return done when the user's entire goal is visibly satisfied.
"""

FAST_SCHEMA = {
    "type": "object",
    "properties": {
        **PLAN_SCHEMA["properties"],
        "actions": {**PLAN_SCHEMA["properties"]["actions"], "minItems": 0},
        "needs_planner": {"type": "boolean"},
        "message": {"type": "string"},
    },
    "required": ["reasoning", "actions", "needs_planner", "message"],
}


class ModelExecutor:
    """One vision-model call per step (z.ai or Gemini, see `language_model`). With `routing`
    (legacy-dual) hard steps go to the planner; without it (direct) the model acts alone."""

    def __init__(self, cfg: Config, cancel: CancelToken, *, routing: bool = True) -> None:
        self.model = language_model(cfg, cancel)
        self.system, self.schema = ((SYSTEM_PROMPT + ROUTING_PROMPT, FAST_SCHEMA) if routing
                                    else (SYSTEM_PROMPT, PLAN_SCHEMA))

    def next_actions(self, image_jpeg: bytes, ctx: StepContext) -> Plan:
        data = self.model.generate(image_jpeg, context_prompt(ctx), self.system, self.schema,
                                   role="fast", run_id=ctx.run_id, step=ctx.step)
        plan = parse_plan(data)
        first = next((action for action in plan.actions if action.type != "remember"), None)
        # GLM claims done from its own history; once it has acted, the screen must show it.
        if first is not None and first.type == "done" and ctx.history:
            missing = self.missing(image_jpeg, ctx, first.summary)
            if missing:
                return replace(plan, actions=[a for a in plan.actions if a.type == "remember"], rejected=[
                    f"Not confirmed done: {missing}. Finish it or bring the result into view; "
                    "never repeat a send, purchase or deletion."])
        return plan

    def missing(self, image_jpeg: bytes, ctx: StepContext, claim: str) -> str:
        """What the screen does not yet show of the goal, or "" when the claim holds."""
        prompt = DONE_TEMPLATE.format(goal=ctx.goal, os=ctx.profile.os,
                                      facts=json.dumps(ctx.memory.facts) if ctx.memory.facts else "(none)",
                                      history=_lines(ctx.history[-5:]), claim=claim)
        data = self.model.generate(image_jpeg, prompt, DONE_PROMPT, DONE_SCHEMA,
                                   role="check", run_id=ctx.run_id, step=ctx.step)
        return "" if data.get("complete") is True else str(data.get("missing") or "not shown on screen")

    def classify_os(self, image_jpeg: bytes) -> OSName:
        return detect_os(self.model, image_jpeg)

    def close(self) -> None:
        self.model.close()


def detect_os(model: GeminiJSON | ZaiJSON, image_jpeg: bytes) -> OSName:
    """One model call naming the screen's OS, or "unknown" (PLAN.md §5.4)."""
    os = model.generate(image_jpeg, OS_PROMPT, OS_PROMPT, OS_SCHEMA, role="device").get("os", "unknown")
    if os not in OS_NAMES:
        raise ValueError("invalid operating system classification")
    return cast(OSName, os)


KEYBOARD_ONLY = (
    "\nThis device takes keyboard input only: click, double_click, right_click, tap, swipe, scroll and hover do "
    "nothing on it. Use hotkey and type_text. GUI+SPACE, then the app's name, then ENTER opens an app; GUI+H goes to "
    "the home screen. In Messages, GUI+N starts a new message: type the contact's name, ENTER, then type the message "
    "(ENTER sends it, so ask first). TAB and the arrow keys move between fields where the app allows it; ENTER or "
    "SPACE activates the focused control."
)


def context_prompt(ctx: StepContext) -> str:
    keyboard = KEYBOARD_ONLY if ctx.profile.input == "keyboard" else ""
    return USER_TEMPLATE.format(
        goal=ctx.goal, os=ctx.profile.os, input=ctx.profile.input,
        step=ctx.step, max_steps=ctx.max_steps,
        conversation=_lines(f"{t['role']}: {t['text']}" for t in ctx.memory.recent_turns()),
        facts=json.dumps(ctx.memory.facts) if ctx.memory.facts else "(none)",
        history=_lines(ctx.history[-12:]), note=ctx.note or "(none)",
    ) + keyboard + "\nPlanner guidance (use only when consistent with the current screen):\n" + (ctx.guidance or "(none)")


def parse_plan(data: dict[str, Any]) -> Plan:
    raw_actions = data.get("actions")
    if not isinstance(raw_actions, list):
        raise ValueError("executor must return an actions list")
    raw_actions = raw_actions[:5]  # GLM ignores maxItems; the rest is re-planned from a new screen
    delegate = data.get("needs_planner", False)
    if not isinstance(delegate, bool):
        raise ValueError("needs_planner must be boolean")
    actions: list[Action] = []
    rejected: list[str] = []
    for raw in raw_actions:
        try:
            if not isinstance(raw, dict):
                raise ActionError("action must be an object")
            actions.append(Action.from_json(_to_contract(raw)))
        except (ActionError, TypeError, ValueError, OverflowError) as exc:
            rejected.append(str(exc))
    # Never execute the tail of a partly invalid batch: it may depend on the rejected action.
    if rejected or delegate:
        actions = []
    return Plan(str(data.get("reasoning", "")), actions, rejected,
                needs_planner=delegate, message=str(data.get("message", "")))


def _to_contract(raw: dict[str, Any]) -> dict[str, Any]:
    if raw.get("type") in SKILLS:  # GLM often names the skill as the action type
        raw = {**raw, "type": "skill", "name": raw["type"]}
    elif raw.get("type") == "key":  # and a key press as "key"
        raw = {**raw, "type": "hotkey", "keys": raw.get("keys") or [raw.get("key", "")]}
    out: dict[str, Any] = {"type": raw.get("type")}
    for axis in ("x", "y", "x2", "y2"):
        if axis in raw:
            value = raw[axis]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1000:
                raise ActionError(f"{axis} must be in [0, 1000]")
            out[axis] = value / 1000
    for key in ("dy", "keys", "text", "ms"):
        if key in raw:
            out[key] = raw[key]
    kind = raw.get("type")
    message = str(raw.get("message", ""))
    if kind == "skill":
        out["name"] = raw.get("name", "")
        out["args"] = {"name": raw.get("text", "")}
    elif kind == "remember":
        out["key"] = raw.get("key", "")
        out["value"] = raw.get("text", "")
    elif kind == "done":
        out["summary"] = message
    elif kind == "ask_user":
        out["question"] = message
    elif kind == "fail":
        out["reason"] = message
    return out


def _lines(items: Any) -> str:
    return "\n".join(f"- {item}" for item in items) or "(none)"
