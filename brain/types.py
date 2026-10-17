"""Shared data types. Field names follow the contracts in PLAN.md §5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]  # BGR image, shape (height, width, 3)
# "bios" covers firmware setup, boot menus and other pre-OS screens.
OSName = Literal["windows", "macos", "linux", "chromeos", "android", "ios", "bios", "unknown"]
OS_NAMES: tuple[OSName, ...] = get_args(OSName)
InputMode = Literal["mouse", "touch"]


@dataclass(frozen=True)
class Rect:
    """Active screen area inside a captured frame, in frame pixels."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def crop(self, frame: Frame) -> Frame:
        return frame[self.y0 : self.y1, self.x0 : self.x1]

    def as_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass(frozen=True)
class DeviceProfile:
    """What we know about the connected target (PLAN.md §5.4)."""

    os: OSName
    rect: Rect
    input: InputMode
    connected_at: float

    def to_json(self) -> dict[str, Any]:
        return {
            "os": self.os,
            "rect": self.rect.as_list(),
            "input": self.input,
            "connected_at": self.connected_at,
        }


TERMINAL_ACTIONS = frozenset({"done", "ask_user", "fail"})
POINTER_ACTIONS = frozenset({"click", "double_click", "right_click", "tap"})
ACTION_TYPES = POINTER_ACTIONS | TERMINAL_ACTIONS | {
    "swipe",
    "type_text",
    "hotkey",
    "skill",
    "scroll",
    "remember",
}


class ActionError(ValueError):
    """An action is malformed or cannot be executed."""


def _unit(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ActionError(f"{name} must be a number in [0, 1]")
    return min(max(float(value), 0.0), 1.0)


@dataclass(frozen=True)
class Action:
    """One step for the controller (PLAN.md §5.3). Coordinates are normalized
    to [0, 1] within the active screen rect."""

    type: str
    x: float | None = None
    y: float | None = None
    x2: float | None = None
    y2: float | None = None
    ms: int = 300
    text: str = ""
    keys: tuple[str, ...] = ()
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    dy: int = 0
    key: str = ""
    value: Any = None
    summary: str = ""
    question: str = ""
    reason: str = ""

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Action:
        kind = data.get("type")
        if kind not in ACTION_TYPES:
            raise ActionError(f"unknown action type: {kind!r}")
        fields: dict[str, Any] = {"type": kind}
        if kind in POINTER_ACTIONS or kind == "swipe":
            fields["x"] = _unit(data.get("x"), "x")
            fields["y"] = _unit(data.get("y"), "y")
        if kind == "swipe":
            fields["x2"] = _unit(data.get("x2"), "x2")
            fields["y2"] = _unit(data.get("y2"), "y2")
            fields["ms"] = int(data.get("ms", 300))
        if kind == "type_text":
            fields["text"] = str(data.get("text", ""))
        if kind == "hotkey":
            keys = data.get("keys") or []
            if not keys:
                raise ActionError("hotkey needs keys")
            fields["keys"] = tuple(str(k) for k in keys)
        if kind == "skill":
            fields["name"] = str(data.get("name", ""))
            fields["args"] = dict(data.get("args") or {})
        if kind == "scroll":
            fields["dy"] = int(data.get("dy", 0))
        if kind == "remember":
            fields["key"] = str(data.get("key", "")).strip()
            fields["value"] = data.get("value")
            if not fields["key"]:
                raise ActionError("remember needs a key")
        for text_field in ("summary", "question", "reason"):
            if text_field in data:
                fields[text_field] = str(data[text_field])
        return cls(**fields)

    def brief(self) -> str:
        """Short human-readable form for logs and model history."""
        # 0-1000 like the models' own output, so they never copy a different scale back.
        if self.type in POINTER_ACTIONS:
            return f"{self.type}({self.x * 1000:.0f},{self.y * 1000:.0f})"
        if self.type == "swipe":
            return f"swipe({self.x * 1000:.0f},{self.y * 1000:.0f}->{self.x2 * 1000:.0f},{self.y2 * 1000:.0f})"
        if self.type == "type_text":
            shown = self.text if len(self.text) <= 40 else self.text[:40] + "…"
            return f"type_text({shown!r})"
        if self.type == "hotkey":
            return f"hotkey({'+'.join(self.keys)})"
        if self.type == "skill":
            return f"skill({self.name} {self.args})"
        if self.type == "scroll":
            return f"scroll({self.dy})"
        if self.type == "remember":
            return f"remember({self.key})"
        return self.type
