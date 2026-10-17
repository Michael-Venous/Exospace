"""Deterministic shortcuts invoked by the controller."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import math

from brain.types import ActionError, DeviceProfile

if TYPE_CHECKING:
    from brain.controller import Controller


class SkillUnavailable(ActionError):
    """The executor should fall back to ordinary screen actions."""


# Each OS's own launcher search: the one name-based app launcher every OS has.
OPEN_APP_KEYS: dict[str, list[str]] = {
    "macos": ["GUI", "SPACE"], "ios": ["GUI", "SPACE"],
    "windows": ["GUI"], "linux": ["GUI"], "chromeos": ["GUI"], "android": ["GUI"],
}


def run_skill(
    name: str,
    args: dict[str, Any],
    profile: DeviceProfile,
    controller: Controller,
) -> None:
    """Open an app through OS search, checking cancellation between stages."""
    controller.cancel.check()
    if name == "hover":
        if profile.input == "touch":
            raise SkillUnavailable("Hover is unavailable on touch targets")
        x, y = args.get("x"), args.get("y")
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or not 0 <= v <= 1 for v in (x, y)):
            raise ActionError("hover requires normalized finite x/y")
        controller.hover(x, y)
        return
    if name != "open_app":
        raise SkillUnavailable(f"Unknown skill: {name!r}")
    if profile.os not in OPEN_APP_KEYS:
        raise SkillUnavailable(f"open_app is unavailable on {profile.os}; use screen actions")
    app = args.get("name")
    if not isinstance(app, str) or not app.strip():
        raise ActionError("open_app requires a nonempty app name")
    if any(ord(char) < 32 or ord(char) == 127 for char in app):
        raise ActionError("open_app name must not contain control characters")

    controller.hotkey(OPEN_APP_KEYS[profile.os])
    controller.wait_settled()
    controller.cancel.check()
    controller.type_text(app.strip())
    controller.wait_settled()
    controller.cancel.check()
    if profile.os == "android":
        controller.hotkey(["DOWN"])  # ENTER alone starts a web search; DOWN focuses the top app
    controller.hotkey(["ENTER"])
