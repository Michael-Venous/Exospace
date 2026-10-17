"""Deterministic shortcuts invoked by the controller."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import math

from brain.keys import canonical
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
# Setup and boot-menu keys only: tapped dozens of times, ENTER or letters could change settings.
BOOT_KEYS = frozenset({"ESC", "DELETE", *(f"F{n}" for n in range(1, 13))})


def run_skill(
    name: str,
    args: dict[str, Any],
    profile: DeviceProfile,
    controller: Controller,
) -> None:
    """Run a named skill (hover, wait, boot_keys, open_app), checking cancellation between stages."""
    controller.cancel.check()
    if name == "wait":
        controller.wait_for_change()
        return
    if name == "boot_keys":
        keys = args.get("keys") or args.get("name")  # models also put the key in "text"
        try:
            keys = [canonical(key) for key in ([keys] if isinstance(keys, str) else keys)]
        except (AttributeError, TypeError, ValueError):
            keys = []
        if not 1 <= len(keys) <= 3 or not set(keys) <= BOOT_KEYS:
            raise ActionError("boot_keys needs 1-3 keys from ESC, DELETE, F1-F12")
        controller.arm_boot_keys(keys)
        return
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
