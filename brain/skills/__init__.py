"""Deterministic shortcuts invoked by the controller."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import math

from brain.capture import thumbnail
from brain.types import ActionError, DeviceProfile

if TYPE_CHECKING:
    from brain.controller import Controller


class SkillUnavailable(ActionError):
    """The executor should fall back to ordinary screen actions."""


APP_START_S = 4.0  # longest wait for a launched app's window to appear
LAUNCHER_S = 2.0  # longest wait for the launcher to appear after its key
# The desktop launcher search; on phones and other targets the model uses the screen instead.
OPEN_APP_KEYS: dict[str, list[str]] = {
    "macos": ["GUI", "SPACE"], "windows": ["GUI"], "linux": ["GUI"], "chromeos": ["GUI"], "android": ["GUI"],
}
# A launcher that needs a second key before its search takes typing: Android's app drawer (measured on a
# Samsung through the Pico: Meta opens the drawer, Ctrl+F its search; touch input didn't register there).
SEARCH_KEYS: dict[str, list[str]] = {"android": ["CTRL", "F"]}
# System shortcuts for apps a launcher search finds but won't open by keyboard (Samsung's search ignored Enter).
# Android's own: Meta+I settings, +S messages, +B the default browser, +C contacts, +L calendar, +E email.
_ANDROID = {("settings",): ["GUI", "I"], ("messages", "messaging", "texts", "sms", "google messages"): ["GUI", "S"],
            ("browser", "brave", "chrome", "web browser", "internet"): ["GUI", "B"], ("contacts",): ["GUI", "C"],
            ("calendar",): ["GUI", "L"], ("email", "gmail", "mail"): ["GUI", "E"]}
APP_KEYS: dict[str, dict[str, list[str]]] = {"android": {name: keys for names, keys in _ANDROID.items() for name in names}}


def run_skill(
    name: str,
    args: dict[str, Any],
    profile: DeviceProfile,
    controller: Controller,
) -> None:
    """Run a named skill (hover, wait, open_app), checking cancellation between stages."""
    controller.cancel.check()
    if name == "wait":
        cap = args.get("s")  # an optional cap: the default waits up to WAIT_S for the screen to change
        if isinstance(cap, (int, float)) and cap > 0:
            controller.wait_for_change(min(float(cap), 10.0))
        else:
            controller.wait_for_change()
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

    shortcut = APP_KEYS.get(profile.os, {}).get(app.strip().lower())
    if shortcut:
        controller.hotkey(shortcut)
        controller.wait_settled()
        return
    frame = controller.capture.grab()
    start = thumbnail(frame) if frame is not None else None
    controller.hotkey(OPEN_APP_KEYS[profile.os])
    # Type only once the launcher shows: through the capture card the screen looked "settled" before the
    # launcher appeared, the name went nowhere, and Enter opened the launcher's first entry (an emulator).
    if start is not None:
        controller.wait_for_change(LAUNCHER_S, since=start)
    controller.wait_settled()
    if profile.os in SEARCH_KEYS:
        controller.hotkey(SEARCH_KEYS[profile.os])
        controller.wait_settled()
    controller.cancel.check()
    controller.type_text(app.strip())
    controller.wait_settled()
    controller.cancel.check()
    controller.hotkey(["ENTER"])
    # Apps take a second or two to map their window; the screen is "settled" before that, and a
    # check on that screen saw no app and launched it again (measured: three KCalc instances).
    # So wait until the screen differs from before the launch, not merely from the launcher.
    if start is not None:
        controller.wait_settled()
        controller.wait_for_change(APP_START_S, since=start)
