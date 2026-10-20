"""Deterministic shortcuts invoked by the controller."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import math

from brain.capture import thumbnail
from brain.types import Action, ActionError, DeviceProfile

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
# Launchers whose search shows its result as the name is typed: Cinnamon's did by the first look after the typing
# returned (0.09-0.19 s), changing the screen too little to wait for (0.1-1.0 inside the menu), and a wait for a
# reaction held Enter 0.45 s (2026-10-20). shortcut: only Linux measured; add others once timed.
INSTANT_SEARCH = {"linux"}
APP_KEYS: dict[str, dict[str, list[str]]] = {"android": {name: keys for names, keys in _ANDROID.items() for name in names}}


def run_skill(
    name: str,
    args: dict[str, Any],
    profile: DeviceProfile,
    controller: Controller,
) -> None:
    """Run a named skill (hover, select, wait, open_app), checking cancellation between stages."""
    controller.cancel.check()
    if name == "wait":
        cap = args.get("s")  # an optional cap: the default waits up to WAIT_S for the screen to change
        if isinstance(cap, (int, float)) and cap > 0:
            controller.wait_for_change(min(float(cap), 10.0))
        else:
            controller.wait_for_change()
        return
    if name in ("hover", "select"):
        if profile.input == "touch":
            raise SkillUnavailable(f"{name} is unavailable on touch targets")
        x, y = args.get("x"), args.get("y")
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or not 0 <= v <= 1 for v in (x, y)):
            raise ActionError(f"{name} requires normalized finite x/y")
        if name == "hover":
            controller.hover(x, y)
        else:
            # A triple click selects the line or paragraph under the pointer, with no settle between its clicks.
            # Three single clicks: after a double click command, Chrome took the third as a new gesture and kept
            # one word ("Genghis" of the title "Genghis Khan", 2026-10-19).
            for _ in range(3):
                controller.execute(Action("click", x=x, y=y), profile)
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
    # When it never shows, typing goes wherever the focus is: a terminal off the captured screen ran
    # "web browser" (2026-10-19).
    if start is not None and not controller.wait_for_change(LAUNCHER_S, since=start):
        raise ActionError("the launcher didn't open, so nothing was typed")
    # Cinnamon's menu showed in one frame and took the name typed at once; a settle here waited 0.45 s on a
    # still screen (2026-10-20). Android's drawer still slides in before its search key.
    if profile.os in SEARCH_KEYS:
        controller.wait_settled()
        controller.hotkey(SEARCH_KEYS[profile.os])
        controller.wait_settled()
    controller.cancel.check()
    controller.type_text(app.strip())
    controller.wait_settled(react=profile.os not in INSTANT_SEARCH)
    controller.cancel.check()
    frame = controller.capture.grab()
    typed = thumbnail(frame) if frame is not None else None
    controller.hotkey(["ENTER"])
    # Cinnamon's menu stayed up 1.2 s after Enter, and that differed from the screen before the launch: a check
    # on it saw no app and launched LibreOffice again (2026-10-20). So first wait for the launcher to go.
    if typed is not None:
        controller.wait_for_change(APP_START_S, since=typed)
    # Apps take a second or two to map their window; the screen is "settled" before that, and a
    # check on that screen saw no app and launched it again (measured: three KCalc instances).
    # So wait until the screen differs from before the launch, not merely from the launcher.
    if start is not None:
        controller.wait_settled()
        controller.wait_for_change(APP_START_S, since=start)
