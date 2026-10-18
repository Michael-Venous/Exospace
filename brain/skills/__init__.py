"""Deterministic shortcuts invoked by the controller."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
import math
import os
import re

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
    "macos": ["GUI", "SPACE"], "windows": ["GUI"], "linux": ["GUI"], "chromeos": ["GUI"],
}


def installed_apps(category: str | None = None) -> frozenset[str]:
    """This machine's app names, lowercased: the desktop entries a Linux launcher searches (none elsewhere),
    or only those in one freedesktop category ("WebBrowser")."""
    return frozenset(name for name, _, categories in desktop_entries() if category is None or category in categories)


def desktop_entries() -> list[tuple[str, set[str], set[str]]]:
    """The desktop entries a Linux launcher shows, as (name lowercased, the ids its windows carry lowercased:
    file name and StartupWMClass, its freedesktop Categories)."""
    # shortcut: Linux only; read the Start menu or /Applications when the local backend runs on Windows or macOS.
    dirs = [os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"),
            *(os.environ.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share").split(":")]
    entries = []
    for entry in (e for d in dirs if (Path(d) / "applications").is_dir() for e in (Path(d) / "applications").rglob("*.desktop")):
        try:
            main = entry.read_text(errors="ignore").split("\n[Desktop Action")[0]
        except OSError:
            continue
        if (name := _field(main, "Name")) and "true" not in (_field(main, "NoDisplay"), _field(main, "Hidden")):
            entries.append((name.lower(), {entry.stem.lower(), _field(main, "StartupWMClass").lower()} - {""},
                            set(filter(None, _field(main, "Categories").split(";")))))
    return entries


def _field(entry: str, key: str) -> str:
    return match[1].strip() if (match := re.search(rf"^{key}=(.*)$", entry, re.M)) else ""


def run_skill(
    name: str,
    args: dict[str, Any],
    profile: DeviceProfile,
    controller: Controller,
) -> None:
    """Run a named skill (hover, wait, open_app), checking cancellation between stages."""
    controller.cancel.check()
    if name == "wait":
        controller.wait_for_change(float(args["seconds"]) if args.get("seconds") else None)
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

    frame = controller.capture.grab()
    start = thumbnail(frame) if frame is not None else None
    controller.hotkey(OPEN_APP_KEYS[profile.os])
    # Type only once the launcher shows: through the capture card the screen looked "settled" before the
    # launcher appeared, the name went nowhere, and Enter opened the launcher's first entry (an emulator).
    if start is not None:
        controller.wait_for_change(LAUNCHER_S, since=start)
    controller.wait_settled()
    controller.cancel.check()
    controller.type_text(app.strip())
    controller.wait_settled()
    controller.cancel.check()
    controller.hotkey(["ENTER"])
    # Apps take a second or two to map their window; the screen is "settled" before that, and a
    # check on that screen saw no app and launched it again (measured: three KCalc instances).
    # So wait until the screen differs from before the launch, not merely from the launcher. Keys chained
    # after a launch run blind, so no window (an empty search just closes the launcher) drops them.
    if start is not None:
        controller.wait_settled()
        if not controller.wait_for_change(APP_START_S, since=start):
            raise ActionError(f"no window opened for {app.strip()!r}; look at the screen and continue from it")
