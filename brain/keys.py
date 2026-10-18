"""Canonical key names shared by the brain and the Pico (PLAN.md §5.1).

Models say "cmd", "Command", "win", "Return"… — everything is normalized here
so the Pico only ever sees one spelling.
"""

from __future__ import annotations

MODIFIERS = frozenset({"CTRL", "SHIFT", "ALT", "GUI"})
NAMED = frozenset(
    {
        "ENTER", "ESC", "TAB", "BACKSPACE", "DELETE", "INSERT", "SPACE",
        "UP", "DOWN", "LEFT", "RIGHT", "HOME", "END", "PAGEUP", "PAGEDOWN",
        *(f"F{n}" for n in range(1, 13)),
    }
)

ALIASES = {
    "CONTROL": "CTRL",
    "CMD": "GUI", "COMMAND": "GUI", "WIN": "GUI", "WINDOWS": "GUI",
    "META": "GUI", "SUPER": "GUI", "OPTION": "ALT", "OPT": "ALT",
    "RETURN": "ENTER", "ESCAPE": "ESC", "DEL": "DELETE", "SPACEBAR": "SPACE",
    "PGUP": "PAGEUP", "PAGE_UP": "PAGEUP", "PGDN": "PAGEDOWN", "PAGE_DOWN": "PAGEDOWN",
    "ARROWUP": "UP", "ARROWDOWN": "DOWN", "ARROWLEFT": "LEFT", "ARROWRIGHT": "RIGHT",
    "UP_ARROW": "UP", "DOWN_ARROW": "DOWN", "LEFT_ARROW": "LEFT", "RIGHT_ARROW": "RIGHT",
}


def canonical(key: str) -> str:
    """Normalize one key name, or raise ValueError if we can't send it."""
    k = key.strip().upper().replace(" ", "").replace("-", "_")
    if k.endswith(("_L", "_R")) and len(k) > 2:  # X11 keysym sides: Control_L, Shift_R, Super_L
        k = k[:-2]
    k = ALIASES.get(k, k)
    if len(k) == 1 and k.isascii() and k.isalnum():
        return k
    if k in MODIFIERS or k in NAMED:
        return k
    raise ValueError(f"unsupported key: {key!r}")


def canonical_combo(keys: tuple[str, ...] | list[str]) -> list[str]:
    """Normalize a combo, accepting "ctrl+l" style entries too."""
    out: list[str] = []
    for entry in keys:
        out.extend(canonical(part) for part in entry.split("+") if part.strip())
    if not out:
        raise ValueError("empty key combo")
    return out
