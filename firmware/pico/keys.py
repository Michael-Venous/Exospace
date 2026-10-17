"""Key names used by the protocol (PLAN.md §5.1) -> adafruit_hid Keycode names."""

_ALIASES = {
    "CTRL": "CONTROL", "ESC": "ESCAPE", "RETURN": "ENTER",
    "GUI": "GUI", "CMD": "GUI", "WIN": "GUI", "META": "GUI", "SUPER": "GUI",
    "UP": "UP_ARROW", "DOWN": "DOWN_ARROW", "LEFT": "LEFT_ARROW", "RIGHT": "RIGHT_ARROW",
    "PAGEUP": "PAGE_UP", "PAGEDOWN": "PAGE_DOWN",
    "0": "ZERO", "1": "ONE", "2": "TWO", "3": "THREE", "4": "FOUR",
    "5": "FIVE", "6": "SIX", "7": "SEVEN", "8": "EIGHT", "9": "NINE",
}


def resolve(name: str, keycode) -> int:
    """Look up a key name on the Keycode class. Raises ValueError if unknown."""
    attr = _ALIASES.get(name.upper(), name.upper())
    code = getattr(keycode, attr, None)
    if not isinstance(code, int):
        raise ValueError("unknown key " + name)
    return code
