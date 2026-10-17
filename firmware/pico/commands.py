"""Protocol ops (PLAN.md §5.1). Hardware-free: devices are passed in."""

from reports import BUTTON_LEFT, BUTTON_RIGHT, HID_MAX

MAX_TYPE_CHARS = 32
MAX_SWIPE_MS = 2000
BUTTONS = {"left": BUTTON_LEFT, "right": BUTTON_RIGHT}


def _coord(value) -> int:
    if not isinstance(value, int) or not 0 <= value <= HID_MAX:
        raise ValueError("coordinate must be an int in 0-%d" % HID_MAX)
    return value


class Commands:
    """Runs one command dict and returns its reply dict.

    keyboard: press_keys(names), type_text(text), release_all()
    mouse:    click(x, y, buttons, count), move(x, y), scroll(dy), release_all()
    touch:    tap(x, y), swipe(x1, y1, x2, y2, ms, abort), release_all()
    """

    def __init__(self, keyboard, mouse, touch, is_mounted, should_abort):
        self._keyboard = keyboard
        self._mouse = mouse
        self._touch = touch
        self._is_mounted = is_mounted
        self._should_abort = should_abort
        self._ops = {
            "ping": self._ping, "key": self._key, "type": self._type,
            "move": self._move, "click": self._click, "scroll": self._scroll,
            "tap": self._tap, "swipe": self._swipe, "release_all": self.release_all,
        }

    def handle(self, cmd: dict) -> dict:
        reply = {"id": cmd.get("id")}
        op = self._ops.get(cmd.get("op"))
        if op is None:
            reply.update(ok=False, err="unknown op")
            return reply
        try:
            extra = op(cmd)
        except KeyError as exc:
            reply.update(ok=False, err="missing field " + str(exc))
        except (ValueError, TypeError) as exc:
            reply.update(ok=False, err=str(exc))
        except OSError:
            reply.update(ok=False, err="usb send failed")
        else:
            reply["ok"] = True
            if extra:
                reply.update(extra)
        return reply

    def release_all(self, cmd=None) -> None:
        self._keyboard.release_all()
        self._mouse.release_all()
        self._touch.release_all()

    def _ping(self, cmd) -> dict:
        return {"mounted": self._is_mounted()}

    def _key(self, cmd) -> None:
        names = cmd["keys"]
        if not isinstance(names, list) or not names:
            raise ValueError("keys must be a non-empty list")
        self._keyboard.press_keys(names)

    def _type(self, cmd) -> None:
        text = cmd["text"]
        if not isinstance(text, str) or len(text) > MAX_TYPE_CHARS:
            raise ValueError("text must be a string of at most %d chars" % MAX_TYPE_CHARS)
        self._keyboard.type_text(text)

    def _move(self, cmd) -> None:
        self._mouse.move(_coord(cmd["x"]), _coord(cmd["y"]))

    def _click(self, cmd) -> None:
        button = BUTTONS.get(cmd.get("button", "left"))
        count = cmd.get("count", 1)
        if button is None:
            raise ValueError("button must be left or right")
        if count not in (1, 2):
            raise ValueError("count must be 1 or 2")
        self._mouse.click(_coord(cmd["x"]), _coord(cmd["y"]), button, count)

    def _scroll(self, cmd) -> None:
        dy = cmd["dy"]
        if not isinstance(dy, int):
            raise ValueError("dy must be an int")
        self._mouse.scroll(dy)

    def _tap(self, cmd) -> None:
        self._touch.tap(_coord(cmd["x"]), _coord(cmd["y"]))

    def _swipe(self, cmd) -> None:
        ms = cmd["ms"]
        if not isinstance(ms, int):
            raise ValueError("ms must be an int")
        ms = max(1, min(MAX_SWIPE_MS, ms))  # the Pi's model picks ms; keep it usable
        done = self._touch.swipe(
            _coord(cmd["x1"]), _coord(cmd["y1"]), _coord(cmd["x2"]), _coord(cmd["y2"]),
            ms, self._should_abort,
        )
        if not done:
            raise ValueError("aborted")
