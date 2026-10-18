"""Driving the target: HID backends that speak the PLAN.md §5.1 protocol.

`PicoHID` talks to the Pico over UART. `LocalHID` is the development
stand-in that performs the same commands on the machine you're on.
"""

from __future__ import annotations

import itertools
import json
import platform
import re
import subprocess
import threading
import time
from typing import Any, Callable, Iterator, Protocol

import serial

from brain.types import ActionError

HID_MAX = 32767
ACK_TIMEOUT_S = 2.0
PING_ATTEMPTS = 3

EventCallback = Callable[[str], None]  # receives "mounted" / "unmounted"
VOICE_WINDOW = "Exospace Voice"  # brain/voice/layer.html's title
# freedesktop Categories whose apps run keys meant for a page as commands, or edit code with them.
_NO_KEYS = frozenset({"TerminalEmulator", "TextEditor", "IDE"})


class HIDError(RuntimeError):
    pass


class HID(Protocol):
    def send(self, cmd: dict[str, Any]) -> None:
        """Send one command and wait for its acknowledgement."""
        ...

    def on_event(self, callback: EventCallback) -> None: ...

    def is_mounted(self) -> bool: ...

    def close(self) -> None: ...


class PicoHID:
    def __init__(self, port: str, baud: int) -> None:
        try:
            # Exclusive: a second opener (doctor while the service runs) gets a clear error
            # instead of silently stealing the replies and killing this link.
            self._serial = serial.Serial(port, baud, timeout=0.1, exclusive=True)
        except serial.SerialException as exc:
            raise HIDError(f"cannot open {port}: {exc} (is exospace-agent already using it?)") from exc
        self._ids = itertools.count(1)
        self._write_lock = threading.Lock()
        self._pending: dict[int, tuple[threading.Event, dict[str, Any]]] = {}
        self._callbacks: list[EventCallback] = []
        self._mounted = False
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, name="pico-uart", daemon=True)
        self._reader.start()
        self._serial.write(b"\n")  # end any stray partial line from power-up noise
        for attempt in range(PING_ATTEMPTS):
            try:
                reply = self._request({"op": "ping"})
                break
            except HIDError as exc:
                if attempt == PING_ATTEMPTS - 1:
                    self.close()
                    raise HIDError(f"no reply from Pico on {port}; check wiring and firmware") from exc
        self._mounted = bool(reply.get("mounted"))

    def send(self, cmd: dict[str, Any]) -> None:
        reply = self._request(cmd)
        if not reply.get("ok"):
            raise HIDError(f"{cmd.get('op')} failed: {reply.get('err', 'unknown error')}")

    def on_event(self, callback: EventCallback) -> None:
        self._callbacks.append(callback)

    def is_mounted(self) -> bool:
        return self._mounted

    def close(self) -> None:
        self._running = False
        self._reader.join(timeout=1.0)
        self._serial.close()

    def _request(self, cmd: dict[str, Any]) -> dict[str, Any]:
        cmd_id = next(self._ids)
        done = threading.Event()
        reply: dict[str, Any] = {}
        self._pending[cmd_id] = (done, reply)
        line = json.dumps({**cmd, "id": cmd_id}, separators=(",", ":")) + "\n"
        try:
            with self._write_lock:
                self._serial.write(line.encode())
            if not done.wait(ACK_TIMEOUT_S):
                raise HIDError(f"Pico did not acknowledge {cmd.get('op')}")
            return reply
        finally:
            self._pending.pop(cmd_id, None)

    def _read_loop(self) -> None:
        while self._running:
            try:
                raw = self._serial.readline()
            except (serial.SerialException, OSError):
                if not self._running:
                    return
                time.sleep(0.2)  # transient UART error: keep the link alive
                continue
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except ValueError:
                continue  # partial or garbled line
            if not isinstance(msg, dict):
                continue
            if "ev" in msg:
                self._handle_event(str(msg["ev"]))
            elif (entry := self._pending.get(msg.get("id", -1))) is not None:
                done, reply = entry
                reply.update(msg)
                done.set()

    def _handle_event(self, event: str) -> None:
        self._mounted = event == "mounted"
        for callback in self._callbacks:
            callback(event)


class LocalHID:
    """Development stand-in using pyautogui. Move the mouse to a screen corner
    to trigger pyautogui's emergency stop. With `keyboard`, key commands go to that
    device instead: some desktops (Cinnamon) ignore a faked Super, not a real one. Clicks map onto
    `area` (mss's left/top/width/height, the captured monitor), or the whole desktop without it."""

    def __init__(self, keyboard: HID | None = None, area: dict[str, int] | None = None) -> None:
        import pyautogui  # optional dependency, only needed for the local backend

        pyautogui.PAUSE = 0  # we do our own settling
        self._gui = pyautogui
        self._keyboard = keyboard
        self._area = area
        self.menu_open = False  # Super opened the desktop menu, which X never reports as focused
        gui_key = "command" if platform.system() == "Darwin" else "win"
        self._keymap = {
            "CTRL": "ctrl", "SHIFT": "shift", "ALT": "alt", "GUI": gui_key,
            "ENTER": "enter", "ESC": "esc", "TAB": "tab", "BACKSPACE": "backspace",
            "DELETE": "delete", "INSERT": "insert", "SPACE": "space",
            "UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "right",
            "HOME": "home", "END": "end", "PAGEUP": "pageup", "PAGEDOWN": "pagedown",
        }

    def send(self, cmd: dict[str, Any]) -> None:
        op = cmd["op"]
        gui = self._gui
        if op == "ping":
            return
        if op in ("key", "type"):
            self._check_focus(cmd)
        else:
            self.menu_open = False
        if self._keyboard and op in ("key", "type", "release_all"):
            self._keyboard.send(cmd)
            if op != "release_all":
                return
        if op == "key":
            gui.hotkey(*(self._key(k) for k in cmd["keys"]))
        elif op == "type":
            gui.write(cmd["text"])
        elif op == "move":
            gui.moveTo(*self._px(cmd["x"], cmd["y"]))
        elif op in ("click", "tap"):
            gui.click(*self._px(cmd["x"], cmd["y"]), clicks=cmd.get("count", 1),
                      button=cmd.get("button", "left"))
        elif op == "scroll":
            if spot := _scroll_spot(gui.position()):
                gui.moveTo(*spot)
            gui.scroll(-int(cmd["dy"]) * 5)  # pyautogui: positive = up
        elif op == "swipe":
            gui.moveTo(*self._px(cmd["x1"], cmd["y1"]))
            gui.dragTo(*self._px(cmd["x2"], cmd["y2"]), duration=cmd.get("ms", 300) / 1000)
        elif op == "release_all":
            for key in ("ctrl", "shift", "alt", self._keymap["GUI"]):
                gui.keyUp(key)
            gui.mouseUp()
        else:
            raise HIDError(f"unknown op {op!r}")

    def on_event(self, callback: EventCallback) -> None:
        pass  # the local machine never unplugs

    def is_mounted(self) -> bool:
        return True

    def close(self) -> None:
        if self._keyboard:
            self._keyboard.close()

    def _check_focus(self, cmd: dict[str, Any]) -> None:
        """Keep keys from the voice app: it keeps focus after the user talks to it, which the model
        cannot see, and a typed space there barges in on the run. They go to the window the user
        was in before it, else they are refused. A click moves focus."""
        keys = set(cmd.get("keys", ()))
        if "GUI" in keys:
            self.menu_open = True
        elif self.menu_open:
            self.menu_open = not keys & {"ENTER", "ESC"}
        elif _focused_title() == VOICE_WINDOW and not _focus_window_below():
            raise ActionError(f"keyboard focus is on the {VOICE_WINDOW} window; click the field to type into first")

    def _key(self, name: str) -> str:
        return self._keymap.get(name) or name.lower()

    def _px(self, x: int, y: int) -> tuple[int, int]:
        area = self._area or dict(zip(("left", "top", "width", "height"), (0, 0, *self._gui.size())))
        return (area["left"] + round(x / HID_MAX * (area["width"] - 1)),
                area["top"] + round(y / HID_MAX * (area["height"] - 1)))


def _focus_window_below() -> bool:
    """Focus the topmost window on this workspace other than the voice app; False if none took it."""
    try:
        if (window := _window_below()) is None or not _takes_keys(window):
            return False
        _run("wmctrl", "-i", "-a", window)
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return False
    for _ in range(10):  # activation lands a moment later; a key sent before it reaches the voice app
        if _focused_title() != VOICE_WINDOW:
            return True
        time.sleep(0.03)
    return False


def _takes_keys(window: str) -> bool:
    """Whether keys meant for the target can go to `window`: an app this machine's launcher names, and
    never a terminal or an editor (`_NO_KEYS`)."""
    from brain.skills import desktop_entries  # here: brain.skills imports brain.capture, which imports this

    ids = {v.lower() for v in re.findall(r'"([^"]+)"', _run("xprop", "-id", window, "WM_CLASS", "_GTK_APPLICATION_ID"))}
    kinds = [categories for _, entry_ids, categories in desktop_entries() if entry_ids & ids]
    return bool(kinds) and not any(categories & _NO_KEYS for categories in kinds)


def _scroll_spot(pointer: tuple[int, int]) -> tuple[int, int] | None:
    """Where a scroll reaches the window the run drives (the topmost but the voice app, which keys go to too):
    the middle of its part the voice app leaves bare, unless the pointer is on that part already. A scroll under
    a pointer left elsewhere scrolled nothing the agent saw (a terminal beside the new browser, 2026-10-18).
    None scrolls under the pointer."""
    try:
        stack = list(_stack())
        if (below := next((window for window, title in stack if title != VOICE_WINDOW), None)) is None:
            return None
        x, y, w, h = window_rect(below)
        for window, title in stack:
            if window == below:
                break
            x, y, w, h = _beside((x, y, w, h), window_rect(window))
    except (OSError, subprocess.SubprocessError, IndexError, ValueError, TypeError):
        return None
    if w <= 0 or h <= 0 or (x <= pointer[0] < x + w and y <= pointer[1] < y + h):
        return None
    return x + w // 2, y + h // 2


def _beside(rect: tuple[int, int, int, int], cover: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """The largest strip of `rect` that `cover` leaves bare, all as (left, top, width, height); `rect` when
    they don't overlap."""
    x, y, w, h = rect
    cx, cy, cw, ch = cover
    if cx >= x + w or cx + cw <= x or cy >= y + h or cy + ch <= y:
        return rect
    strips = [(x, y, cx - x, h), (cx + cw, y, x + w - cx - cw, h), (x, y, w, cy - y), (x, cy + ch, w, y + h - cy - ch)]
    return max(strips, key=lambda s: max(s[2], 0) * max(s[3], 0))


def _window_below() -> str | None:
    """The topmost window on this workspace other than the voice app."""
    return next((window for window, title in _stack() if title != VOICE_WINDOW), None)


def _stack() -> Iterator[tuple[str, str]]:
    """This workspace's windows as (X11 id, title), topmost first; never the desktop itself."""
    workspace = _run("xprop", "-root", "_NET_CURRENT_DESKTOP").split()[-1]
    listed = _listed()
    for window in reversed(re.findall(r"0x[0-9a-f]+", _run("xprop", "-root", "_NET_CLIENT_LIST_STACKING"))):
        on, title = listed.get(int(window, 16), ("", ""))
        if on in (workspace, "-1") and "_DESKTOP" not in _run("xprop", "-id", window, "_NET_WM_WINDOW_TYPE"):
            yield window, title


def window_rects(title: str) -> list[tuple[int, int, int, int]]:
    """Where each window whose title starts with `title` is, as `window_rect` gives it."""
    return [rect for window, (_, name) in _listed().items() if name.startswith(title) and (rect := window_rect(hex(window)))]


def rects_above(title: str) -> list[tuple[int, int, int, int]]:
    """Where the shown windows stacked above the topmost one titled `title` are: they cover it."""
    rects = []
    for window, name in _stack():
        if name.startswith(title):
            return rects
        if "IsViewable" in _run("xwininfo", "-id", window) and (rect := window_rect(window)):
            rects.append(rect)
    return []


def window_rect(window: str) -> tuple[int, int, int, int] | None:
    """`window`'s left, top, width and height in desktop pixels, frame included (`wmctrl -lG` misplaces
    framed windows); None once it's closed."""
    info = _run("xwininfo", "-id", window)
    found = [re.search(rf"{key}:\s*(-?\d+)", info) for key in ("Absolute upper-left X", "Absolute upper-left Y", "Width", "Height")]
    if not all(found):
        return None
    x, y, w, h = (int(m[1]) for m in found)  # type: ignore[index]
    extents = re.findall(r"\d+", _run("xprop", "-id", window, "_NET_FRAME_EXTENTS").partition("=")[2])
    left, right, top, bottom = (int(v) for v in extents) if len(extents) == 4 else (0, 0, 0, 0)
    return x - left, y - top, w + left + right, h + top + bottom


def _listed() -> dict[int, tuple[str, str]]:
    """`wmctrl -l`: each window as its X11 id -> (its workspace, its title)."""
    return {int(w, 16): (on, title) for w, on, _, title in
            ((line.split(None, 3) + [""])[:4] for line in _run("wmctrl", "-l").splitlines())}


def _run(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=1).stdout


def _focused_title() -> str:
    """The focused X11 window's title, or "" where xprop cannot tell."""
    try:
        root = subprocess.run(["xprop", "-root", "_NET_ACTIVE_WINDOW"], capture_output=True, text=True, timeout=1)
        name = subprocess.run(["xprop", "-id", root.stdout.split()[-1], "_NET_WM_NAME"],
                              capture_output=True, text=True, timeout=1)
    except (OSError, subprocess.SubprocessError, IndexError):
        return ""
    return name.stdout.partition(" = ")[2].strip().strip('"')
