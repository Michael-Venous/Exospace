"""Driving the target: HID backends that speak the PLAN.md §5.1 protocol.

`PicoHID` talks to the Pico over UART. `LocalHID` is the development
stand-in that performs the same commands on the machine you're on.
"""

from __future__ import annotations

import itertools
import json
import platform
import subprocess
import threading
import time
from typing import Any, Callable, Protocol

import serial

from brain.types import ActionError

HID_MAX = 32767
ACK_TIMEOUT_S = 2.0
PING_ATTEMPTS = 3

EventCallback = Callable[[str], None]  # receives "mounted" / "unmounted"
VOICE_WINDOW = "Exospace Voice"  # brain/voice/layer.html's title


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
        """Refuse keys bound for the voice app: it keeps focus after the user talks to it, which
        the model cannot see, and a typed space there barges in on the run. A click moves focus."""
        keys = set(cmd.get("keys", ()))
        if "GUI" in keys:
            self.menu_open = True
        elif self.menu_open:
            self.menu_open = not keys & {"ENTER", "ESC"}
        elif _focused_title() == VOICE_WINDOW:
            raise ActionError(f"keyboard focus is on the {VOICE_WINDOW} window; click the field to type into first")

    def _key(self, name: str) -> str:
        return self._keymap.get(name) or name.lower()

    def _px(self, x: int, y: int) -> tuple[int, int]:
        area = self._area or dict(zip(("left", "top", "width", "height"), (0, 0, *self._gui.size())))
        return (area["left"] + round(x / HID_MAX * (area["width"] - 1)),
                area["top"] + round(y / HID_MAX * (area["height"] - 1)))


def _focused_title() -> str:
    """The focused X11 window's title, or "" where xprop cannot tell."""
    try:
        root = subprocess.run(["xprop", "-root", "_NET_ACTIVE_WINDOW"], capture_output=True, text=True, timeout=1)
        name = subprocess.run(["xprop", "-id", root.stdout.split()[-1], "_NET_WM_NAME"],
                              capture_output=True, text=True, timeout=1)
    except (OSError, subprocess.SubprocessError, IndexError):
        return ""
    return name.stdout.partition(" = ")[2].strip().strip('"')
