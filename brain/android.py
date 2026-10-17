"""Phone development target over adb: an Android emulator, or a phone with USB debugging,
stands in for a phone behind the capture card and Pico. Needs adb and one device."""
from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from brain.hid import HID_MAX, EventCallback, HIDError

# Android key codes (KEYCODE_*) for names in brain/keys.py that differ.
_KEYCODES = {"CTRL": "CTRL_LEFT", "SHIFT": "SHIFT_LEFT", "ALT": "ALT_LEFT", "GUI": "META_LEFT",
             "ESC": "ESCAPE", "BACKSPACE": "DEL", "DELETE": "FORWARD_DEL", "UP": "DPAD_UP",
             "DOWN": "DPAD_DOWN", "LEFT": "DPAD_LEFT", "RIGHT": "DPAD_RIGHT", "HOME": "MOVE_HOME",
             "END": "MOVE_END", "PAGEUP": "PAGE_UP", "PAGEDOWN": "PAGE_DOWN"}


class AndroidTarget:
    def __init__(self) -> None:
        self._adb = shutil.which("adb") or str(Path.home() / "Android/Sdk/platform-tools/adb")
        sizes = re.findall(r"(\d+)x(\d+)", self._shell("wm size"))
        if not sizes:
            raise HIDError("no Android screen size; check `adb devices`")
        self._size = tuple(map(int, sizes[-1]))  # an override size is listed last

    def grab(self):
        frame = cv2.imdecode(np.frombuffer(self._run("exec-out", "screencap", "-p"), np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            self._size = frame.shape[1], frame.shape[0]  # follows rotation
        return frame

    def signal_present(self) -> bool:
        return True

    def is_mounted(self) -> bool:
        return True

    def on_event(self, callback: EventCallback) -> None:
        pass  # adb reports no unplugging

    def send(self, cmd: dict[str, Any]) -> None:
        op = cmd["op"]
        if op in ("ping", "release_all"):
            return
        if op == "key":
            codes = " ".join("KEYCODE_" + _KEYCODES.get(key, key) for key in cmd["keys"])
            self._shell(("input keycombination " if " " in codes else "input keyevent ") + codes)
        elif op == "type":
            # shortcut: adb types "%s" as a space and drops non-ASCII; fine for tests, the Pico types exactly
            self._shell("input text " + shlex.quote(cmd["text"]))
        elif op == "tap":
            self._shell("input tap %d %d" % self._px(cmd["x"], cmd["y"]))
        elif op == "swipe":
            self._shell("input swipe %d %d %d %d %d" % (*self._px(cmd["x1"], cmd["y1"]),
                                                        *self._px(cmd["x2"], cmd["y2"]), cmd["ms"]))
        elif op == "scroll":
            center = self._px(HID_MAX // 2, HID_MAX // 2)
            self._shell("input mouse scroll %d %d --axis VSCROLL,%d" % (*center, -cmd["dy"]))
        else:
            raise HIDError(f"Unsupported Android input: {op}")

    def close(self) -> None:
        pass  # adb holds no keys down

    def _px(self, x: int, y: int) -> tuple[int, int]:
        return round(x / HID_MAX * (self._size[0] - 1)), round(y / HID_MAX * (self._size[1] - 1))

    def _shell(self, command: str) -> str:
        return self._run("shell", command).decode()

    def _run(self, *args: str) -> bytes:
        try:
            done = subprocess.run([self._adb, *args], capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise HIDError(f"adb failed: {exc}") from None
        if done.returncode:
            raise HIDError(f"adb {args[0]} failed: {done.stderr.decode().strip()[:200]}")
        return done.stdout
