"""A real browser on a virtual X display, as a capture + HID target (study 15, bench only).

Xvfb (:99, 1280x800) with openbox and Google Chrome, unpacked into ~/.local/opt/xbench (see README); nothing
shows on the user's screen. `XTarget` has the browser target's interface (`brain/browser.py`): `grab()` is
the display's pixels (mss), `send()` turns HID commands into xdotool input. Each run gets a fresh Chrome
profile. The oracle reads the open tabs from Chrome's debugging port; the agent never sees it, only pixels.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

from brain.hid import HID_MAX, HIDError

ROOT = Path.home() / ".local/opt/xbench/root/usr"
DISPLAY = ":99"
WIDTH, HEIGHT = 1280, 800
DEBUG_PORT = 9333
KEYS = {"CTRL": "ctrl", "SHIFT": "shift", "ALT": "alt", "GUI": "super", "ENTER": "Return", "ESC": "Escape",
        "TAB": "Tab", "BACKSPACE": "BackSpace", "DELETE": "Delete", "SPACE": "space", "UP": "Up", "DOWN": "Down",
        "LEFT": "Left", "RIGHT": "Right", "HOME": "Home", "END": "End", "PAGEUP": "Prior", "PAGEDOWN": "Next"}


def _env() -> dict:
    return {**os.environ, "DISPLAY": DISPLAY, "LD_LIBRARY_PATH": str(ROOT / "lib")}


def start_display() -> list[subprocess.Popen]:
    """Xvfb and openbox, unless the display already runs."""
    if subprocess.run([ROOT / "bin/xdotool", "getdisplaygeometry"], env=_env(), capture_output=True).returncode == 0:
        return []
    xvfb = subprocess.Popen([ROOT / "bin/Xvfb", DISPLAY, "-screen", "0", f"{WIDTH}x{HEIGHT}x24", "-nolisten", "tcp"],
                            env=_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    wm = subprocess.Popen([ROOT / "bin/openbox"], env=_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)
    return [xvfb, wm]


def _visible(socket_url: str) -> bool:
    """Whether a tab is the one shown: only the tab in front is `visible` to its page."""
    from websockets.sync.client import connect

    if not socket_url:
        return False
    try:
        with connect(socket_url, open_timeout=2, max_size=None) as ws:
            ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {"expression": "document.visibilityState"}}))
            reply = json.loads(ws.recv(timeout=2))
        return reply.get("result", {}).get("result", {}).get("value") == "visible"
    except Exception:  # noqa: BLE001 - a tab still loading answers nothing: not the one in front
        return False


class XTarget:
    def __init__(self, urls: list[str]) -> None:
        self._profile = tempfile.mkdtemp(prefix="xbench-chrome-")
        self._chrome = subprocess.Popen(
            ["google-chrome-stable", f"--user-data-dir={self._profile}", "--no-first-run", "--no-default-browser-check",
             "--disable-gpu", "--ozone-platform=x11", "--window-position=0,0", f"--window-size={WIDTH},{HEIGHT}",
             "--password-store=basic", f"--remote-debugging-port={DEBUG_PORT}", "--disable-session-crashed-bubble",
             *urls], env=_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import mss
        self._mss = mss.MSS(display=DISPLAY) if hasattr(mss, "MSS") else mss.mss(display=DISPLAY)
        self._open = True
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not self.tabs():
            time.sleep(0.3)
        time.sleep(2.0)  # the first page draws

    def tabs(self) -> list[dict]:
        """Chrome's open pages (url, title, active), the tab in front first; for the oracle only."""
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json/list", timeout=2) as response:
                pages = [t for t in json.load(response) if t.get("type") == "page"]
        except OSError:
            return []
        for page in pages:
            page["active"] = _visible(page.get("webSocketDebuggerUrl", ""))
        return sorted(pages, key=lambda p: not p["active"])

    def grab(self):
        shot = self._mss.grab({"left": 0, "top": 0, "width": WIDTH, "height": HEIGHT})
        return np.ascontiguousarray(np.array(shot)[:, :, :3])

    def signal_present(self) -> bool:
        return self._open

    def is_mounted(self) -> bool:
        return self._open

    def on_event(self, callback) -> None:
        pass

    def _xdo(self, *args: str) -> None:
        subprocess.run([ROOT / "bin/xdotool", *args], env=_env(), check=True, capture_output=True)

    def _px(self, x: int, y: int) -> tuple[str, str]:
        return str(round(x / HID_MAX * (WIDTH - 1))), str(round(y / HID_MAX * (HEIGHT - 1)))

    def send(self, cmd: dict) -> None:
        op = cmd["op"]
        if op == "ping":
            return
        if op == "key":
            self._xdo("key", "--clearmodifiers", "+".join(KEYS.get(k, k.lower()) for k in cmd["keys"]))
        elif op == "type":
            self._xdo("type", "--delay", "8", "--", cmd["text"])
        elif op in {"click", "tap"}:
            button = "3" if cmd.get("button") == "right" else "1"
            self._xdo("mousemove", *self._px(cmd["x"], cmd["y"]), "click", "--repeat", str(cmd.get("count", 1)), button)
        elif op == "move":
            self._xdo("mousemove", *self._px(cmd["x"], cmd["y"]))
        elif op == "scroll":
            if cmd["dy"]:
                self._xdo("click", "--repeat", str(abs(cmd["dy"])), "5" if cmd["dy"] > 0 else "4")
        elif op == "swipe":
            self._xdo("mousemove", *self._px(cmd["x1"], cmd["y1"]), "mousedown", "1",
                      "mousemove", *self._px(cmd["x2"], cmd["y2"]), "mouseup", "1")
        elif op == "release_all":
            self._xdo("keyup", "ctrl", "shift", "alt", "super")
            self._xdo("mouseup", "1")
        else:
            raise HIDError(f"Unsupported desktop input: {op}")

    def close(self) -> None:
        self._open = False
        self._chrome.terminate()
        try:
            self._chrome.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._chrome.kill()
        self._mss.close()
        shutil.rmtree(self._profile, ignore_errors=True)
