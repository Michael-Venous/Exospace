"""The test laptop as a capture + HID target over SSH (study 16, bench only).

The agent runs here; the laptop (CachyOS, KDE Plasma on Wayland; address in the Assistant hub's MACHINES.md)
only shows its screen and takes input. `LaptopTarget` has the browser target's interface (`brain/browser.py`):
`grab()` is a spectacle screenshot, `send()` is a PLAN.md §5.1 HID command, both through `laptop_hid.py`
running on the laptop (a uinput keyboard + absolute mouse, as the Pico presents). `shell()` runs a command
there for test setup and oracles only; the agent never sees it.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from brain.hid import HIDError

HOST = "mike@100.103.48.85"
SESSION = ("XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-1 DISPLAY=:0 "
           "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus")
HELPER = Path(__file__).with_name("laptop_hid.py")


def shell(script: str, timeout: float = 60) -> str:
    """A bash script run in the laptop's desktop session (test setup and oracles only), its stdout."""
    done = subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, f"env {SESSION} bash -s"], input=script,
                          capture_output=True, text=True, timeout=timeout)
    return done.stdout


class LaptopTarget:
    def __init__(self) -> None:
        subprocess.run(["scp", "-q", str(HELPER), f"{HOST}:.cache/exo-bench/hid.py"], check=True, timeout=30)
        self._ssh = subprocess.Popen(["ssh", "-o", "BatchMode=yes", HOST,
                                      f"env {SESSION} python3 -u ~/.cache/exo-bench/hid.py"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self._lock = threading.Lock()  # the executor, the settle checks and the agent's grabs share one channel
        self._open = True
        self.grabs: list[float] = []  # seconds each screenshot took: the harness's cost, which a capture card hasn't
        self._call({"op": "ping"})

    def _call(self, cmd: dict) -> bytes | None:
        with self._lock:
            self._ssh.stdin.write((json.dumps(cmd) + "\n").encode())
            self._ssh.stdin.flush()
            reply = json.loads(self._ssh.stdout.readline())
            data = self._ssh.stdout.read(reply["bytes"]) if reply.get("bytes") else None
        if not reply["ok"]:
            raise HIDError(reply["err"])
        return data

    def grab(self):
        started = time.monotonic()
        try:
            image = self._call({"op": "grab"})
        except HIDError:  # spectacle failed or timed out once: one more try
            image = self._call({"op": "grab"})
        self.grabs.append(time.monotonic() - started)
        return cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)

    def send(self, cmd: dict) -> None:
        self._call(cmd)

    def signal_present(self) -> bool:
        return self._open

    def is_mounted(self) -> bool:
        return self._open

    def on_event(self, callback) -> None:
        pass

    def close(self) -> None:
        self._open = False
        try:
            self._call({"op": "release_all"})
        finally:
            self._ssh.stdin.close()
            self._ssh.wait(timeout=10)
