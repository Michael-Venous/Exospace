"""The test laptop as a capture + HID target (study 16, bench only).

The whole test runs on the laptop itself (CachyOS, KDE Plasma on Wayland; address in the Assistant hub's
MACHINES.md), so no SSH sits between the agent and the screen; `laptop_run.sh` starts runs from the workstation.
`LaptopTarget` has the browser target's interface (`brain/browser.py`): `grab()` is a spectacle screenshot,
`send()` is a PLAN.md §5.1 HID command, both through `laptop_hid.py` (a uinput keyboard + absolute mouse, as
the Pico presents) in a child process. `shell()` runs a command in the desktop session for test setup and
oracles only; the agent never sees it.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from brain.hid import HIDError

RUNTIME = f"/run/user/{os.getuid()}"
SESSION = {"XDG_RUNTIME_DIR": RUNTIME, "DISPLAY": ":0", "DBUS_SESSION_BUS_ADDRESS": f"unix:path={RUNTIME}/bus",
           # the desktop session's socket (wayland-1 on the laptop, wayland-0 on the workstation)
           "WAYLAND_DISPLAY": min(p.name for p in Path(RUNTIME).glob("wayland-[0-9]"))}
HELPER = Path(__file__).with_name("laptop_hid.py")


def shell(script: str, timeout: float = 60) -> str:
    """A bash script run in the laptop's desktop session (test setup and oracles only), its stdout."""
    done = subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, **SESSION})
    return done.stdout


class LaptopTarget:
    def __init__(self) -> None:
        self._hid = subprocess.Popen(["python3", "-u", str(HELPER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     env={**os.environ, **SESSION})
        self._lock = threading.Lock()  # the executor and the brain share one channel
        self._open = True
        self.grabs: list[float] = []  # seconds each screenshot took: the harness's cost, which a capture card hasn't
        self._call({"op": "ping"})

    def _call(self, cmd: dict) -> bytes | None:
        with self._lock:
            self._hid.stdin.write((json.dumps(cmd) + "\n").encode())
            self._hid.stdin.flush()
            reply = json.loads(self._hid.stdout.readline())
            data = self._hid.stdout.read(reply["bytes"]) if reply.get("bytes") else None
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
            self._hid.stdin.close()
            self._hid.wait(timeout=10)
