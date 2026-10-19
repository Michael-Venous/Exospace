"""The push-to-talk button on the Pi header: GPIO17 (pin 11) to GND (pin 9), read with libgpiod's gpiomon.

Pressed pulls the line low against the internal pull-up. Each debounced press or release is passed on,
and the voice page treats it like holding or letting go of its orb (docs/hardware-bringup.md).
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from collections.abc import Callable

CHIP, LINE = "gpiochip0", 17
DEBOUNCE_S = 0.03  # contact bounce settles within this; a change faster than it is noise


def available() -> bool:
    return shutil.which("gpiomon") is not None


def watch(on_change: Callable[[bool], None], chip: str = CHIP, line: int = LINE) -> None:
    """Call on_change(pressed) for each debounced change, forever (run on its own thread): every edge
    restarts a DEBOUNCE_S timer, and the level it settles at is what counts."""
    # gpiomon block-buffers into a pipe: without stdbuf a press sat unsent (0 bytes written after many presses).
    line_buffered = ["stdbuf", "-oL"] if shutil.which("stdbuf") else []
    proc = subprocess.Popen([*line_buffered, "gpiomon", "--bias=pull-up", "--format=%e", chip, str(line)],
                            stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    lock = threading.Lock()
    state = {"level": False, "reported": False, "timer": None}

    def settle() -> None:
        with lock:
            if state["level"] == state["reported"]:
                return
            state["reported"] = state["level"]
        on_change(state["reported"])

    for event in proc.stdout:
        with lock:
            state["level"] = event.strip() == "0"  # falling edge: the button closed to ground
            if state["timer"] is not None:
                state["timer"].cancel()
            state["timer"] = threading.Timer(DEBOUNCE_S, settle)
            state["timer"].start()
