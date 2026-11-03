"""A keyboard's Home key as push-to-talk, for a computer that is the brain and the target at once (the laptop rig).

The real keyboards are held by this process alone (EVIOCGRAB) and every key but Home is passed on through a virtual
keyboard, so Home talks and never reaches the app in front. Used by bench/arch/talk.py and, with EXO_TALK_KEY=home,
by the voice page (brain.voice.layer), where a press holds the orb as the Pi's GPIO button does.
"""

from __future__ import annotations

import fcntl
import os
import select
import struct
import time
from collections.abc import Callable
from pathlib import Path

# Home, and the keypad's 7/Home, which the laptop's Home key sends (code 71; Home while Num Lock is off).
HOME_KEYS = {102, 71}
EV_SYN, EV_KEY, EV_LED, KEY_A, KEY_MAX = 0, 1, 0x11, 30, 0x2FF
EVIOCGRAB, UI_SET_EVBIT, UI_SET_KEYBIT, UI_SET_LEDBIT = 0x40044590, 0x40045564, 0x40045565, 0x40045569
UI_DEV_SETUP, UI_DEV_CREATE, UI_DEV_DESTROY = 0x405C5503, 0x5501, 0x5502
EVENT = struct.Struct("llHHi")  # struct input_event on 64-bit Linux


def keyboards() -> list[str]:
    """Event devices of the physical keyboards (they have letter keys: not power buttons or hotkey strips), not
    virtual ones (the agent's, this script's pass-through, other software's)."""
    found = []
    for block in Path("/proc/bus/input/devices").read_text().split("\n\n"):
        fields = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        bits = int("".join(w.zfill(16) for w in fields.get("B: KEY", "0").split()), 16)
        if fields.get("S: Sysfs", "").startswith("/devices/virtual") or not bits >> KEY_A & 1:
            continue  # other software's virtual keyboards (this agent's, an input switcher) are left alone
        found += [f"/dev/input/{h}" for h in fields.get("H: Handlers", "").split() if h.startswith("event")]
    return found


class PassThrough:
    """The real keyboards held by this script alone (EVIOCGRAB), every key but Home sent on through a virtual one:
    Home talks to the agent and never reaches the app in front. The kernel lets go of them when this process ends,
    however it ends. Caps/Num Lock lights are passed back."""

    def __init__(self, paths: list[str]) -> None:
        if not paths:  # no keyboard plugged in: only the GPIO button talks, and there is nothing to pass on
            self.real, self.virtual = [], os.pipe()[0]  # a pipe that never becomes readable
            return
        self.real = [os.open(p, os.O_RDWR) for p in paths]
        self.virtual = os.open("/dev/uinput", os.O_RDWR | os.O_NONBLOCK)
        for kind in (EV_KEY, EV_LED):
            fcntl.ioctl(self.virtual, UI_SET_EVBIT, kind)
        for code in range(1, KEY_MAX):
            fcntl.ioctl(self.virtual, UI_SET_KEYBIT, code)
        for led in range(3):  # Num, Caps, Scroll Lock
            fcntl.ioctl(self.virtual, UI_SET_LEDBIT, led)
        fcntl.ioctl(self.virtual, UI_DEV_SETUP, struct.pack("4H80sI", 0x03, 0x1209, 0xE506, 1,
                                                             b"Exospace pass-through keyboard", 0))
        fcntl.ioctl(self.virtual, UI_DEV_CREATE)
        time.sleep(0.3)  # the compositor picks the new keyboard up before the real ones go quiet
        for fd in self.real:
            fcntl.ioctl(fd, EVIOCGRAB, 1)

    def forward(self, data: bytes) -> list[tuple[int, int]]:
        """Pass a real keyboard's events on, all but Home's; Home's (code, value) presses come back."""
        out, home = [], []
        for i in range(0, len(data) - EVENT.size + 1, EVENT.size):
            sec, usec, kind, code, value = EVENT.unpack_from(data, i)
            if kind == EV_KEY and code in HOME_KEYS:
                home.append((code, value))
            elif kind in (EV_KEY, EV_SYN):
                out.append(EVENT.pack(sec, usec, kind, code, value))
        if out:
            os.write(self.virtual, b"".join(out))
        return home

    def lights(self) -> None:
        """Lock lights the desktop set on the virtual keyboard, shown on the real ones."""
        try:
            data = os.read(self.virtual, EVENT.size * 16)
        except BlockingIOError:
            return
        for i in range(0, len(data) - EVENT.size + 1, EVENT.size):
            sec, usec, kind, code, value = EVENT.unpack_from(data, i)
            if kind == EV_LED:
                for fd in self.real:
                    os.write(fd, EVENT.pack(sec, usec, kind, code, value) + EVENT.pack(sec, usec, EV_SYN, 0, 0))

    def close(self) -> None:
        for fd in self.real:
            os.close(fd)  # closing lets go of the grab
        if self.real:  # without a keyboard, virtual is a plain pipe
            fcntl.ioctl(self.virtual, UI_DEV_DESTROY)
        os.close(self.virtual)


def watch(on_change: Callable[[bool], None]) -> None:
    """Call on_change(pressed) for each Home press and release, forever (run on its own thread). Needs read access
    to the keyboards (the input group) and /dev/uinput."""
    keys = PassThrough(keyboards())
    try:
        while True:
            ready, _, _ = select.select([*keys.real, keys.virtual], [], [])
            if keys.virtual in ready:
                keys.lights()
            for fd in ready:
                if fd == keys.virtual:
                    continue
                for _code, value in keys.forward(os.read(fd, EVENT.size * 64)):
                    if value != 2:  # key repeat
                        on_change(value == 1)
    finally:
        keys.close()
