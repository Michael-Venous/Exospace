"""Runs ON the test laptop (copied there by `laptop.py`): a USB-HID stand-in over SSH. Standard library only.

It creates a virtual keyboard and an absolute mouse through /dev/uinput, the same two devices the Pico
presents (PLAN.md §5.1: absolute 0-32767), and takes screenshots with spectacle. One JSON command per stdin
line, PLAN.md §5.1 ops plus `grab`; each reply is a JSON line, a grab's PNG follows as that many raw bytes.
"""

import fcntl
import json
import os
import struct
import subprocess
import sys
import time

EV_SYN, EV_KEY, EV_REL, EV_ABS = 0, 1, 2, 3
REL_WHEEL, ABS_X, ABS_Y = 8, 0, 1
BTN = {"left": 0x110, "right": 0x111, "middle": 0x112}
HID_MAX = 32767
UI_SET_EVBIT, UI_SET_KEYBIT, UI_SET_RELBIT, UI_SET_ABSBIT = 0x40045564, 0x40045565, 0x40045566, 0x40045567
UI_DEV_SETUP, UI_ABS_SETUP, UI_DEV_CREATE, UI_DEV_DESTROY = 0x405C5503, 0x401C5504, 0x5501, 0x5502
NAMED = {"ESC": 1, "BACKSPACE": 14, "TAB": 15, "ENTER": 28, "CTRL": 29, "SHIFT": 42, "ALT": 56, "SPACE": 57,
         "CAPSLOCK": 58, "HOME": 102, "UP": 103, "PAGEUP": 104, "LEFT": 105, "RIGHT": 106, "END": 107, "DOWN": 108,
         "PAGEDOWN": 109, "INSERT": 110, "DELETE": 111, "GUI": 125, "MENU": 127, "PRINTSCREEN": 99,
         **{f"F{i}": 58 + i for i in range(1, 11)}, "F11": 87, "F12": 88}
ROWS = {"1234567890-=": 2, "qwertyuiop[]": 16, "asdfghjkl;'`": 30, "zxcvbnm,./": 44}
CHARS = {c: (code + i, False) for row, code in ROWS.items() for i, c in enumerate(row)}
CHARS.update({"\\": (43, False), " ": (57, False), "\n": (28, False), "\t": (15, False)})
SHIFTED = dict(zip('!@#$%^&*()_+{}:"~<>?|', "1234567890-=[];'`,./\\"))
CHARS.update({c: (CHARS[base][0], True) for c, base in SHIFTED.items()})
CHARS.update({c.upper(): (CHARS[c][0], True) for c in "abcdefghijklmnopqrstuvwxyz"})


def device(name: str, keys: list[int], rel: list[int] = (), absolute: bool = False) -> int:
    fd = os.open("/dev/uinput", os.O_WRONLY | os.O_NONBLOCK)
    for ev in [EV_KEY, *([EV_REL] if rel else []), *([EV_ABS] if absolute else [])]:
        fcntl.ioctl(fd, UI_SET_EVBIT, ev)
    for key in keys:
        fcntl.ioctl(fd, UI_SET_KEYBIT, key)
    for axis in rel:
        fcntl.ioctl(fd, UI_SET_RELBIT, axis)
    if absolute:
        for axis in (ABS_X, ABS_Y):
            fcntl.ioctl(fd, UI_SET_ABSBIT, axis)
            fcntl.ioctl(fd, UI_ABS_SETUP, struct.pack("HH6i", axis, 0, 0, 0, HID_MAX, 0, 0, 0))
    fcntl.ioctl(fd, UI_DEV_SETUP, struct.pack("4H80sI", 0x03, 0x1209, 0xE505, 1, name.encode(), 0))
    fcntl.ioctl(fd, UI_DEV_CREATE)
    return fd


def emit(fd: int, *events: tuple[int, int, int]) -> None:
    now = time.time()
    sec, usec = int(now), int((now % 1) * 1e6)
    data = b"".join(struct.pack("llHHi", sec, usec, t, c, v) for t, c, v in [*events, (EV_SYN, 0, 0)])
    os.write(fd, data)


KEYBOARD = device("Exospace bench keyboard", sorted(set(NAMED.values()) | {c for c, _ in CHARS.values()}))
POINTER = device("Exospace bench pointer", list(BTN.values()), [REL_WHEEL], absolute=True)
time.sleep(0.5)  # the compositor picks the new devices up


def tap(codes: list[int]) -> None:
    for code in codes:
        emit(KEYBOARD, (EV_KEY, code, 1))
    for code in reversed(codes):
        emit(KEYBOARD, (EV_KEY, code, 0))


def key(names: list[str]) -> None:
    codes = []
    for name in names:
        up = name.upper()
        if up in NAMED:
            codes.append(NAMED[up])
        elif len(name) == 1 and name.lower() in CHARS:
            codes.append(CHARS[name.lower()][0])
        else:
            raise ValueError(f"unknown key {name!r}")
    tap(codes)


def type_text(text: str) -> None:
    for char in text:
        if char not in CHARS:
            raise ValueError(f"can't type {char!r}")
        code, shift = CHARS[char]
        tap([NAMED["SHIFT"], code] if shift else [code])
        time.sleep(0.004)


def move(x: int, y: int) -> None:
    emit(POINTER, (EV_ABS, ABS_X, x), (EV_ABS, ABS_Y, y))


def handle(cmd: dict) -> None:
    op = cmd["op"]
    if op == "key":
        key(cmd["keys"])
    elif op == "type":
        type_text(cmd["text"])
    elif op == "move":
        move(cmd["x"], cmd["y"])
    elif op in ("click", "tap"):
        move(cmd["x"], cmd["y"])
        button = BTN[cmd.get("button", "left")]
        for _ in range(cmd.get("count", 1)):
            emit(POINTER, (EV_KEY, button, 1))
            emit(POINTER, (EV_KEY, button, 0))
            time.sleep(0.03)
    elif op == "scroll":
        for _ in range(abs(cmd["dy"])):
            emit(POINTER, (EV_REL, REL_WHEEL, -1 if cmd["dy"] > 0 else 1))
            time.sleep(0.01)
    elif op == "swipe":
        move(cmd["x1"], cmd["y1"])
        emit(POINTER, (EV_KEY, BTN["left"], 1))
        for i in range(1, 11):
            move(cmd["x1"] + (cmd["x2"] - cmd["x1"]) * i // 10, cmd["y1"] + (cmd["y2"] - cmd["y1"]) * i // 10)
            time.sleep(cmd.get("ms", 300) / 10000)
        emit(POINTER, (EV_KEY, BTN["left"], 0))
    elif op == "release_all":
        for code in (NAMED["CTRL"], NAMED["SHIFT"], NAMED["ALT"], NAMED["GUI"]):
            emit(KEYBOARD, (EV_KEY, code, 0))
        for code in BTN.values():
            emit(POINTER, (EV_KEY, code, 0))
    elif op != "ping":
        raise ValueError(f"unsupported op {op!r}")


def grab() -> bytes:
    path = "/tmp/exo_grab.jpg"  # JPEG: a fifth of the PNG's bytes over SSH
    subprocess.run(["spectacle", "-b", "-n", "-f", "-o", path], check=True, timeout=3,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(path, "rb") as f:
        return f.read()


def main() -> None:
    out = sys.stdout.buffer
    for line in sys.stdin:
        cmd = json.loads(line)
        try:
            if cmd["op"] == "grab":
                image = grab()
                out.write((json.dumps({"ok": True, "bytes": len(image)}) + "\n").encode() + image)
            else:
                handle(cmd)
                out.write((json.dumps({"ok": True}) + "\n").encode())
        except Exception as exc:  # noqa: BLE001 - the error goes back to the caller
            out.write((json.dumps({"ok": False, "err": f"{type(exc).__name__}: {exc}"}) + "\n").encode())
        out.flush()
    for fd in (KEYBOARD, POINTER):
        fcntl.ioctl(fd, UI_DEV_DESTROY)


if __name__ == "__main__":
    main()
