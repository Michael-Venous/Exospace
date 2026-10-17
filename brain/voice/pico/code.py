"""Exospace Pico "hands": PLAN.md §5.1 JSON lines in, USB keyboard out. No logic beyond the protocol."""
import json

import supervisor
import usb_cdc
import usb_hid
from adafruit_hid.keyboard import Keyboard
from adafruit_hid.keyboard_layout_us import KeyboardLayoutUS
from adafruit_hid.keycode import Keycode

# shortcut: commands come over the USB data port and only keyboard ops exist, enough for the laptop test;
# the Pi build needs busio.UART on GP0/GP1, the mouse/touch descriptors and mounted/unmounted events.
port = usb_cdc.data
if port is None:
    raise RuntimeError("USB data port is off: copy boot.py, then unplug and replug the Pico")
port.timeout = None
keyboard = Keyboard(usb_hid.devices)
layout = KeyboardLayoutUS(keyboard)
# brain/keys.py canonical names that adafruit_hid spells differently.
NAMES = {"CTRL": "CONTROL", "ESC": "ESCAPE", "SPACE": "SPACEBAR", "UP": "UP_ARROW", "DOWN": "DOWN_ARROW",
         "LEFT": "LEFT_ARROW", "RIGHT": "RIGHT_ARROW", "PAGEUP": "PAGE_UP", "PAGEDOWN": "PAGE_DOWN"}
NAMES.update({str(i): n for i, n in enumerate(("ZERO", "ONE", "TWO", "THREE", "FOUR",
                                                "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"))})


def keycode(name):
    code = getattr(Keycode, NAMES.get(name, name), None)
    if not isinstance(code, int):
        raise ValueError("unsupported key " + repr(name))
    return code


def run(cmd):
    op = cmd["op"]
    if op == "ping":
        return {"mounted": supervisor.runtime.usb_connected}
    if op == "key":
        keyboard.send(*[keycode(k) for k in cmd["keys"]])
    elif op == "type":
        if len(cmd["text"]) > 32:
            raise ValueError("type text is over 32 characters")
        layout.write(cmd["text"])
    elif op == "release_all":
        keyboard.release_all()
    else:
        raise ValueError("unsupported op " + repr(op))
    return {}


while True:
    line = port.readline(512)
    if not line.strip():
        continue
    reply = {"ok": True}
    try:
        cmd = json.loads(line.decode())
        reply["id"] = cmd["id"]
        reply.update(run(cmd))
    except Exception as e:  # a bad command gets an error reply, and nothing is left held down
        keyboard.release_all()
        reply = {"id": reply.get("id"), "ok": False, "err": repr(e)}
    port.write(json.dumps(reply).encode() + b"\n")
