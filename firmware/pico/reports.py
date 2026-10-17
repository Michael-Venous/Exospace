"""HID input report layouts. Pure byte packing, shared with the descriptors."""

import struct

HID_MAX = 32767

BUTTON_LEFT = 0x01
BUTTON_RIGHT = 0x02

TOUCH_TIP = 0x01
TOUCH_IN_RANGE = 0x02


def mouse_report(buttons: int, x: int, y: int, wheel: int) -> bytes:
    """Absolute mouse: buttons, X, Y (0-32767) and a relative wheel (+ = up)."""
    return struct.pack("<BHHb", buttons, x, y, max(-127, min(127, wheel)))


def touch_report(touching: bool, x: int, y: int) -> bytes:
    """One-contact multitouch report: flags, contact id 0, X, Y, contact count 1.
    The contact id makes Linux/Android bind hid-multitouch, which marks the device as a
    direct touchscreen; without it Android treats it as a touchpad with a cursor."""
    flags = (TOUCH_TIP | TOUCH_IN_RANGE) if touching else 0
    return struct.pack("<BBHHB", flags, 0, x, y, 1)
