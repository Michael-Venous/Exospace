"""Senders for the three HID devices declared in descriptors.py."""

import time

import supervisor
import usb_hid
from adafruit_hid.keyboard import Keyboard as HidKeyboard
from adafruit_hid.keyboard_layout_us import KeyboardLayoutUS
from adafruit_hid.keycode import Keycode

import keys
from descriptors import MOUSE_REPORT_ID, TOUCH_REPORT_ID
from reports import mouse_report, touch_report

CLICK_HOLD_S = 0.04  # over libinput's 25 ms debounce: a shorter release is held back and lands after the next move (a drag)
TAP_HOLD_S = 0.05
SWIPE_STEP_MS = 20


def _find(usage_page: int, usage: int):
    for device in usb_hid.devices:
        if device.usage_page == usage_page and device.usage == usage:
            return device
    raise RuntimeError("HID device missing; boot.py did not enable it")


class Keyboard:
    """Created lazily: adafruit_hid's Keyboard waits for a USB host, and the UART must
    answer the Pi before any target is plugged in."""

    def __init__(self):
        self._keyboard = None
        self._layout = None

    def _ready(self):
        if not supervisor.runtime.usb_connected:
            raise OSError("no USB host")
        if self._keyboard is None:
            self._keyboard = HidKeyboard(usb_hid.devices)
            self._layout = KeyboardLayoutUS(self._keyboard)
        return self._keyboard

    def press_keys(self, names: list) -> None:
        codes = [keys.resolve(name, Keycode) for name in names]
        keyboard = self._ready()
        try:
            keyboard.press(*codes)
        finally:
            keyboard.release_all()

    def type_text(self, text: str) -> None:
        self._ready()
        self._layout.write(text)

    def release_all(self) -> None:
        if self._keyboard is not None and supervisor.runtime.usb_connected:
            self._keyboard.release_all()


class AbsMouse:
    def __init__(self):
        self._device = _find(0x01, 0x02)
        self._x = 0
        self._y = 0

    def move(self, x: int, y: int) -> None:
        self._x, self._y = x, y
        self._send(0, 0)

    def click(self, x: int, y: int, buttons: int, count: int) -> None:
        self.move(x, y)
        for _ in range(count):
            self._send(buttons, 0)
            time.sleep(CLICK_HOLD_S)
            self._send(0, 0)
            time.sleep(CLICK_HOLD_S)

    def scroll(self, dy: int) -> None:
        """Positive dy scrolls down (as in LocalHID); the HID wheel is + = up."""
        self._send(0, -dy)

    def release_all(self) -> None:
        self._send(0, 0)

    def _send(self, buttons: int, wheel: int) -> None:
        self._device.send_report(mouse_report(buttons, self._x, self._y, wheel), MOUSE_REPORT_ID)


class Touch:
    def __init__(self):
        self._device = _find(0x0D, 0x04)
        self._x = 0
        self._y = 0

    def tap(self, x: int, y: int) -> None:
        self._send(True, x, y)
        time.sleep(TAP_HOLD_S)
        self._send(False, x, y)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, ms: int, should_abort) -> bool:
        """Drag a finger from (x1, y1) to (x2, y2). Returns False if aborted."""
        steps = max(1, ms // SWIPE_STEP_MS)
        self._send(True, x1, y1)
        for step in range(1, steps + 1):
            if should_abort():
                self._send(False, self._x, self._y)
                return False
            time.sleep(ms / steps / 1000)
            self._send(True, x1 + (x2 - x1) * step // steps, y1 + (y2 - y1) * step // steps)
        self._send(False, x2, y2)
        return True

    def release_all(self) -> None:
        self._send(False, self._x, self._y)

    def _send(self, touching: bool, x: int, y: int) -> None:
        self._x, self._y = x, y
        self._device.send_report(touch_report(touching, x, y), TOUCH_REPORT_ID)
