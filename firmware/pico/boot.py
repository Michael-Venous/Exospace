"""Runs once at power-up, before USB starts. The CIRCUITPY drive is left
visible on purpose, so a mistake here can always be fixed by editing the drive."""

import usb_hid

from descriptors import KEYBOARD, MOUSE, TOUCH

# The keyboard comes first so hosts treat it as the boot-protocol keyboard.
usb_hid.enable((KEYBOARD, MOUSE, TOUCH), boot_device=1)
