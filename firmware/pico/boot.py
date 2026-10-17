"""Runs once at power-up, before USB starts (boot.py only runs on a hard reset or replug).

Default (GP22 not connected): maintenance mode. The CIRCUITPY drive and serial console
stay visible so files can be edited, a second USB serial port takes the same commands as the
UART (so the computer it is plugged into can drive itself for testing), and HID uses report
protocol. With the drive and
console enabled, CircuitPython refuses a boot-protocol keyboard ("Boot device must be
first"), so maintenance mode never asks for one.

GP22 wired to GND at power-up: demo mode. HID only, so the target sees no drive or
serial-port popups, and the keyboard is boot-protocol for BIOS/UEFI menus. To edit files
again, remove the jumper and replug. Safe mode (if anything here fails) skips boot.py and
shows the drive, so a mistake can always be fixed.
"""

import board
import digitalio
import storage
import usb_cdc
import usb_hid

from descriptors import KEYBOARD, MOUSE, TOUCH

_jumper = digitalio.DigitalInOut(board.GP22)
_jumper.switch_to_input(pull=digitalio.Pull.UP)
DEMO_MODE = not _jumper.value  # GP22 grounded
_jumper.deinit()

if DEMO_MODE:
    storage.disable_usb_drive()
    usb_cdc.disable()
    try:
        import usb_midi
        usb_midi.disable()
    except ImportError:
        pass
    usb_hid.enable((KEYBOARD, MOUSE, TOUCH), boot_device=1)
else:
    usb_cdc.enable(console=True, data=True)  # data: the computer it is plugged into can send commands
    usb_hid.enable((KEYBOARD, MOUSE, TOUCH))
