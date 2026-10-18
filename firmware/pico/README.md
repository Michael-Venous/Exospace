# Pico firmware ("hands")

**Owner:** Hardware

CircuitPython firmware that presents as a composite HID device: a keyboard, an absolute mouse (0–32767) and a
one-contact multitouch digitizer. It runs the newline-delimited JSON UART protocol in
[PLAN.md §5.1](../../PLAN.md) and holds no logic beyond it. Wiring and the Pi side:
[docs/hardware-bringup.md](../../docs/hardware-bringup.md).

## Files

| File | Role |
|---|---|
| `boot.py` | Picks maintenance or demo mode from GP22 and enables keyboard, mouse and touch |
| `descriptors.py` | HID report descriptors and `usb_hid.Device` objects |
| `reports.py` | Byte layout of the mouse and touch reports (pure, tested) |
| `keys.py` | Protocol key names → `adafruit_hid` Keycode names (pure, tested) |
| `commands.py` | Validates and runs each protocol op (pure, tested) |
| `devices.py` | Sends keyboard, mouse and touch reports; the keyboard is created once a host is attached |
| `link.py` | UART GP0 (TX) / GP1 (RX), 115200, JSON lines |
| `code.py` | Main loop; reports `mounted`/`unmounted` the moment USB changes |

`tests/test_firmware.py` checks the pure modules on a PC, and `tests/test_pico_loopback.py` runs the real Pi
client against `commands.py` over a pseudo-terminal. The rest is verified on the board.

## Modes (boot.py)

- **Maintenance (default, nothing on GP22):** the CIRCUITPY drive and serial console stay visible so you can
  copy files; HID uses report protocol, which every OS and modern UEFI accepts.
- **Demo (GP22 wired to GND at power-up):** HID only, so the target shows no drive or serial-port popups, and
  the keyboard is boot-protocol for older BIOS menus. Remove the jumper and power-cycle to edit files again.

## Flashing

The repo is the source of truth; never edit files on the Pico directly.
1. In maintenance mode, copy **all eight** `.py` files above onto the `CIRCUITPY` drive.
2. Eject the drive before unplugging (the flash can corrupt mid-write).
3. Power-cycle: `boot.py` only runs on a hard reset. The Pi powers the Pico, so unplugging the target does not
   reset it; power-cycle the Pi or ground the Pico's RUN pin.

The UART answers the Pi even before any target is plugged in. Whoever holds the Pico copies their own latest
files onto it first.

**Board:** Raspberry Pi Pico 2 (RP2350), CircuitPython 10.3.1 (`raspberry_pi_pico2`). `lib/adafruit_hid/` is on
the drive but not tracked; reinstall it from the Adafruit bundle for 10.x if the drive is wiped. `settings.toml`
stays empty and untracked. To recover a Pico that no longer mounts, hold BOOTSEL while plugging it in and
reinstall CircuitPython.

## First check on the phone (30 s)

Android should treat the touch device as a real touchscreen (no cursor). With the phone mirrored, ask Exospace
to tap something near each corner. If taps land squeezed toward the middle or not at all, Android is mapping
touch to the whole HDMI frame rather than the phone screen: tell the agent owner, since `brain/controller.py`
then needs to map touch coordinates in frame space.
