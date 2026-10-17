# Pico firmware ("hands")

**Owner:** Hardware

CircuitPython firmware that presents as a composite HID device: a boot-protocol keyboard, an absolute mouse (0–32767) and a single-touch digitizer. It runs the newline-delimited JSON UART protocol in [PLAN.md §5.1](../../PLAN.md).

The firmware holds no logic beyond the protocol. On USB `unmounted`, it must `release_all`.

Wiring, Pi setup and the bring-up order: [docs/hardware-bringup.md](../../docs/hardware-bringup.md).

## Files

| File | Role |
|---|---|
| `boot.py` | Enables keyboard, mouse and touch at power-up; leaves the CIRCUITPY drive visible |
| `descriptors.py` | HID report descriptors and `usb_hid.Device` objects |
| `reports.py` | Byte layout of the mouse and touch reports (pure, tested) |
| `keys.py` | Protocol key names -> `adafruit_hid` Keycode names (pure, tested) |
| `commands.py` | Validates and runs each protocol op (pure, tested) |
| `devices.py` | Sends keyboard, mouse and touch reports to the host |
| `link.py` | UART GP0 (TX) / GP1 (RX), 115200, JSON lines |
| `code.py` | Main loop and `mounted`/`unmounted` events |

Copy all of these onto the CIRCUITPY drive, then replug (`boot.py` only runs on a hard reset). `tests/test_firmware.py` checks the pure modules on a PC; the rest is verified on the board.

## Board setup

- Board: Raspberry Pi Pico 2 (RP2350), CircuitPython 10.3.1 (`raspberry_pi_pico2`).
- The `CIRCUITPY` drive has `lib/adafruit_hid/` preinstalled. It is not tracked; reinstall it from the Adafruit CircuitPython bundle matching the firmware's major version (10.x) if the drive is wiped.
- `settings.toml` is empty and untracked, because it can hold secrets.

## Workflow

The repo is the source of truth. Never edit files on the Pico directly.

1. Edit `boot.py` and `code.py` in this folder.
2. Copy them onto the `CIRCUITPY` drive.
3. Eject the drive in Windows before unplugging, since the Pico's flash can corrupt if it is unplugged mid-write.

`boot.py` runs only on a hard reset or replug, not when `code.py` is saved. Do not make `boot.py` hide the drive or make it read-only without a recovery path.

## Sharing the Pico

Whoever holds the Pico copies their own latest files onto it first. To recover a Pico that no longer mounts, hold BOOTSEL while plugging it in and reinstall CircuitPython.
