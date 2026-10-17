# Hardware bring-up: connecting everything physically

**Status:** guide. Nothing here changes the contracts in PLAN.md §5; it describes how to build what §5.1 already
specifies. `firmware/pico/` has no firmware yet, while `brain/hid.py` (Pi side) is already written against the
protocol, so the Pico firmware is the missing piece.

## 1. The physical setup
```
            ┌──────── USB-C hub (HDMI out + USB-A data + power pass-through) ────────┐
 TARGET ════╡ one USB-C cable                                                         │
 laptop or  │   HDMI out ──────────► capture card ──USB──► Pi 5 (USB 3, blue port)    │
 phone      │   USB-A ─── micro-USB cable ───► Pico 2 W  (shows up as kbd + mouse)   │
            └──────────────────────────────────────────────────────────────────────┘
 Pi 5 GPIO ──UART──► Pico          Pi 5 DSI ──ribbon──► Touch Display 2
 Pi 5 USB ──► mic / speaker        Pi 5 GPIO17 ──► push-to-talk button (+ 3.5 mm switch jack)
 Pi 5 power: official 27 W (5 V / 5 A) supply, its own wall plug
```
- **The hub's host is the target.** Anything in the hub's USB-A port (the Pico) belongs to the target; the capture
  card belongs to the Pi.
- **The target must output video over USB-C:** Macs, most recent laptops, Samsung phones (DeX). Set laptops to
  **mirror**, not extend; turn off DeX auto-start on the phone (PLAN.md §10).
- **The hub needs real USB data on its USB-A ports.** Some cheap "HDMI adapters" only pass power. Check first.

## 2. Wiring Pi ↔ Pico (PLAN.md §5.1)
| Pi 5 pin | → | Pico pin | Purpose |
|---|---|---|---|
| 8 (GPIO14, TX) | → | 2 (GP1, RX) | Pi sends commands |
| 10 (GPIO15, RX) | ← | 1 (GP0, TX) | Pico replies |
| 6 (GND) | — | 3 (GND) | **Shared ground is required** |
| 2 (5 V) | → **1N5817 diode** → | 39 (VSYS) | Pi powers the Pico so it survives target swaps; the diode stops the target's USB power from flowing back |
| 11 (GPIO17) | — button — | 9 (GND), on the Pi | Push-to-talk; wire the 3.5 mm switch jack in parallel |

Both boards use 3.3 V logic, so no level shifter. **Never connect the Pi's 5 V to a Pico GPIO pin.**

## 3. Pi 5 setup (about 10 minutes)
1. **Turn on the header UART.** `ttyAMA10` is the Pi 5's separate debug port; don't use it.
   - `sudo raspi-config` → Interface → Serial Port → login shell **No**, hardware **Yes**
   - Add `dtparam=uart0=on` to `/boot/firmware/config.txt`, reboot
   - Check: `ls /dev/ttyAMA0`
2. **Test the UART alone:** jumper pin 8 to pin 10, then
   `python3 -c "import serial;s=serial.Serial('/dev/ttyAMA0',115200,timeout=1);s.write(b'hi\n');print(s.readline())"`
   should print `b'hi\n'`.
3. **Capture card:** `v4l2-ctl --list-devices`, then `v4l2-ctl -d /dev/videoN --list-formats-ext`. Look for MJPG at
   1920×1080. `/dev/video19+` are the Pi's internal devices, not a capture card. Put the real path in
   `CAPTURE_DEVICE` in `.env`.

## 4. Pico firmware (missing)
CircuitPython is fine: the Pico's share of a step is under 5 ms, while Jev alone is about 200 ms. Two files plus the
`adafruit_hid` library in `lib/`:

**`boot.py`** (runs before USB starts)
- Hide the CIRCUITPY drive and serial console from the target: `storage.disable_usb_drive()`, `usb_cdc.disable()`.
  Otherwise every target gets a "new drive" popup and macOS opens its keyboard setup wizard.
- **Dev-mode escape:** skip the hiding when a jumper pulls GP22 to ground, so the team can still edit code over USB.
- Present three USB devices: a boot keyboard, an **absolute mouse** (0–32767, so clicks land exactly with no drift),
  and a **single-touch digitizer** for Android, via `usb_hid.enable(...)` with custom report descriptors.

**`code.py`** (main loop)
- Read newline-delimited JSON from the Pi over UART (GP0/GP1, 115200).
- Do each command and reply with the same `id`: `ping`, `key`, `type`, `move`, `click`, `scroll`, `tap`, `swipe`,
  `release_all` (matching `brain/controller.py`). Reply `{"id":N,"ok":true}` or `{"id":N,"ok":false,"err":"..."}`.
- **The `ping` reply must include `"mounted": true/false`**; `brain/hid.py` reads it at startup.
- Watch `supervisor.runtime.usb_connected` and send `{"ev":"mounted"}` / `{"ev":"unmounted"}` when it changes. On
  unmount, release every key and button itself; the Pi uses the event to cancel the current task.

## 5. Bring-up order (each step isolates one thing)
1. **Pico alone on a laptop** (dev mode): a test that types "hello" every 5 s proves the USB keyboard works.
2. **Pi ↔ Pico UART:** send `{"op":"ping","id":1}`, expect `{"id":1,"ok":true,"mounted":true}`. Then
   `key ["CMD","SPACE"]` and `move`/`click` on a laptop.
3. **Capture:** grab one frame in Python and save it as a JPEG.
4. **Hub test:** one USB-C cable to laptop A gives video on the Pi *and* Pico input at the same time. Repeat on
   laptop B and the Samsung (PLAN.md §10).
5. **Full system:** `python -m brain --backend pi --display 8770`, with Chromium in kiosk mode on the touch display.

## 6. Shopping list
- USB-C hub with HDMI, a USB-A **data** port and power pass-through
- Micro-USB to USB-A **data** cable for the Pico (many are charge-only)
- 1N5817 (or any Schottky) diode
- Official 27 W Pi 5 power supply: on weaker supplies the Pi limits USB current and the capture card or mic can drop
- Jumper wires
