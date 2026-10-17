# Hardware bring-up

How to build and check what PLAN.md §5.1 specifies. Firmware details: [firmware/pico/README.md](../firmware/pico/README.md).

## 1. The physical setup
```
            ┌──────── USB-C hub (HDMI out + USB-A data + power pass-through) ────────┐
 TARGET ════╡ one USB-C cable                                                         │
 laptop or  │   HDMI out ──────────► capture card ──USB──► Pi 5 (USB 3, blue port)    │
 phone      │   USB-A ─── micro-USB cable ───► Pico 2  (shows up as kbd + mouse)     │
            └──────────────────────────────────────────────────────────────────────┘
 Pi 5 GPIO ──UART──► Pico          Pi 5 DSI ──ribbon──► Touch Display 2
 Pi 5 USB ──► mic / speaker        Pi 5 power: official 27 W (5 V / 5 A) supply, its own wall plug
```
- **The hub's host is the target.** Anything in the hub's USB-A port (the Pico) belongs to the target; the capture
  card belongs to the Pi.
- **The target must output video over USB-C:** Macs, most recent laptops, Samsung phones. Set laptops to
  **mirror**, not extend; turn off DeX auto-start on the phone so it mirrors.
- **The hub needs real USB data on its USB-A ports.** Some cheap "HDMI adapters" only pass power. Check first.

## 2. Wiring Pi ↔ Pico
| Pi 5 pin | → | Pico pin | Purpose |
|---|---|---|---|
| 8 (GPIO14, TX) | → | 2 (GP1, RX) | Pi sends commands |
| 10 (GPIO15, RX) | ← | 1 (GP0, TX) | Pico replies |
| 6 (GND) | — | 3 (GND) | **Shared ground is required** |
| 2 (5 V) | → **1N5817 diode** → | 39 (VSYS) | Pi powers the Pico so it survives target swaps; the diode stops the target's USB power from flowing back |
| — | | 29 (GP22) → GND | Demo-mode jumper (HID only); leave open to edit files |

Both boards use 3.3 V logic, so no level shifter. **Never connect the Pi's 5 V to a Pico GPIO pin.**
A push-to-talk button on GPIO17 (with a 3.5 mm switch jack in parallel) is planned but not read by software
yet; the voice page's Talk button and the Space bar work today, so a USB switch interface that sends Space works.

## 3. Pi setup
From the laptop, `scripts/pi/deploy.sh` copies the code (never `.env`) and runs `scripts/pi/setup.sh` on the Pi:
venv, `pip install -e '.[voice]'`, user services, offline tests and `brain.doctor`. It then prints the one-time
sudo steps:
1. **Header UART:** serial console off, UART hardware on, `dtparam=uart0=on` in `/boot/firmware/config.txt`.
   Use `/dev/ttyAMA0`; `ttyAMA10` is the Pi 5's separate debug port.
2. `sudo usermod -aG video,dialout "$USER"` (capture card and UART).
3. `sudo loginctl enable-linger "$USER"` (services start at boot without a login).

Reboot, copy `.env` (chmod 600), then `systemctl --user enable --now exospace-agent exospace-voice`.

Manual checks if `brain.doctor` complains:
- **UART alone:** jumper pin 8 to pin 10, then
  `python3 -c "import serial;s=serial.Serial('/dev/ttyAMA0',115200,timeout=1);s.write(b'hi\n');print(s.readline())"`
  prints `b'hi\n'`.
- **Capture card:** `v4l2-ctl --list-devices`, then `v4l2-ctl -d /dev/videoN --list-formats-ext`; look for MJPG at
  1920×1080. `/dev/video19+` are the Pi's internal devices. Put the real path in `CAPTURE_DEVICE`.

## 4. Bring-up order (each step isolates one thing)
1. **Pico alone on a laptop** (maintenance mode): copy the firmware, replug, and check the drive and console appear.
2. **Pi ↔ Pico UART:** `brain.doctor` pings the Pico; the reply carries `mounted`. Then try a `key` and a `click`
   on a laptop.
3. **Capture:** `brain.doctor` grabs one frame.
4. **Hub test:** one USB-C cable to each target gives video on the Pi *and* Pico input at the same time.
5. **Full system:** the two services, with Chromium in kiosk mode on the touch display (`:8770` and `:8780`).

## 5. Checklist before the demo
- [ ] Same hub gives video **and** Pico input on laptop A, laptop B and the phone
- [ ] Phone: video out works, DeX auto-start off (mirroring), "Show taps" on
- [ ] Phone: taps near each corner land where asked (firmware README, "First check on the phone")
- [ ] Laptops set to **mirror**, not extend
- [ ] Pico pre-plugged into every target once in demo mode (Windows driver install, macOS keyboard setup popup)
- [ ] Hot-swap: the Pi stays up; time from plug-in to ready measured
- [ ] Round-trip latency on venue Wi-Fi measured; phone hotspot ready as backup

## 6. Shopping list
- USB-C hub with HDMI, a USB-A **data** port and power pass-through
- Micro-USB to USB-A **data** cable for the Pico (many are charge-only)
- 1N5817 (or any Schottky) diode
- Official 27 W Pi 5 power supply: on weaker supplies the Pi limits USB current and the capture card or mic can drop
- Jumper wires
