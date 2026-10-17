"""Hardware and setup check: `python -m brain.doctor`.

Run it on the Pi when the capture card and Pico are plugged in. Each line says
what is wrong and what to do. It never prints key values. Exit code 1 if any
check fails.
"""

from __future__ import annotations

import grp
import os
import time
from dataclasses import dataclass
from pathlib import Path

from brain.config import Config

REQUIRED_KEYS = ("GEMINI_API_KEY", "TYPESAFE_API_KEY")
OPTIONAL_KEYS = ("DEEPGRAM_API_KEY",)  # voice layer


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def check_keys(env: dict[str, str]) -> list[Check]:
    checks = [Check(key, bool(env.get(key)), "set" if env.get(key) else "missing: add it to .env")
              for key in REQUIRED_KEYS]
    checks += [Check(key, bool(env.get(key)), "set" if env.get(key) else
                     "missing: the voice service (exospace-voice) won't start without it")
               for key in OPTIONAL_KEYS]
    return checks


def check_groups(user_groups: set[str]) -> list[Check]:
    return [Check(f"group {g}", g in user_groups,
                  "member" if g in user_groups else f"run: sudo usermod -aG {g} $USER, then log in again")
            for g in ("video", "dialout")]


def check_uart(port: str) -> Check:
    if Path(port).exists():
        if os.access(port, os.R_OK | os.W_OK):
            return Check("UART", True, f"{port} is readable and writable")
        return Check("UART", False, f"{port} exists but isn't accessible: join the dialout group")
    hint = " (only the Pi 5 debug UART /dev/ttyAMA10 exists: enable uart0, see scripts/pi/setup.sh)" \
        if Path("/dev/ttyAMA10").exists() else ""
    return Check("UART", False, f"{port} not found{hint}")


def check_pico(port: str, baud: int) -> Check:
    from brain.hid import HIDError, PicoHID

    try:
        pico = PicoHID(port, baud)
    except (HIDError, OSError) as exc:
        return Check("Pico", False, f"no reply ({exc}); check wiring GP0/GP1/GND and the firmware")
    try:
        mounted = pico.is_mounted()
    finally:
        pico.close()
    return Check("Pico", True, "answers; USB host " + ("connected" if mounted else "not connected yet"))


def check_capture(device: str, width: int, height: int) -> Check:
    if not Path(device).exists():
        return Check("capture card", False, f"{device} not found: plug in the HDMI capture card")
    from brain.capture import CaptureError, V4L2Capture

    try:
        capture = V4L2Capture(device, width, height)
    except CaptureError as exc:
        return Check("capture card", False, f"{exc} (in use, or not a capture device?)")
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and capture.grab() is None:
            time.sleep(0.05)
        frame = capture.grab()
        if frame is None:
            return Check("capture card", False, "opened, but no frames arrived")
        h, w = frame.shape[:2]
        signal = "showing video" if capture.signal_present() else "blank (no HDMI signal yet)"
        return Check("capture card", True, f"{w}x{h}, {signal}")
    finally:
        capture.close()


def run() -> list[Check]:
    cfg = Config.load()
    checks = check_keys(dict(os.environ))
    checks += check_groups({grp.getgrgid(g).gr_name for g in os.getgroups()})
    uart = check_uart(cfg.pico_serial)
    checks += [uart]
    if uart.ok:
        checks.append(check_pico(cfg.pico_serial, cfg.pico_baud))
    checks.append(check_capture(cfg.capture_device, cfg.capture_width, cfg.capture_height))
    return checks


def main() -> int:
    checks = run()
    for c in checks:
        print(f"{'ok ' if c.ok else 'FIX'}  {c.name:18} {c.detail}")
    failed = [c for c in checks if not c.ok]
    print("\nReady." if not failed else f"\n{len(failed)} to fix.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
