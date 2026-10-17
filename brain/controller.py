"""Deterministic controller: turns actions into HID commands, waits for the
screen to settle instead of sleeping, and honors cancellation everywhere."""

from __future__ import annotations

import time
import unicodedata
from dataclasses import dataclass

from brain.cancel import Cancelled, CancelToken
from brain.capture import Capture, frame_change, is_blank, thumbnail
from brain.hid import HID, HID_MAX, HIDError
from brain.keys import canonical_combo
from brain.types import Action, ActionError, DeviceProfile
from brain.verification import highlight_rows

TYPE_CHUNK = 32  # chars per UART "type" command, so cancel lands quickly
WAIT_S = 10.0  # longest block of the wait skill
BOOT_KEY_TTL_S = 120.0  # armed firmware keys expire when no restart follows
BOOT_TAP_S = 0.2  # firmware polls for its setup key; tap faster than that
BOOT_TAP_AFTER_S = 5.0  # the boot logo shown once video returns is still the key window
FIRMWARE_MOVES = 30  # arrow presses one firmware click may take

_ASCII_FIXES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "..."})


@dataclass(frozen=True)
class SettleConfig:
    min_s: float = 0.12  # give the target time to react before judging
    stable_s: float = 0.12  # screen must stay still this long
    timeout_s: float = 2.5  # stop waiting on spinners and animations
    poll_s: float = 0.03
    threshold: float = 1.5  # mean thumbnail change (0-255) that counts as "still"


class Controller:
    def __init__(
        self,
        hid: HID,
        capture: Capture,
        cancel: CancelToken,
        settle: SettleConfig = SettleConfig(),
    ) -> None:
        self.hid = hid
        self.capture = capture
        self.cancel = cancel
        self.settle = settle
        self._boot_keys: list[str] = []
        self._boot_until = 0.0

    # -- actions -----------------------------------------------------------

    def execute(self, action: Action, profile: DeviceProfile) -> None:
        """Run one non-terminal, non-memory action."""
        self.cancel.check()
        kind = action.type
        if profile.os == "bios" and kind in ("right_click", "swipe", "scroll"):
            raise ActionError("firmware screens are keyboard only: move with arrow keys, select with ENTER")
        touch = profile.input == "touch"
        if kind in ("click", "tap", "double_click", "right_click"):
            x, y = _hid_xy(action.x, action.y)
            if profile.os == "bios":
                # Firmware rarely drives an absolute pointer such as the Pico's: move its highlight with keys.
                self.select_row(y / HID_MAX, profile)
                if kind == "double_click":
                    self.hotkey(["ENTER"])
            elif touch:
                for _ in range(2 if kind == "double_click" else 1):
                    self._send({"op": "tap", "x": x, "y": y})
            else:
                self._send({
                    "op": "click", "x": x, "y": y,
                    "button": "right" if kind == "right_click" else "left",
                    "count": 2 if kind == "double_click" else 1,
                })
        elif kind == "swipe":
            x1, y1 = _hid_xy(action.x, action.y)
            x2, y2 = _hid_xy(action.x2, action.y2)
            self._send({"op": "swipe", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "ms": action.ms})
        elif kind == "scroll":
            self._send({"op": "scroll", "dy": action.dy})
        elif kind == "type_text":
            self.type_text(action.text)
        elif kind == "hotkey":
            self.hotkey(action.keys)
        elif kind == "skill":
            from brain.skills import run_skill  # skills build on this controller

            run_skill(action.name, action.args, profile, self)
        else:
            raise ActionError(f"controller cannot execute {kind!r}")

    def hotkey(self, keys: tuple[str, ...] | list[str]) -> None:
        try:
            combo = canonical_combo(keys)
        except ValueError as exc:
            raise ActionError(str(exc)) from exc
        self._send({"op": "key", "keys": combo})

    def hover(self, x: float, y: float) -> None:
        """Move without pressing a button; reuse the existing UART move command."""
        px, py = _hid_xy(x, y)
        self._send({"op": "move", "x": px, "y": py})

    def type_text(self, text: str) -> None:
        clean = _ascii(text)
        for start in range(0, len(clean), TYPE_CHUNK):
            self.cancel.check()
            self._send({"op": "type", "text": clean[start : start + TYPE_CHUNK]})

    def select_row(self, y: float, profile: DeviceProfile) -> None:
        """Press DOWN/UP until a firmware menu's highlight bar covers screen row y (0-1),
        or the selectable row nearest it."""
        frame = self.capture.grab()
        if frame is None:
            raise ActionError("no video")
        target = profile.rect.y0 + y * profile.rect.crop(frame).shape[0]
        key, last = "DOWN", None
        for _ in range(FIRMWARE_MOVES):
            self.hotkey([key])
            self.wait_settled()
            after = self.capture.grab()
            band = None if after is None else highlight_rows(frame, after)
            if band is None:
                if last is None:
                    raise ActionError(f"pressed {key} but no menu highlight moved; use the arrow keys")
                return  # the end of a menu that does not wrap round: the row nearest the target
            frame = after
            if band[0] <= target <= band[1]:
                return
            wanted = "DOWN" if target > band[1] else "UP"
            if last is not None:
                wrapped = (band[0] < last[0]) == (key == "DOWN")
                if wrapped or wanted != key:  # went past the target: keep the nearer of the last two rows
                    if wrapped or _gap(last, target) < _gap(band, target):
                        self.hotkey(["UP" if key == "DOWN" else "DOWN"])
                        self.wait_settled()
                    return
            key, last = wanted, band
        raise ActionError("could not reach that row; use the arrow keys")

    def arm_boot_keys(self, keys: list[str]) -> None:
        """Tap these firmware keys (e.g. F2) through the next restart; see `await_target`."""
        self._boot_keys, self._boot_until = keys, time.monotonic() + BOOT_KEY_TTL_S

    def await_target(self, timeout_s: float) -> bool:
        """After a restart, replug or blank screen: wait (while cancelled) until the target
        is mounted with live video, tapping armed firmware keys. False on timeout."""
        armed = time.monotonic() < self._boot_until
        self._boot_until = 0.0
        hold = BOOT_TAP_AFTER_S if armed else BOOT_TAP_S  # >= one monitor poll, so it has seen the return
        deadline = time.monotonic() + timeout_s
        live_since: float | None = None
        while (now := time.monotonic()) < deadline:
            mounted = self.hid.is_mounted()
            frame = self.capture.grab() if mounted else None
            live = frame is not None and not is_blank(frame)
            live_since = (now if live_since is None else live_since) if live else None
            if live_since is not None and now - live_since >= hold:
                return True
            if armed and mounted:
                for key in self._boot_keys:
                    try:
                        self.hid.send({"op": "key", "keys": [key]})
                    except HIDError:
                        pass  # the firmware's USB stack is not up yet
            time.sleep(BOOT_TAP_S)
        return False

    def release_all(self) -> None:
        """Best effort: never leave a key or button held down."""
        if self.hid.is_mounted():
            try:
                self.hid.send({"op": "release_all"})
            except HIDError:
                pass

    # -- settling ----------------------------------------------------------

    def wait_settled(self) -> float:
        """Block until the screen stops changing. Returns seconds waited."""
        cfg = self.settle
        start = time.monotonic()
        prev = None
        still_since: float | None = None
        while True:
            self.cancel.check()
            now = time.monotonic()
            elapsed = now - start
            frame = self.capture.grab()
            if frame is not None:
                thumb = thumbnail(frame)
                if prev is not None and frame_change(prev, thumb) < cfg.threshold:
                    still_since = still_since if still_since is not None else now
                    if elapsed >= cfg.min_s and now - still_since >= cfg.stable_s:
                        return elapsed
                else:
                    still_since = None
                prev = thumb
            if elapsed >= cfg.timeout_s:
                return elapsed
            self.cancel.sleep(cfg.poll_s)

    def wait_for_change(self) -> None:
        """Block until the screen changes (a restart, boot or load progressing) or WAIT_S passes."""
        frame = self.capture.grab()
        if frame is None:
            return
        before = thumbnail(frame)
        deadline = time.monotonic() + WAIT_S
        while time.monotonic() < deadline:
            self.cancel.sleep(self.settle.poll_s)
            frame = self.capture.grab()
            if frame is None or frame_change(before, thumbnail(frame)) >= self.settle.threshold:
                return

    def _send(self, cmd: dict[str, object]) -> None:
        self.cancel.check()
        try:
            self.hid.send(cmd)
        except HIDError:
            if self.cancel.cancelled:
                raise Cancelled("target disconnected") from None
            raise


def _hid_xy(x: float | None, y: float | None) -> tuple[int, int]:
    if x is None or y is None:
        raise ActionError("pointer action needs x and y")
    return round(x * HID_MAX), round(y * HID_MAX)


def _gap(band: tuple[int, int], target: float) -> float:
    return max(band[0] - target, target - band[1], 0)


def _ascii(text: str) -> str:
    """The Pico types US-ASCII only; fold smart quotes and accents."""
    folded = unicodedata.normalize("NFKD", text.translate(_ASCII_FIXES))
    return folded.encode("ascii", "ignore").decode()
