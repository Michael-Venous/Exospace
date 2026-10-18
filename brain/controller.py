"""Deterministic controller: turns actions into HID commands, waits for the
screen to settle instead of sleeping, and honors cancellation everywhere."""

from __future__ import annotations

import time
import unicodedata
from dataclasses import dataclass

import numpy as np

from brain.cancel import Cancelled, CancelToken
from brain.capture import Capture, frame_change, is_blank, thumbnail
from brain.hid import HID, HID_MAX, HIDError
from brain.keys import canonical_combo
from brain.types import Action, ActionError, DeviceProfile

TYPE_CHUNK = 32  # chars per UART "type" command, so cancel lands quickly
WAIT_S = 10.0  # longest block of the wait skill
MAX_SCROLL = 15  # wheel notches per action; a pixel count (e.g. 800) would fling to the page end
RETURN_HOLD_S = 0.2  # live video this long counts as back: at least one monitor poll, so it has seen it

_ASCII_FIXES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "..."})


@dataclass(frozen=True)
class SettleConfig:
    min_s: float = 0.12  # give the target time to react before judging
    # With no change yet, wait this long for one: through the Pi's capture card the effect of an input
    # appeared after the screen had already looked settled, and checks on the stale screen failed and
    # replanned (3-6 s each, 4 of 4 tasks).
    react_s: float = 0.45
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

    # -- actions -----------------------------------------------------------

    def execute(self, action: Action, profile: DeviceProfile) -> None:
        """Run one non-terminal, non-memory action."""
        self.cancel.check()
        kind = action.type
        touch = profile.input == "touch"
        if kind in ("click", "tap", "double_click", "right_click"):
            x, y = _hid_xy(action.x, action.y)
            if touch:
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
            self._send({"op": "scroll", "dy": max(-MAX_SCROLL, min(MAX_SCROLL, action.dy))})
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

    def await_target(self, timeout_s: float) -> bool:
        """After a restart, replug or blank screen: wait (while cancelled) until the target
        is mounted with live video. False on timeout."""
        deadline = time.monotonic() + timeout_s
        live_since: float | None = None
        while (now := time.monotonic()) < deadline:
            frame = self.capture.grab() if self.hid.is_mounted() else None
            live = frame is not None and not is_blank(frame)
            live_since = (now if live_since is None else live_since) if live else None
            if live_since is not None and now - live_since >= RETURN_HOLD_S:
                return True
            time.sleep(RETURN_HOLD_S)
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
        changed = False
        while True:
            self.cancel.check()
            now = time.monotonic()
            elapsed = now - start
            frame = self.capture.grab()
            if frame is not None:
                thumb = thumbnail(frame)
                if prev is not None and frame_change(prev, thumb) < cfg.threshold:
                    still_since = still_since if still_since is not None else now
                    if elapsed >= (cfg.min_s if changed else max(cfg.min_s, cfg.react_s)) \
                            and now - still_since >= cfg.stable_s:
                        return elapsed
                else:
                    changed = changed or prev is not None
                    still_since = None
                prev = thumb
            if elapsed >= cfg.timeout_s:
                return elapsed
            self.cancel.sleep(cfg.poll_s)

    def wait_for_change(self, timeout_s: float | None = None, since: np.ndarray | None = None) -> None:
        """Block until the screen differs from `since` (a thumbnail; default: now), as a load,
        restart or app start progresses, or `timeout_s` (default WAIT_S) passes."""
        if since is None:
            frame = self.capture.grab()
            if frame is None:
                return
            since = thumbnail(frame)
        before = since
        deadline = time.monotonic() + (WAIT_S if timeout_s is None else timeout_s)
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


def _ascii(text: str) -> str:
    """The Pico types US-ASCII only; fold smart quotes and accents."""
    folded = unicodedata.normalize("NFKD", text.translate(_ASCII_FIXES))
    return folded.encode("ascii", "ignore").decode()
