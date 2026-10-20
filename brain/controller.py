"""Deterministic controller: turns actions into HID commands, waits for the
screen to settle instead of sleeping, and honors cancellation everywhere."""

from __future__ import annotations

import time
import unicodedata
from dataclasses import dataclass

import cv2
import numpy as np

from brain.cancel import Cancelled, CancelToken
from brain.capture import Capture, frame_change, is_blank, thumbnail
from brain.hid import HID, HID_MAX, HIDError
from brain.keys import canonical_combo
from brain.types import Action, ActionError, DeviceProfile

TYPE_CHUNK = 32  # chars per UART "type" command, so cancel lands quickly
WAIT_S = 10.0  # longest block of the wait skill
MAX_SCROLL = 15  # wheel notches per action; a pixel count (e.g. 800) would fling to the page end
SWIPE_PER_NOTCH = 0.06  # screen heights a finger drags per notch on touch: 5 notches, 0.65 to 0.35, scrolled a page
RETURN_HOLD_S = 0.2  # live video this long counts as back: at least one monitor poll, so it has seen it
# Where the pointer waits between clicks: left on a page it hid text and drew Jev's box to it, "copy the header" boxed
# the citation row under the I-beam for the one a footnote jump marked (3 of 6); with it erased, the marked one (9/9).
# The right edge's middle: a scrollbar in a maximized window, no hot corner, no panel.
PARK = (0.995, 0.5)
# A thumbnail cell this different between polls is moving; one moving in at least half of the last MOTION_POLLS
# polls is moving on its own (a playing video): a YouTube video behind the launcher made "the launcher showed" true
# 44 ms after its key and held each settle to its 2.5 s timeout (2026-10-19).
MOVING_DIFF = 8
MOTION_POLLS = 8
# Cells moving on their own over the last MOTION_POLLS polls of one settle are left out of it while they are at most
# this part of the screen: Brave restored a playing YouTube Short at launch and the settle held to its 2.5 s timeout,
# which delayed every later step of the request (2026-10-20). More is a page being drawn or scrolled.
MAX_MOVING = 1 / 3

_ASCII_FIXES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "..."})


@dataclass(frozen=True)
class SettleConfig:
    min_s: float = 0.12  # give the target time to react before judging
    # With no change yet, wait this long for one: through the Pi's capture card the effect of an input
    # appeared after the screen had already looked settled, and checks on the stale screen failed and
    # replanned (3-6 s each, 4 of 4 tasks).
    react_s: float = 0.45
    # After Enter or a click, which may start a page load: a load pauses between drawing steps, so a short
    # stillness passed as "settled" and the check after "search venus effects" ran on a half-loaded page
    # (two replans, ~7 s, on the Pi).
    load_stable_s: float = 0.4
    # ...and the screen may take this long to start changing: a phone app opening from its icon showed
    # nothing at 0.45 s, the step counted as "unchanged" and the request went to the planner.
    load_react_s: float = 1.0
    stable_s: float = 0.12  # screen must stay still this long
    timeout_s: float = 2.5  # stop waiting on spinners and animations
    # A run's first look, with no input of ours to show: a Short playing behind held it 1.4-1.7 s (2026-10-19);
    # a still screen settles in ~0.15 s.
    look_s: float = 0.4
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
        # Thumbnail cells left out of change checks: learned on a run's first look and on each settle timeout.
        self.moving: np.ndarray | None = None

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
        elif kind == "scroll" and touch:
            # A phone takes no wheel (Samsung DeX scrolled nothing for 3x5 notches, 2026-10-19): a finger drags the
            # page the other way. shortcut: through the screen's middle, as a wheel without a pointer; swipe at
            # Jev's box if a panel away from the middle needs scrolling.
            travel = max(-MAX_SCROLL, min(MAX_SCROLL, action.dy)) * SWIPE_PER_NOTCH / 2
            x, y1 = _hid_xy(0.5, 0.5 + travel)
            _, y2 = _hid_xy(0.5, 0.5 - travel)
            self._send({"op": "swipe", "x1": x, "y1": y1, "x2": x, "y2": y2, "ms": 300})
        elif kind == "scroll":
            # A notch per report, as a wheel turns: a browser just started took the first scroll event as its
            # baseline, and 5 notches in one report scrolled nothing (4 of 5 went through one at a time, v19-t1).
            dy = max(-MAX_SCROLL, min(MAX_SCROLL, action.dy))
            for _ in range(abs(dy)):
                self._send({"op": "scroll", "dy": 1 if dy > 0 else -1})
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

    def park(self) -> None:
        """Move the pointer off what the screen shows."""
        self.hover(*PARK)

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

    def wait_settled(self, loading: bool = False, react: bool = True, timeout_s: float | None = None,
                     since: np.ndarray | None = None) -> float:
        """Block until the screen stops changing. Returns seconds waited. `loading`: the last input may
        have started a page load, so the screen must stay still longer. `react`: input was just sent, so a
        still screen may only be slow to react; False skips that wait (a run's first look). `timeout_s`
        overrides the config's. `since`: a thumbnail from before the input; a first look that already
        differs from it has seen the reaction (a tab closed by Ctrl+W showed by then, and the wait for one
        held ~0.45 s on a still screen, 2026-10-20)."""
        cfg = self.settle
        stable_s = cfg.load_stable_s if loading else cfg.stable_s
        react_s = (max(cfg.react_s, cfg.load_react_s) if loading else cfg.react_s) if react else 0.0
        start = time.monotonic()
        prev = None
        still_since: float | None = None
        changed = False
        diffs: list[np.ndarray] = []
        while True:
            self.cancel.check()
            now = time.monotonic()
            elapsed = now - start
            frame = self.capture.grab()
            if frame is not None:
                thumb = thumbnail(frame)
                if prev is not None:
                    diffs = [*diffs[1 - MOTION_POLLS:], cv2.absdiff(prev, thumb) >= MOVING_DIFF]
                if prev is None and since is not None:
                    changed = self._change(since, thumb) >= cfg.threshold
                playing = self._motion(diffs) if len(diffs) == MOTION_POLLS else None
                if playing is not None and playing.mean() > MAX_MOVING:
                    playing = None
                if prev is not None and self._change(prev, thumb, playing) < cfg.threshold:
                    still_since = still_since if still_since is not None else now
                    if elapsed >= (cfg.min_s if changed else max(cfg.min_s, react_s)) \
                            and now - still_since >= stable_s:
                        if not react:  # nothing of ours moved it
                            self._learn(diffs)
                        return elapsed
                else:
                    changed = changed or prev is not None
                    still_since = None
                prev = thumb
            if elapsed >= (cfg.timeout_s if timeout_s is None else timeout_s):
                self._learn(diffs)
                return elapsed
            self.cancel.sleep(cfg.poll_s)

    def _learn(self, diffs: list[np.ndarray]) -> None:
        """Remember the cells that kept moving over the last polls."""
        if len(diffs) >= MOTION_POLLS // 2:
            self.moving = self._motion(diffs)

    @staticmethod
    def _motion(diffs: list[np.ndarray]) -> np.ndarray | None:
        """The cells moving in at least half of `diffs`, grown by one cell for a video's edges; None if none."""
        moving = (np.sum(diffs, axis=0) * 2 >= len(diffs)).astype(np.uint8)
        return cv2.dilate(moving, np.ones((3, 3), np.uint8)).astype(bool) if moving.any() else None

    def _change(self, a: np.ndarray, b: np.ndarray, playing: np.ndarray | None = None) -> float:
        """frame_change outside the cells moving on their own (all of them if nothing else is left)."""
        moving = self.moving if playing is None else playing if self.moving is None else self.moving | playing
        if moving is None or moving.all():
            return frame_change(a, b)
        return float(np.mean(cv2.absdiff(a, b)[~moving]))

    def wait_for_change(self, timeout_s: float | None = None, since: np.ndarray | None = None) -> bool:
        """Block until the screen differs from `since` (a thumbnail; default: now), as a load,
        restart or app start progresses, or `timeout_s` (default WAIT_S) passes. True if it changed."""
        if since is None:
            frame = self.capture.grab()
            if frame is None:
                return False
            since = thumbnail(frame)
        before = since
        deadline = time.monotonic() + (WAIT_S if timeout_s is None else timeout_s)
        while time.monotonic() < deadline:
            self.cancel.sleep(self.settle.poll_s)
            frame = self.capture.grab()
            if frame is None:
                return False
            if self._change(before, thumbnail(frame)) >= self.settle.threshold:
                return True
        return False

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
