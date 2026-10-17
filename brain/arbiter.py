"""Commit gate for pipelined, interruptible execution (PLAN.md §10, §11).

The voice layer submits segments through this gate while the user keeps talking;
the typed interface runs one segment at a time and does not need it yet.

Jev and the LLM never touch HID. They submit proposals; the arbiter is the only
path to input and enforces:
  1. one winner per segment: the first proposal claims it, later ones are dropped;
  2. in-order commit: segment N+1 may plan early but executes only after segment N;
  3. generations: an interruption/correction bumps the generation, and every
     proposal from an older generation is rejected, even if it arrives later;
  4. stale screens: a proposal planned on a screen that has since changed is
     returned for re-grounding instead of being executed blind.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from brain.types import Action


class Verdict(Enum):
    COMMITTED = "committed"
    LOST_RACE = "lost_race"  # another source already claimed this segment
    STALE_GENERATION = "stale_generation"  # interrupted/corrected since it was planned
    STALE_SCREEN = "stale_screen"  # screen changed: re-ground before executing
    CANCELLED = "cancelled"  # interrupted while waiting for its turn


@dataclass(frozen=True)
class Proposal:
    segment: int
    generation: int
    source: str  # "jev" | "llm"
    actions: tuple[Action, ...]
    screen_id: int  # id of the frame the proposal was planned on


@dataclass
class Arbiter:
    execute: Callable[[tuple[Action, ...]], None]  # the single HID writer
    current_screen: Callable[[], int]
    generation: int = 0
    next_segment: int = 0
    _claims: dict[int, str] = field(default_factory=dict)
    _cond: threading.Condition = field(default_factory=threading.Condition)

    def interrupt(self) -> int:
        """Correction/barge-in: void every in-flight proposal and pending claim."""
        with self._cond:
            self.generation += 1
            self._claims.clear()
            self._cond.notify_all()
            return self.generation

    def submit(self, proposal: Proposal) -> Verdict:
        with self._cond:
            if proposal.generation != self.generation:
                return Verdict.STALE_GENERATION
            owner = self._claims.setdefault(proposal.segment, proposal.source)
            if owner != proposal.source:
                return Verdict.LOST_RACE
            # in-order commit: wait for earlier segments (or an interruption)
            self._cond.wait_for(lambda: self.next_segment >= proposal.segment
                                or self.generation != proposal.generation)
            if self.generation != proposal.generation:
                return Verdict.CANCELLED
            if proposal.screen_id != self.current_screen():
                del self._claims[proposal.segment]  # let a fresh grounding re-claim
                return Verdict.STALE_SCREEN
            self.execute(proposal.actions)
            return Verdict.COMMITTED

    def finish_segment(self, segment: int) -> None:
        with self._cond:
            self.next_segment = max(self.next_segment, segment + 1)
            self._cond.notify_all()
