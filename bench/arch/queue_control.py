"""Bench-only request authority and per-packet telemetry for the optimized queue."""
from __future__ import annotations

import json
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

THUMB = (320, 200)  # change detection: about 9x9 laptop pixels per thumbnail pixel
CHANGED_LEVEL = 24  # a thumbnail pixel this much brighter or darker has changed
# Changed thumbnail pixels that count as a new view. Measured on a 2880x1800 frame: a blinking caret 3, one
# typed character 4, a checkbox tick 5, a typed word 16. The previous whole-screen mean saw none of these.
CHANGED_PIXELS = 5
EVENTS_KEPT = 2000  # recent events kept in memory; every event is also appended to the JSONL log


def thumbnail(frame) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), THUMB, interpolation=cv2.INTER_AREA)


def changed(before: np.ndarray | None, after: np.ndarray) -> bool:
    return before is None or int(np.count_nonzero(cv2.absdiff(before, after) > CHANGED_LEVEL)) >= CHANGED_PIXELS


class Superseded(RuntimeError):
    pass


@dataclass(frozen=True)
class Planned:
    action: dict
    epoch: int
    request: int
    view: int


class Control:
    """One authority for request lifecycle, mutation validity and timing."""
    def __init__(self) -> None:
        self.epoch = 0
        self.view = 0
        self.input_seq = 0
        self.seen_input_seq = 0
        self.started = time.monotonic()
        self.events: deque[dict] = deque(maxlen=EVENTS_KEPT)
        self.events_log = None  # an open JSONL file receiving every event, when the agent has a log folder
        self.requests: dict[int, dict] = {}
        self.attempts: Counter = Counter()
        self.calls = 0
        self.frame = None
        self.thumb = None
        self.changed_at = self.started
        self.agent = None
        self.executor = None
        self.last_snapshot = None
        self.save_lock = threading.Lock()

    def event(self, kind: str, **fields) -> None:
        event = {'t': round(time.monotonic()-self.started, 4), 'request': self.agent.shared.request,
                 'kind': kind, **fields}
        self.events.append(event)
        if self.events_log is not None:
            self.events_log.write(json.dumps(event) + '\n')

    def valid(self, epoch: int) -> bool:
        return (epoch == self.epoch and not self.agent.shared.finished
                and not self.agent.brain.speaking and not self.agent.cancel.cancelled)

    def invalidate(self, *, batch_ended: bool = False) -> None:
        """`batch_ended`: the plan's own result ended it (an observation or a failure), not the user or a stop."""
        if batch_ended:
            self.agent.brain.ended_epoch = self.epoch
        self.epoch += 1
        self.agent.cancel.cancel('plan interrupted or stopped')
        self.agent.controller.release_all()
        self.agent.shared.queue.clear()
        self.agent.brain.cancel_call.set()

    def finish(self, status: str, message: str = '', *, speak: bool = True) -> None:
        """Must be called with the condition held. Also revokes in-flight input authority."""
        request = self.agent.shared.request
        if self.agent.shared.finished and self.requests.get(request,{}).get('status') in ('verified','asked','blocked','uncertain','cancelled'):
            return
        self.invalidate()
        self.agent.shared.finished = True
        self.agent.shared.held, self.agent.shared.held_goal = [], ''
        now = round(time.monotonic()-self.started, 4)
        self.requests.setdefault(request, {}).update(terminal_at=now, status=status)
        if message and speak:
            self.agent.brain.said.append(message)
            self.agent.brain.said_requests.append(request)
            self.agent.brain.conversation.append({'you': message})
        self.event('terminal', status=status)
        self.agent.cond.notify_all()

    def observe(self, frame) -> None:
        thumb = thumbnail(frame)
        if changed(self.thumb, thumb):
            self.view += 1
            self.changed_at = time.monotonic()
        self.thumb, self.frame = thumb, frame
        self.seen_input_seq = self.input_seq

    def save(self, path: Path) -> None:
        # Request states only: events stream to the JSONL log, so a long session's save stays small and holds
        # the shared lock (which every input packet takes) only briefly.
        with self.save_lock:
            with self.agent.cond:
                data = json.dumps({'requests': self.requests,
                                   'events': self.events_log.name if self.events_log else None}, indent=1)
                if self.events_log is not None:
                    self.events_log.flush()
            if data != self.last_snapshot:
                tmp = path.with_suffix('.tmp')
                tmp.write_text(data)
                tmp.replace(path)
                self.last_snapshot = data


class TimedTarget:
    def __init__(self, target, control: Control) -> None:
        self.target, self.control = target, control

    def __getattr__(self, key):
        return getattr(self.target, key)

    def grab(self):
        started = time.monotonic()
        frame = self.target.grab()
        if self.control.agent is not None:
            with self.control.agent.cond:
                self.control.event('capture', duration_s=round(time.monotonic()-started, 4))
        return frame

    def send(self, command: dict) -> None:
        c = self.control
        with c.agent.cond:
            if command['op'] != 'release_all':
                token = c.executor.active_plan
                if token is None or not c.valid(token.epoch):
                    raise Superseded('input belongs to a stopped or revised plan')
                now = round(time.monotonic()-c.started, 4)
                c.requests.setdefault(token.request, {}).setdefault('first_input_at', now)
                c.input_seq += 1
                c.event('hid', epoch=token.epoch, op=command['op'],
                        characters=len(command.get('text','')))
            # Lock only one protocol packet, not an entire typed document.
            self.target.send(command)


