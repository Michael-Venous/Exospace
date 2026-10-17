"""Live agent events for the touch display and projector (brain/agent/DESIGN.md §5).

The agent publishes; display clients subscribe. Publishing never blocks the
agent: a client that falls behind loses events instead of slowing a step.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any

Event = dict[str, Any]

HISTORY = 60  # replayed to a client that connects mid-task
CLIENT_BUFFER = 256


class EventBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: set[queue.Queue[Event]] = set()
        self._history: deque[Event] = deque(maxlen=HISTORY)
        self._frame = b""
        self._frame_id = 0

    def publish(self, ev: str, **fields: Any) -> None:
        event = {"ev": ev, "t": time.time(), **fields}
        with self._lock:
            self._history.append(event)
            clients = list(self._clients)
        _fan_out(clients, event)

    def set_frame(self, jpeg: bytes) -> None:
        """Latest screen image. Clients get a short notice, not the bytes, and
        frames stay out of the history so a reconnect isn't flooded."""
        with self._lock:
            self._frame = jpeg
            self._frame_id += 1
            clients = list(self._clients)
            notice = {"ev": "frame", "id": self._frame_id}
        _fan_out(clients, notice)

    def frame(self) -> tuple[int, bytes]:
        with self._lock:
            return self._frame_id, self._frame

    def subscribe(self) -> queue.Queue[Event]:
        client: queue.Queue[Event] = queue.Queue(maxsize=CLIENT_BUFFER)
        with self._lock:
            for event in self._history:
                client.put_nowait(event)
            if self._frame_id:
                client.put_nowait({"ev": "frame", "id": self._frame_id})
            self._clients.add(client)
        return client

    def unsubscribe(self, client: queue.Queue[Event]) -> None:
        with self._lock:
            self._clients.discard(client)


def _fan_out(clients: list[queue.Queue[Event]], event: Event) -> None:
    for client in clients:
        try:
            client.put_nowait(event)
        except queue.Full:
            pass
