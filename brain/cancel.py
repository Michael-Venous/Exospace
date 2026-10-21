"""The single cancellation path (PLAN.md §5.6).

Disconnects, voice barge-in and repeated verification failures all call
`CancelToken.cancel`. Anything long-running checks the token, and blocking
model calls go through `CancelToken.run` so they can be abandoned instantly.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")

_POLL_S = 0.02
# As many as Jev's own pool (brain/agent/jev.py): a step asks up to 6 at once, and with 4 workers the 5th and 6th
# waited for one to finish (5-8 calls of 0.3 s took 0.6 s; 1-4, 0.3 s).
_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="exospace-call")


class Cancelled(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason = ""

    def cancel(self, reason: str) -> None:
        if not self._event.is_set():
            self._reason = reason
            self._event.set()

    def reset(self) -> None:
        self._reason = ""
        self._event.clear()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise Cancelled(self._reason)

    def sleep(self, seconds: float) -> None:
        if self._event.wait(seconds):
            raise Cancelled(self._reason)

    def run(self, fn: Callable[[], T]) -> T:
        """Run a blocking call, but stop waiting the moment we're cancelled.
        An abandoned call finishes in the background and its result is dropped."""
        future: Future[T] = _pool.submit(fn)
        try:
            while True:
                self.check()
                try:
                    result = future.result(timeout=_POLL_S)
                    self.check()
                    return result
                except TimeoutError:
                    if future.done():
                        return future.result()  # propagate a timeout raised by fn
        finally:
            future.cancel()  # also remove queued calls abandoned by cancellation
