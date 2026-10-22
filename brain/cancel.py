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
_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="exospace-call")
_worker = threading.local()


def _invoke(token: CancelToken, epoch: int, fn: Callable[[], T]) -> T:
    _worker.token, _worker.epoch = token, epoch
    try:
        token.check()
        return fn()
    finally:
        _worker.token = None


class Cancelled(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason = ""
        self._epoch = 0

    def cancel(self, reason: str) -> None:
        if not self._event.is_set():
            self._reason = reason
            self._event.set()

    def reset(self) -> None:
        self._epoch += 1  # a new request must never revive an abandoned model worker
        self._reason = ""
        self._event.clear()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set() or (getattr(_worker, "token", None) is self
                                       and _worker.epoch != self._epoch)

    def check(self) -> None:
        if self.cancelled:
            raise Cancelled(self._reason or "superseded request")

    def _check_epoch(self, epoch: int) -> None:
        self.check()
        if epoch != self._epoch:
            raise Cancelled("superseded request")

    def sleep(self, seconds: float) -> None:
        epoch = self._epoch
        self.check()
        if self._event.wait(seconds):
            raise Cancelled(self._reason or "superseded request")
        self._check_epoch(epoch)

    def run(self, fn: Callable[[], T]) -> T:
        """Run a blocking call, but stop waiting the moment we're cancelled.
        An abandoned call finishes in the background and its result is dropped."""
        # Model wrappers call run() again for the HTTP request. Queueing that nested call can fill
        # every worker with a caller waiting for another worker. The outer run already supplies the
        # cancellable wait, so nested work executes here instead of occupying another pool slot.
        if getattr(_worker, "token", None) is self:
            self.check()
            result = fn()
            self.check()
            return result
        epoch = self._epoch
        self.check()
        future: Future[T] = _pool.submit(_invoke, self, epoch, fn)
        try:
            while True:
                self._check_epoch(epoch)
                try:
                    result = future.result(timeout=_POLL_S)
                    self._check_epoch(epoch)
                    return result
                except TimeoutError:
                    if future.done():
                        return future.result()  # propagate a timeout raised by fn
        finally:
            future.cancel()  # also remove queued calls abandoned by cancellation
