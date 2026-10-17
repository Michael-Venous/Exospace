"""Turning exceptions into safe, useful messages, with full tracebacks kept locally.

Hardware and OS errors (Pico, capture card, serial port) carry no secrets and
are shown in full. Anything else, such as model-provider errors that can echo
request payloads, is summarized; its traceback goes only to the ignored log.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

from brain.capture import CaptureError
from brain.hid import HIDError

ERROR_LOG = Path("work/errors.log")
_LOCAL_ERRORS = (HIDError, CaptureError, OSError)


def describe(exc: BaseException, context: str) -> str:
    """Log the traceback and return a message safe for the console and transcript."""
    log_exception(exc, context)
    name = type(exc).__name__
    if isinstance(exc, _LOCAL_ERRORS):
        return f"{name}: {exc}"
    return f"{name}: {context} failed; details in {ERROR_LOG}"


def log_exception(exc: BaseException, context: str) -> None:
    try:
        ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ERROR_LOG.open("a") as log:
            log.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} {context}\n")
            log.write("".join(traceback.format_exception(exc)))
    except OSError:
        pass  # logging must never mask the original error
