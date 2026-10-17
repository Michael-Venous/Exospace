"""ScreenCapture (local backend) recovers when the monitor layout changes."""

import importlib.util
import threading
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from brain.capture import ScreenCapture, V4L2Capture

HAS_MSS = importlib.util.find_spec("mss") is not None
if HAS_MSS:
    import mss


@unittest.skipUnless(HAS_MSS, "mss is only installed with the local extra")
class ScreenCaptureTests(unittest.TestCase):
    def test_a_stale_monitor_layout_gets_a_fresh_handle(self):
        stale, fresh = Mock(monitors=[{}]), Mock(monitors=[{}])
        stale.grab.side_effect = mss.exception.ScreenShotError("BadMatch")
        fresh.grab.return_value = np.zeros((2, 2, 4), np.uint8)
        with patch.object(mss, "mss", side_effect=[stale, fresh]):
            self.assertEqual(ScreenCapture().grab().shape, (2, 2, 3))
        stale.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()


class CaptureReadLoopTests(unittest.TestCase):
    def test_a_bad_mjpg_frame_does_not_end_the_capture_thread(self):
        """The card's occasional empty MJPG frame makes OpenCV raise; reading must carry on."""
        good = np.full((36, 64, 3), 120, np.uint8)
        reads = iter([cv2.error("imdecode: empty buffer"), (True, good)])

        class Card:
            def read(self):
                item = next(reads, (True, good))
                if isinstance(item, Exception):
                    raise item
                return item

        capture = object.__new__(V4L2Capture)
        capture._cap, capture._lock, capture._frame, capture._stamp = Card(), threading.Lock(), None, 0.0
        capture._running = True
        thread = threading.Thread(target=capture._read_loop, daemon=True)
        thread.start()
        deadline = time.monotonic() + 2
        while capture.grab() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        capture._running = False
        thread.join(timeout=1)
        self.assertIsNotNone(capture.grab())
