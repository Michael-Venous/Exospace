"""ScreenCapture (local backend) recovers when the monitor layout changes."""

import importlib.util
import unittest
from unittest.mock import Mock, patch

import numpy as np

from brain.capture import ScreenCapture

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
