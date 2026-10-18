"""ScreenCapture (local backend) recovers when the monitor layout changes."""

import unittest
from unittest.mock import Mock, patch

import mss
import numpy as np

from brain.capture import ScreenCapture


class ScreenCaptureTests(unittest.TestCase):
    def test_a_stale_monitor_layout_gets_a_fresh_handle(self):
        stale, fresh = Mock(monitors=[{}]), Mock(monitors=[{}])
        stale.grab.side_effect = mss.exception.ScreenShotError("BadMatch")
        fresh.grab.return_value = np.zeros((2, 2, 4), np.uint8)
        with patch.object(mss, "mss", side_effect=[stale, fresh]):
            self.assertEqual(ScreenCapture().grab().shape, (2, 2, 3))
        stale.close.assert_called_once()

    def test_one_monitor_is_grabbed(self):
        sct = Mock(monitors=[{"all": 1}, {"top": 1}, {"bottom": 1}])
        sct.grab.return_value = np.zeros((2, 2, 4), np.uint8)
        with patch.object(mss, "mss", return_value=sct):
            ScreenCapture(1).grab()
        sct.grab.assert_called_once_with({"top": 1})

    def test_the_voice_window_is_greyed_out(self):
        sct = Mock(monitors=[{"left": 0, "top": 10, "width": 4, "height": 4}])
        sct.grab.return_value = np.zeros((4, 4, 4), np.uint8)
        windows = Mock(stdout="0x1  0 1 12 2 1 host Exospace Voice — Mozilla Firefox\n0x2  0 0 10 4 4 host Editor\n")
        with patch.object(mss, "mss", return_value=sct), patch("shutil.which", return_value="/usr/bin/wmctrl"), \
                patch("subprocess.run", return_value=windows):
            frame = ScreenCapture(0, hide=lambda: "Exospace Voice").grab()
        self.assertEqual(int(frame.sum()), 2 * 3 * 128)  # only the voice window's 2x1 px, row 2 of the area


if __name__ == "__main__":
    unittest.main()
