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


class BlankTests(unittest.TestCase):
    def test_a_black_desktop_with_a_cursor_is_live_video(self):
        from brain.capture import is_blank
        frame = np.zeros((1800, 2880, 3), np.uint8)
        self.assertTrue(is_blank(frame))
        frame[900:930, 1400:1420] = 255  # the mouse cursor
        self.assertFalse(is_blank(frame))


class ActiveRectTests(unittest.TestCase):
    def test_a_pillarboxed_phone_is_cropped(self):
        from brain.capture import active_rect
        frame = np.zeros((1080, 1920, 3), np.uint8)
        frame[:, 700:1220] = 200
        rect = active_rect(frame)
        self.assertEqual((rect.x0, rect.y0, rect.x1, rect.y1), (700, 0, 1220, 1080))

    def test_a_window_on_a_black_wallpaper_is_not_the_screen(self):
        from brain.capture import active_rect
        frame = np.zeros((1800, 2880, 3), np.uint8)
        frame[460:1340, 738:2142] = 200  # a small app window, black all around it
        rect = active_rect(frame)
        self.assertEqual((rect.x0, rect.y0, rect.x1, rect.y1), (0, 0, 2880, 1800))

    def test_a_dark_frame_lit_only_by_the_cursor_is_not_cropped(self):
        from brain.capture import active_rect
        frame = np.zeros((1800, 2880, 3), np.uint8)
        frame[:, 1431:1449] = 30  # a dim page edge-to-edge in one column
        frame[900:930, 1435:1445] = 255  # the cursor
        rect = active_rect(frame)
        self.assertEqual((rect.x0, rect.y0, rect.x1, rect.y1), (0, 0, 2880, 1800))


if __name__ == "__main__":
    unittest.main()
