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

    def test_one_monitor_is_grabbed(self):
        sct = Mock(monitors=[{"all": 1}, {"top": 1}, {"bottom": 1}])
        sct.grab.return_value = np.zeros((2, 2, 4), np.uint8)
        with patch.object(mss, "mss", return_value=sct):
            ScreenCapture(1).grab()
        sct.grab.assert_called_once_with({"top": 1})

    def test_the_voice_window_is_greyed_out(self):
        sct = Mock(monitors=[{"left": 0, "top": 10, "width": 4, "height": 4}])
        sct.grab.return_value = np.zeros((4, 4, 4), np.uint8)
        x11 = {"wmctrl": "0x01  0 host Exospace Voice — Mozilla Firefox\n0x2  0 host Editor\n",
               "xwininfo": "  Absolute upper-left X:  1\n  Absolute upper-left Y:  12\n  Width: 2\n  Height: 1\n",
               "xprop": "_NET_FRAME_EXTENTS(CARDINAL) = 0, 0, 1, 0\n"}  # its title bar: 1 px above
        with patch.object(mss, "mss", return_value=sct), patch("shutil.which", return_value="/usr/bin/wmctrl"), \
                patch("subprocess.run", lambda cmd, **_: Mock(stdout=x11[cmd[0]])):
            frame = ScreenCapture(0, hide=lambda: "Exospace Voice").grab()
        self.assertEqual(int(frame.sum()), 2 * 2 * 3 * 128)  # only the voice window's 2x2 px, rows 1-2 of the area

    def test_a_window_over_the_voice_window_stays_visible(self):
        sct = Mock(monitors=[{"left": 0, "top": 10, "width": 4, "height": 4}])
        sct.grab.return_value = np.ones((4, 4, 4), np.uint8)
        with patch.object(mss, "mss", return_value=sct), patch("shutil.which", return_value="/usr/bin/wmctrl"), \
                patch("brain.capture.window_rects", return_value=[(0, 10, 4, 4)]), \
                patch("brain.capture.rects_above", return_value=[(2, 12, 2, 2)]):  # a calculator over its corner
            frame = ScreenCapture(0, hide=lambda: "Exospace Voice").grab()
        self.assertEqual(frame[2:, 2:].tolist(), np.ones((2, 2, 3)).tolist())
        self.assertEqual(int((frame == 128).sum()), 12 * 3)


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
