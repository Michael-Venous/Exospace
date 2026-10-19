"""Pointer calibration finds the target's screen inside the capture frame, black wallpaper or not."""

import unittest
from unittest.mock import patch

import numpy as np

from brain.calibrate import pointer_rect
from brain.types import Rect

SCREEN = Rect(64, 0, 1216, 720)  # a 16:10 laptop mirrored into a 1280x720 capture: 5% bars


class FakeTarget:
    """A black desktop on a black-barred frame; the only thing that shows is the cursor."""

    def __init__(self, screen=SCREEN, cursor=True):
        self.screen, self.cursor, self.at = screen, cursor, (0.5, 0.5)

    def move(self, x, y):
        self.at = (x, y)

    def grab(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        if self.cursor:
            s = self.screen
            x = int(s.x0 + self.at[0] * (s.x1 - s.x0))
            y = int(s.y0 + self.at[1] * (s.y1 - s.y0))
            frame[y:y + 20, x:x + 12] = 255  # the cursor image, its tip at the pointer
        return frame


class CalibrateTests(unittest.TestCase):
    def setUp(self):
        sleep = patch("brain.calibrate.time.sleep")
        sleep.start()
        self.addCleanup(sleep.stop)

    def test_the_screen_is_found_where_the_cursor_appears(self):
        target = FakeTarget()
        rect = pointer_rect(target, target.move)
        self.assertLessEqual(max(abs(a - b) for a, b in zip(rect.as_list(), SCREEN.as_list())), 2)
        self.assertEqual(target.at, (0.5, 0.5))  # the pointer is left in the middle

    def test_other_changes_on_screen_are_ignored(self):
        class Busy(FakeTarget):
            ticks = 0

            def grab(self):
                frame = super().grab()
                Busy.ticks += 1
                frame[600:612, 100 + 30 * (Busy.ticks % 3):112 + 30 * (Busy.ticks % 3)] = 255  # a redraw
                return frame
        target = Busy()
        rect = pointer_rect(target, target.move)
        self.assertLessEqual(max(abs(a - b) for a, b in zip(rect.as_list(), SCREEN.as_list())), 2)

    def test_no_cursor_means_no_answer(self):
        target = FakeTarget(cursor=False)
        self.assertIsNone(pointer_rect(target, target.move))


if __name__ == "__main__":
    unittest.main()
