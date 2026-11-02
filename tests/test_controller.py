"""The controller's screen settling and scrolling, which every click, key and scroll of the prototype goes through.

Kept from the runtime suite when this branch dropped the runtime's own tests (synthetic video, recording HID).
"""

import unittest

import numpy as np

from brain.cancel import CancelToken
from brain.capture import thumbnail
from brain.controller import Controller, SettleConfig
from brain.types import Action, DeviceProfile, Rect


class Capture:
    def __init__(self):
        self.frame = np.full((100, 160, 3), 80, dtype=np.uint8)

    def grab(self):
        return self.frame.copy()

    def signal_present(self):
        return True


class HID:
    def __init__(self, capture, change=True):
        self.commands, self.callbacks = [], []
        self.capture, self.change = capture, change

    def send(self, command):
        self.commands.append(command)
        if self.change and command["op"] not in ("release_all", "move"):  # a moved pointer changes a few pixels
            self.capture.frame[:] = 160 if self.capture.frame[0, 0, 0] == 80 else 80

    def on_event(self, callback):
        self.callbacks.append(callback)

    def is_mounted(self):
        return True


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.capture = Capture()
        self.hid = HID(self.capture)
        self.cancel = CancelToken()
        self.controller = Controller(self.hid, self.capture, self.cancel,
                                     SettleConfig(min_s=0, react_s=0, load_stable_s=0, load_react_s=0, stable_s=0,
                                                  timeout_s=.1, poll_s=.001))
        self.profile = DeviceProfile("linux", Rect(0, 0, 160, 100), "mouse", 1)

    def test_a_first_look_at_a_still_screen_waits_for_no_reaction(self):
        controller = Controller(self.hid, self.capture, self.cancel, SettleConfig(
            min_s=0, react_s=.3, load_stable_s=0, load_react_s=0, stable_s=0, timeout_s=1, poll_s=.001))
        self.assertLess(controller.wait_settled(react=False), .1)
        self.assertGreaterEqual(controller.wait_settled(), .3)

    def test_a_reaction_already_shown_needs_no_wait_for_one(self):
        controller = Controller(self.hid, self.capture, self.cancel, SettleConfig(
            min_s=0, react_s=.3, load_stable_s=0, load_react_s=0, stable_s=0, timeout_s=1, poll_s=.001))
        still = thumbnail(self.capture.grab())
        self.assertGreaterEqual(controller.wait_settled(since=still), .3)  # nothing shown yet
        self.assertLess(controller.wait_settled(since=255 - still), .1)

    def test_a_moving_screen_settles_at_the_given_timeout(self):
        controller = Controller(self.hid, self.capture, self.cancel, SettleConfig(timeout_s=5, poll_s=.001))
        frames = iter(np.full((100, 160, 3), v % 256, dtype=np.uint8) for v in range(0, 10**6, 40))
        self.capture.grab = lambda: next(frames)  # a video: every grab differs
        self.assertLess(controller.wait_settled(react=False, timeout_s=.05), .5)

    def test_a_video_learned_on_a_look_neither_holds_a_settle_nor_counts_as_a_change(self):
        controller = Controller(self.hid, self.capture, self.cancel, SettleConfig(timeout_s=5, poll_s=.001))
        tick = iter(range(10**6))

        def video():  # a corner of the screen plays; the rest is still
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            frame[:90, :160] = next(tick) * 40 % 256
            return frame
        self.capture.grab = video
        self.assertLess(controller.wait_settled(react=False, timeout_s=.05), .5)  # the look times out, learning it
        self.assertLess(controller.wait_settled(react=False), 1)  # a 5 s timeout otherwise
        self.assertFalse(controller.wait_for_change(.05))

    def test_a_video_playing_in_a_still_page_settles_on_the_first_look(self):
        controller = Controller(self.hid, self.capture, self.cancel, SettleConfig(timeout_s=5, poll_s=.001))
        tick = iter(range(10**6))

        def video(part):  # a part of the screen plays; the rest is still
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            frame[:int(360 * part), :int(640 * part)] = next(tick) * 40 % 256
            return frame
        self.capture.grab = lambda: video(.25)
        self.assertLess(controller.wait_settled(), 1)  # a 5 s timeout otherwise
        self.assertFalse(controller.changed(thumbnail(video(.25)), thumbnail(video(.25))))  # the video played on
        page = video(.25)
        page[300:] = 255
        self.assertTrue(controller.changed(thumbnail(video(.25)), thumbnail(page)))  # the page around it changed
        self.capture.grab = lambda: video(.8)  # most of the screen moving is a page drawn or scrolled
        self.assertGreaterEqual(controller.wait_settled(timeout_s=.3), .3)

    def test_scroll_is_capped_and_sent_a_notch_at_a_time(self):
        self.controller.execute(Action("scroll", dy=800), self.profile)
        self.controller.execute(Action("scroll", dy=-2), self.profile)
        scrolls = [c for c in self.hid.commands if c["op"] == "scroll"]
        self.assertEqual(scrolls, [{"op": "scroll", "dy": 1}] * 15 + [{"op": "scroll", "dy": -1}] * 2)


if __name__ == "__main__":
    unittest.main()
