"""The voice queue on the real rig's protocol: the optimized executor over the real `PicoHID` and the real firmware
command handler (a pseudo-terminal between them, as in test_pico_loopback), with a capture card's letterboxed frame."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from bench.arch.optimized_queue import Planned, create
from bench.arch.pi_target import PiTarget
from brain.hid import PicoHID
from tests.test_pico_loopback import FakePico


class Card:
    """A 1920x1080 capture of a 16:10 screen: 96 px black bars left and right."""

    def __init__(self):
        self.frame = np.zeros((1080, 1920, 3), np.uint8)
        self.frame[:, 96:1824] = 120

    def grab(self):
        return self.frame.copy()

    def signal_present(self):
        return True

    def close(self):
        pass


class PiRigTests(unittest.TestCase):
    def setUp(self):
        master, slave = os.openpty()
        self.pico = FakePico(master)
        self.addCleanup(self.pico.close)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        with patch("bench.arch.pi_target.V4L2Capture", return_value=Card()), \
             patch("bench.arch.pi_target.PicoHID", side_effect=lambda port, baud: PicoHID(os.ttyname(slave), baud)):
            self.target = PiTarget(Mock(capture_device="/dev/video0", capture_width=1920, capture_height=1080,
                                        pico_serial="tty", pico_baud=115200))
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}), \
             patch("bench.arch.optimized_queue.OptimizedExecutor.start"), \
             patch("bench.arch.optimized_queue.OptimizedBrain.start"):
            self.agent = create(self.target, "deepseek/deepseek-v4.1-flash-1017", "", "test", Path(self.tmp.name), "")
        self.ex, self.c = self.agent.executor, self.agent.control
        self.addCleanup(self.target.close)
        self.addCleanup(lambda: (self.ex.checks.close(), self.agent.jev.close(), self.agent.brain.client.close(),
                                 self.c.events_log.close()))

    def run_action(self, action, at=(.5, .5)):
        self.agent.brain.hear("test request", False)
        self.agent.cancel.reset()
        self.c.observe(self.target.grab())
        self.ex.active_plan = Planned(action, self.c.epoch, self.agent.shared.request, self.c.view)
        self.ex._find = Mock(return_value=at)
        return self.ex._do(action)

    def test_frames_are_cropped_to_the_screen_inside_the_bars(self):
        self.assertEqual(self.target.grab().shape[:2], (1080, 1728))

    def test_every_queue_action_is_accepted_by_the_firmware(self):
        for action in ({"click": "Save"}, {"double_click": "a word"}, {"right_click": "a file"}, {"hover": "a menu"},
                       {"hover": "away"}, {"scroll": 3, "at": "the page"}, {"key": ["CTRL", "L"]},
                       {"key": ["GUI", "PAGEUP"]}, {"type": "hello, world"}):
            result, detail = self.run_action(action)
            self.assertEqual(result, "done", (action, detail))

    def test_triple_click_is_a_double_then_a_single_on_the_pico(self):
        self.assertEqual(self.run_action({"triple_click": "a paragraph"}, (.25, .5))[0], "done")
        counts = [call.args[3] for call in self.pico.mouse.click.call_args_list]
        self.assertEqual(counts, [2, 1])
        self.assertAlmostEqual(self.pico.mouse.click.call_args_list[0].args[0] / 32767, .25, places=3)

    def test_a_held_key_click_is_refused_with_a_keyboard_alternative(self):
        result, detail = self.run_action({"click": "the last word", "hold": ["SHIFT"]})
        self.assertEqual(result, "error")
        self.assertIn("SHIFT+RIGHT", detail)
        self.pico.mouse.click.assert_not_called()


if __name__ == "__main__":
    unittest.main()
