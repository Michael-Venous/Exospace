"""Device detection: a screen that shows nothing to tell the OS by gets the OS key once."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from brain.device import detect_profile


class ProbeTests(unittest.TestCase):
    def setUp(self):
        frame = np.zeros((100, 160, 3), np.uint8)
        frame[50:60, 80:86] = 255  # a cursor on a black desktop
        self.capture = Mock()
        self.capture.grab.return_value = frame
        sleep = patch("brain.cancel.CancelToken.sleep")
        sleep.start()
        self.addCleanup(sleep.stop)

    def test_an_unknown_os_is_probed_with_the_os_key_then_escape(self):
        executor, press = Mock(), Mock()
        executor.classify_os.side_effect = ["unknown", "linux"]  # a black desktop, then its launcher
        profile = detect_profile(self.capture, executor, press=press)
        self.assertEqual(profile.os, "linux")
        self.assertEqual([c.args[0] for c in press.call_args_list], [["GUI"], ["ESCAPE"]])

    def test_a_known_os_presses_nothing(self):
        executor, press = Mock(), Mock()
        executor.classify_os.return_value = "macos"
        self.assertEqual(detect_profile(self.capture, executor, press=press).os, "macos")
        press.assert_not_called()


if __name__ == "__main__":
    unittest.main()
