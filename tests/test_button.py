"""The Pi's push-to-talk button: gpiomon edges become debounced presses and releases."""

import io
import time
import unittest
from unittest.mock import patch

from brain.voice import button


class ButtonTests(unittest.TestCase):
    def watch(self, edges):
        seen = []
        with patch("brain.voice.button.subprocess.Popen") as popen, patch.object(button, "DEBOUNCE_S", 0.02):
            popen.return_value.stdout = io.StringIO(edges)
            button.watch(seen.append)
            time.sleep(0.1)
        return seen

    def test_a_bouncing_press_is_one_press(self):
        self.assertEqual(self.watch("0\n1\n0\n"), [True])

    def test_a_bounce_that_ends_released_is_nothing(self):
        self.assertEqual(self.watch("0\n1\n"), [])


if __name__ == "__main__":
    unittest.main()
