"""Setup check: clear verdicts without hardware, and no key values printed."""

import tempfile
import unittest
from pathlib import Path

from brain.doctor import check_capture, check_groups, check_keys, check_uart


class DoctorTests(unittest.TestCase):
    def test_keys_report_presence_only(self):
        checks = {c.name: c for c in check_keys({"GEMINI_API_KEY": "secret-value"})}
        self.assertTrue(checks["GEMINI_API_KEY"].ok)
        self.assertFalse(checks["TYPESAFE_API_KEY"].ok)
        self.assertFalse(checks["DEEPGRAM_API_KEY"].ok)  # the voice service needs it
        self.assertNotIn("secret-value", " ".join(c.detail for c in checks.values()))

    def test_groups(self):
        checks = check_groups({"video"})
        self.assertEqual([c.ok for c in checks], [True, False])
        self.assertIn("usermod -aG dialout", checks[1].detail)

    def test_missing_uart_and_capture(self):
        self.assertFalse(check_uart("/dev/does-not-exist").ok)
        self.assertIn("plug in", check_capture("/dev/does-not-exist", 1920, 1080).detail)

    def test_accessible_uart(self):
        with tempfile.NamedTemporaryFile() as port:
            self.assertTrue(check_uart(port.name).ok)
        self.assertFalse(Path(port.name).exists())


if __name__ == "__main__":
    unittest.main()
