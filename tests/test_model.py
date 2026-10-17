"""Transport checks using the real SDK configuration and a mocked request."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from brain.agent.model import GeminiJSON
from brain.cancel import CancelToken


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "calls.jsonl"
        self.cancel = CancelToken()
        self.client = GeminiJSON("offline-test-key", "gemini-4-argon", "medium",
                                 self.cancel, 5, self.path)
        self.addCleanup(self.client.close)

    def test_structured_request_and_private_timing_log(self):
        with patch.object(self.client.client.models, "generate_content",
                          return_value=SimpleNamespace(text='{"ok": true}')) as generate:
            result = self.client.generate(b"private image", "private goal", "system",
                                          {"type": "object"}, role="planner", run_id="r", step=2)
        self.assertEqual(result, {"ok": True})
        config = generate.call_args.kwargs["config"]
        self.assertEqual(config.thinking_config.thinking_level.value.lower(), "medium")
        self.assertEqual(generate.call_args.kwargs["model"], "gemini-4-argon")
        raw = self.path.read_text()
        self.assertNotIn("private", raw)
        self.assertNotIn("offline-test-key", raw)
        log = json.loads(raw)
        self.assertEqual((log["role"], log["run_id"], log["step"]), ("planner", "r", 2))
        self.assertTrue(log["success"])

    def test_malformed_json_retry_is_bounded(self):
        with patch.object(self.client.client.models, "generate_content",
                          return_value=SimpleNamespace(text="[]")) as generate, \
                patch.object(self.cancel, "sleep"):
            with self.assertRaises(ValueError):
                self.client.generate(b"image", "goal", "system", {}, role="fast")
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(len(self.path.read_text().splitlines()), 2)

    def test_timeout_is_not_retried_or_swallowed(self):
        with patch.object(self.client.client.models, "generate_content",
                          side_effect=TimeoutError("timeout")) as generate:
            with self.assertRaises(TimeoutError):
                self.client.generate(b"image", "goal", "system", {}, role="fast")
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(json.loads(self.path.read_text())["error"], "TimeoutError")


if __name__ == "__main__":
    unittest.main()
