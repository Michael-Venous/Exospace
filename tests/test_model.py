"""Transport checks using the real SDK configuration and a mocked request."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from brain.agent.model import GeminiJSON, ZaiJSON, first_json_object, shape
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


class ZaiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "calls.jsonl"
        self.cancel = CancelToken()
        self.requests = []
        self.replies = []
        self.client = ZaiJSON("offline-test-key", "glm-test", None, self.cancel, 5, self.path,
                              "https://zai.invalid/api/paas/v4/")
        self.client.client = httpx.Client(base_url=self.client.client.base_url,
                                          headers=self.client.client.headers,
                                          transport=httpx.MockTransport(self._reply))
        self.addCleanup(self.client.close)

    def _reply(self, request):
        self.requests.append(request)
        status, content = self.replies.pop(0)
        return httpx.Response(status, json={"choices": [{"message": {"content": content}}]})

    def test_request_shape_fenced_reply_and_private_log(self):
        self.replies.append((200, 'Sure:\n```json\n{"ok": true, "n": {"x": 1}}\n```'))
        result = self.client.generate(b"private image", "private goal", "system",
                                      {"type": "object"}, role="fast", run_id="r", step=3)
        self.assertEqual(result, {"ok": True, "n": {"x": 1}})
        request = self.requests[0]
        self.assertEqual(request.url.path, "/api/paas/v4/chat/completions")
        self.assertEqual(request.headers["authorization"], "Bearer offline-test-key")
        body = json.loads(request.content)
        self.assertEqual((body["model"], body["thinking"]), ("glm-test", {"type": "disabled"}))
        self.assertTrue(body["messages"][0]["content"].endswith("\n{}"))
        image, text = body["messages"][1]["content"]
        self.assertTrue(image["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(text, {"type": "text", "text": "private goal"})
        raw = self.path.read_text()
        self.assertNotIn("private", raw)
        self.assertNotIn("offline-test-key", raw)
        log = json.loads(raw)
        self.assertEqual((log["role"], log["step"], log["status_code"], log["success"]),
                         ("fast", 3, 200, True))

    def test_reply_without_object_is_retried_once(self):
        self.replies += [(200, "no json here"), (200, "[1, 2]")]
        with patch.object(self.cancel, "sleep"), self.assertRaises(ValueError):
            self.client.generate(b"image", "goal", "system", {}, role="fast")
        self.assertEqual(len(self.requests), 2)

    def test_overload_is_retried_but_client_errors_are_not(self):
        self.replies += [(429, ""), (200, '{"ok": 1}')]
        with patch.object(self.cancel, "sleep"):
            self.assertEqual(self.client.generate(b"i", "g", "s", {}, role="fast"), {"ok": 1})
        self.replies.append((401, ""))
        with self.assertRaises(httpx.HTTPStatusError):
            self.client.generate(b"i", "g", "s", {}, role="fast")
        self.assertEqual(len(self.requests), 3)

    def test_shape_sketches_schema_without_valid_json(self):
        schema = {"type": "object", "required": ["a"], "properties": {
            "a": {"type": "array", "items": {"type": "string", "enum": ["x", "y"]}, "maxItems": 2},
            "b": {"type": "integer", "minimum": 0, "maximum": 9}}}
        sketch = shape(schema)
        self.assertEqual(sketch, '{"a": ["x"|"y", ...] (at most 2), "b"?: integer 0-9}')
        with self.assertRaises(ValueError):
            first_json_object(sketch)

    def test_first_json_object_skips_prose_and_non_objects(self):
        self.assertEqual(first_json_object('<|begin_of_box|>{"a": [1]}<|end_of_box|>'), {"a": [1]})
        self.assertEqual(first_json_object('see {bad} then [1] then {"b": 2} {"c": 3}'), {"b": 2})
        self.assertEqual(first_json_object('{"a": [{"b": "}"}}], "c": 1}'), {"a": [{"b": "}"}], "c": 1})
        self.assertEqual(first_json_object('{"a": [{"b": 1}'), {"a": [{"b": 1}]})
        self.assertEqual(first_json_object('{"a": [{"b": 1}}, "c": "d"}', ("a", "c")),
                         {"a": [{"b": 1}], "c": "d"})
        with self.assertRaises(ValueError):
            first_json_object("[1, 2]")
        reply = '{"reasoning": "r", "actions": [{"type": "click", "y": 0.2"}, {"type": "type_text"}]}'
        with self.assertRaises(ValueError):
            first_json_object(reply, ("reasoning", "actions"))


if __name__ == "__main__":
    unittest.main()
