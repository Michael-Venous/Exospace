"""Transport checks using the real SDK configuration and a mocked request."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from google.genai import errors

from brain.agent.model import GeminiJSON, ZaiJSON, first_json_object, language_model, shape
from brain.cancel import CancelToken
from brain.config import Config


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


    def test_dropped_connection_is_retried_once(self):
        with patch.object(self.client.client.models, "generate_content",
                          side_effect=[httpx.WriteTimeout("slow"), SimpleNamespace(text="{}")]) as generate, \
                patch.object(self.cancel, "sleep"):
            self.assertEqual(self.client.generate(b"image", "goal", "system", {}, role="fast"), {})
        self.assertEqual(generate.call_count, 2)


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
        if isinstance(content, Exception):
            raise content
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

    def test_reply_without_object_is_retried(self):
        self.replies += [(200, "no json here"), (200, "[1, 2]"), (200, "{bad}")]
        with patch.object(self.cancel, "sleep"), self.assertRaises(ValueError):
            self.client.generate(b"image", "goal", "system", {}, role="fast")
        self.assertEqual(len(self.requests), 3)

    def test_high_demand_switches_to_fallback_like_gemini(self):
        self.client.fallback_model = "glm-fallback"
        for code in (503, 429):
            with self.subTest(code=code):
                self.requests.clear()
                self.replies += [(code, "busy"), (200, '{"ok": true}')]
                with patch.object(self.cancel, "sleep") as sleep:
                    result = self.client.generate(b"image", "goal", "system", {}, role="fast")
                self.assertEqual(result, {"ok": True})
                self.assertEqual([json.loads(r.content)["model"] for r in self.requests],
                                 ["glm-test", "glm-fallback"])
                sleep.assert_called_once_with(0.0)

    def test_overload_is_retried_but_client_errors_are_not(self):
        self.replies += [(429, ""), (200, '{"ok": 1}')]
        with patch.object(self.cancel, "sleep"):
            self.assertEqual(self.client.generate(b"i", "g", "s", {}, role="fast"), {"ok": 1})
        self.replies.append((401, ""))
        with self.assertRaises(httpx.HTTPStatusError):
            self.client.generate(b"i", "g", "s", {}, role="fast")
        self.assertEqual(len(self.requests), 3)

    def test_dropped_connection_is_retried(self):
        self.replies += [(200, "no json here"), (0, httpx.ReadTimeout("hung")), (200, '{"ok": 1}')]
        self.replies += [(0, httpx.ConnectError("down"))] * 3
        with patch.object(self.cancel, "sleep"):
            self.assertEqual(self.client.generate(b"i", "g", "s", {}, role="fast"), {"ok": 1})
            with self.assertRaises(httpx.ConnectError):
                self.client.generate(b"i", "g", "s", {}, role="fast")
        self.assertEqual(len(self.requests), 6)

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



class ProviderTests(unittest.TestCase):
    def test_zai_serves_every_executor(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        log = Path(temp.name) / "calls.jsonl"
        base = Config("offline-gemini-key", "gemini-test", None, "", 0, "", 0, 0, log, log,
                      model_log=log, zai_api_key="offline-zai-key")
        for executor in ("direct", "legacy-dual", "race", "jev-flash"):
            with self.subTest(executor=executor):
                cfg = replace(base, executor=executor)
                client = language_model(cfg, CancelToken())
                self.addCleanup(client.close)
                self.assertIsInstance(client, ZaiJSON)
                self.assertEqual(cfg.model_name, "glm-4.6v-flashx")
                self.assertEqual(client.fallback_model, "glm-4.6v")
        client = language_model(replace(base, executor="race", zai_api_key=""), CancelToken())
        self.addCleanup(client.close)
        self.assertIsInstance(client, GeminiJSON)


def api_error(code):
    return errors.APIError(code, {"error": {"message": "test", "status": "TEST"}})


class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "calls.jsonl"
        self.cancel = CancelToken()
        self.client = GeminiJSON("offline-test-key", "primary", "low", self.cancel, 5,
                                 self.path, fallback_model="fallback")
        self.addCleanup(self.client.close)

    def call(self, *responses):
        with patch.object(self.client.client.models, "generate_content",
                          side_effect=list(responses)) as generate, \
                patch.object(self.cancel, "sleep") as sleep:
            try:
                return self.client.generate(b"image", "goal", "system", {}, role="fast")
            finally:
                self.models = [c.kwargs["model"] for c in generate.call_args_list]
                self.sleeps = [c.args[0] for c in sleep.call_args_list]

    def logged_models(self):
        return [json.loads(line)["model"] for line in self.path.read_text().splitlines()]

    def test_high_demand_switches_to_fallback_without_waiting(self):
        result = self.call(api_error(503), SimpleNamespace(text='{"ok": true}'))
        self.assertEqual(result, {"ok": True})
        self.assertEqual(self.models, ["primary", "fallback"])
        self.assertEqual(self.logged_models(), ["primary", "fallback"])
        self.assertEqual(self.sleeps, [0.0])

    def test_quota_switches_to_fallback(self):
        result = self.call(api_error(429), SimpleNamespace(text='{"ok": true}'))
        self.assertEqual(result, {"ok": True})
        self.assertEqual(self.models, ["primary", "fallback"])

    def test_client_errors_do_not_fall_back(self):
        for code in (400, 401):
            with self.subTest(code=code), self.assertRaises(errors.APIError):
                self.call(api_error(code))
            self.assertEqual(self.models, ["primary"])

    def test_fallback_failure_surfaces(self):
        with self.assertRaises(errors.APIError) as raised:
            self.call(api_error(503), api_error(429))
        self.assertEqual(raised.exception.code, 429)
        self.assertEqual(self.models, ["primary", "fallback"])

    def test_other_server_errors_retry_primary(self):
        self.call(api_error(500), SimpleNamespace(text='{"ok": true}'))
        self.assertEqual(self.models, ["primary", "primary"])


if __name__ == "__main__":
    unittest.main()
