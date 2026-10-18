"""Bounded JSON calls to the language model, with separate per-model latency records.

z.ai and Gemini are interchangeable: `ZaiJSON` (z.ai GLM) serves every executor when ZAI_API_KEY is set, otherwise `GeminiJSON`.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, Iterable

import httpx

from brain.cancel import CancelToken
from brain.config import Config

RETRY_STATUS = {429, 500, 502, 503, 504}
OVERLOADED = frozenset({429, 503})  # high demand or quota: switch models instead of waiting


class GeminiJSON:
    def __init__(
        self, api_key: str, model: str, thinking: str | None,
        cancel: CancelToken, timeout_s: float, log_path: Path,
        fallback_model: str = "",
    ) -> None:
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (see .env.example)")
        from google import genai  # imported here so z.ai-only installs start fast
        from google.genai import errors, types

        self._errors, self._types = errors, types
        self.model = model
        self.fallback_model = fallback_model
        self.cancel = cancel
        self.thinking = thinking
        self.log_path = log_path
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout_s * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    def generate(
        self, image: bytes, prompt: str, system: str, schema: dict[str, Any],
        *, role: str, run_id: str = "", step: int = 0,
    ) -> dict[str, Any]:
        types = self._types
        config = types.GenerateContentConfig(
            system_instruction=system, response_mime_type="application/json",
            response_json_schema=schema,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking) if self.thinking else None,
        )
        contents = [types.Part.from_bytes(data=image, mime_type="image/jpeg"), prompt]
        next_model = self.model
        for attempt in range(1, 3):
            model = next_model
            self.cancel.check()
            started = time.monotonic()
            success = False
            error = ""
            status_code = None
            retry_delay = 0.5 * attempt
            try:
                response = self.cancel.run(lambda: self.client.models.generate_content(
                    model=model, contents=contents, config=config,
                ))
                data = json.loads(response.text or "")
                if not isinstance(data, dict):
                    raise ValueError("model response must be a JSON object")
                success = True
                return data
            except self._errors.APIError as exc:
                error = type(exc).__name__
                status_code = exc.code
                retry_delay = 10.0 if exc.code == 429 else 2.0
                if exc.code not in RETRY_STATUS or attempt == 2:
                    raise
                if exc.code in OVERLOADED and self.fallback_model:
                    next_model, retry_delay = self.fallback_model, 0.0
            except (ValueError, TypeError) as exc:
                error = type(exc).__name__
                if attempt == 2:
                    raise
            except httpx.TransportError as exc:  # a timeout or dropped connection (a WriteTimeout ended a live run)
                error = type(exc).__name__
                if attempt == 2:
                    raise
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                _log(self.log_path, run_id, step, role, model, attempt, started, success, error, status_code)
            self.cancel.sleep(retry_delay)

    def close(self) -> None:
        self.client.close()


class ZaiJSON:
    """z.ai's OpenAI-compatible chat API. It has no schema mode, so the schema
    rides in the system prompt and the first JSON object in the reply is used."""

    # In e2e runs 8 of 11 retried calls succeeded on the second try and the other 3 failed again,
    # each ending its run. The third try is made only when the run would otherwise end.
    ATTEMPTS = 3

    def __init__(
        self, api_key: str, model: str, thinking: str | None,
        cancel: CancelToken, timeout_s: float, log_path: Path, base_url: str,
        fallback_model: str = "",
    ) -> None:
        if not api_key:
            raise RuntimeError("ZAI_API_KEY is not set (see .env.example)")
        self.model = model
        self.fallback_model = fallback_model
        self.cancel = cancel
        self.thinking = thinking
        self.log_path = log_path
        self.client = httpx.Client(base_url=base_url.rstrip("/") + "/", timeout=timeout_s,
                                   headers={"Authorization": f"Bearer {api_key}"})

    def generate(
        self, image: bytes, prompt: str, system: str, schema: dict[str, Any],
        *, role: str, run_id: str = "", step: int = 0,
    ) -> dict[str, Any]:
        body = {
            "model": self.model,
            "thinking": {"type": "enabled" if self.thinking else "disabled"},
            "messages": [
                {"role": "system", "content": system
                 + "\n\nReply with only one JSON object of this shape (? = optional):\n" + shape(schema)},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}},
                    {"type": "text", "text": prompt},
                ]},
            ],
        }
        model = self.model
        for attempt in range(1, self.ATTEMPTS + 1):
            body["model"] = model
            self.cancel.check()
            started = time.monotonic()
            success = False
            error = ""
            status_code = None
            retry_delay = 0.5 * attempt
            try:
                response = self.cancel.run(lambda: self.client.post("chat/completions", json=body))
                status_code = response.status_code
                response.raise_for_status()
                data = first_json_object(response.json()["choices"][0]["message"]["content"] or "",
                                         schema.get("required", ()))
                success = True
                return data
            except httpx.HTTPStatusError as exc:
                error = type(exc).__name__
                retry_delay = 10.0 if status_code == 429 else 2.0
                if status_code not in RETRY_STATUS or attempt == self.ATTEMPTS:
                    raise
                if status_code in OVERLOADED and self.fallback_model:
                    model, retry_delay = self.fallback_model, 0.0
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                error = type(exc).__name__
                if attempt == self.ATTEMPTS:
                    raise ValueError("model reply was not a usable JSON object") from exc
            except httpx.TransportError as exc:  # a hung or dropped call; 4 of ~680 ended whole runs
                error = type(exc).__name__
                if attempt == self.ATTEMPTS:
                    raise
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                _log(self.log_path, run_id, step, role, model, attempt, started, success, error, status_code)
            self.cancel.sleep(retry_delay)

    def close(self) -> None:
        self.client.close()


def shape(schema: dict[str, Any]) -> str:
    """Compact type sketch of a JSON Schema. GLM tends to echo raw JSON Schema back; this
    is faster to read and, not being valid JSON, can never be mistaken for the reply."""
    if "enum" in schema:
        return "|".join(json.dumps(value) for value in schema["enum"])
    kind = schema.get("type")
    if kind == "object":
        required = set(schema.get("required", []))
        return "{" + ", ".join(f'"{name}"{"" if name in required else "?"}: {shape(sub)}'
                               for name, sub in schema.get("properties", {}).items()) + "}"
    if kind == "array":
        limit = f" (at most {schema['maxItems']})" if "maxItems" in schema else ""
        return "[" + shape(schema.get("items", {})) + ", ...]" + limit
    if kind in ("integer", "number") and "minimum" in schema and "maximum" in schema:
        return f"{kind} {schema['minimum']}-{schema['maximum']}"
    return kind or "any"


def first_json_object(text: str, required: Iterable[str] = ()) -> dict[str, Any]:
    """GLM may wrap JSON in prose, code fences or box tokens; take the first object
    with the required keys (a malformed reply must not yield one of its inner objects)."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        for candidate in (text[start:], _balanced(text[start:], swap=False),
                          _balanced(text[start:], swap=True)):
            try:
                value, _ = decoder.raw_decode(candidate)
            except ValueError:
                continue
            if isinstance(value, dict) and all(key in value for key in required):
                return value
    raise ValueError("model reply has no JSON object")


def _balanced(text: str, swap: bool) -> str:
    """Repair GLM's bracket slips: a wrong closer is dropped ('}}]') or swapped for the
    expected one ('}}, "next": ...'), and missing closers are added at the end."""
    out: list[str] = []
    expected: list[str] = []
    in_string = escaped = False
    for char in text:
        if in_string:
            escaped, in_string = (False, True) if escaped else (char == "\\", char != '"')
        elif char == '"':
            in_string = True
        elif char in "{[":
            expected.append("}" if char == "{" else "]")
        elif char in "}]":
            if not expected:
                continue
            if char != expected[-1]:
                if not swap:
                    continue
                char = expected[-1]
            expected.pop()
            if not expected:
                return "".join(out) + char
        out.append(char)
    return "".join(out) + "".join(reversed(expected))


def language_model(cfg: Config, cancel: CancelToken, *, planner: bool = False) -> GeminiJSON | ZaiJSON:
    """The configured provider: z.ai when `cfg.uses_zai`, else Gemini."""
    timeout = cfg.planner_timeout_s if planner else cfg.model_timeout_s
    if cfg.uses_zai:
        return ZaiJSON(cfg.zai_api_key, cfg.zai_planner_model if planner else cfg.zai_model,
                       cfg.planner_thinking if planner else cfg.zai_thinking, cancel, timeout,
                       cfg.model_log, cfg.zai_base_url, "" if planner else cfg.zai_fallback_model)
    return GeminiJSON(cfg.gemini_api_key, cfg.planner_model if planner else cfg.gemini_model,
                      cfg.planner_thinking if planner else cfg.gemini_thinking, cancel, timeout,
                      cfg.model_log, "" if planner else cfg.fallback_model)


def _log(path: Path, run_id: str, step: int, role: str, model: str, attempt: int,
         started: float, success: bool, error: str, status_code: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps({
            "run_id": run_id, "step": step, "role": role, "model": model,
            "attempt": attempt, "elapsed_s": time.monotonic() - started,
            "success": success, "error": error, "status_code": status_code,
        }) + "\n")
