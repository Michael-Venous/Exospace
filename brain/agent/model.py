"""Bounded Gemini JSON calls, with separate per-model latency records."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors, types

from brain.cancel import CancelToken


class GeminiJSON:
    def __init__(
        self, api_key: str, model: str, thinking: str | None,
        cancel: CancelToken, timeout_s: float, log_path: Path,
    ) -> None:
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (see .env.example)")
        self.model = model
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
        config = types.GenerateContentConfig(
            system_instruction=system, response_mime_type="application/json",
            response_json_schema=schema,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking) if self.thinking else None,
        )
        contents = [types.Part.from_bytes(data=image, mime_type="image/jpeg"), prompt]
        for attempt in range(1, 3):
            self.cancel.check()
            started = time.monotonic()
            success = False
            error = ""
            status_code = None
            retry_delay = 0.5 * attempt
            try:
                response = self.cancel.run(lambda: self.client.models.generate_content(
                    model=self.model, contents=contents, config=config,
                ))
                data = json.loads(response.text or "")
                if not isinstance(data, dict):
                    raise ValueError("model response must be a JSON object")
                success = True
                return data
            except errors.APIError as exc:
                error = type(exc).__name__
                status_code = exc.code
                retry_delay = 10.0 if exc.code == 429 else 2.0
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise
            except (ValueError, TypeError) as exc:
                error = type(exc).__name__
                if attempt == 2:
                    raise
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a") as stream:
                    stream.write(json.dumps({
                        "run_id": run_id, "step": step, "role": role, "model": self.model,
                        "attempt": attempt, "elapsed_s": time.monotonic() - started,
                        "success": success, "error": error,
                        "status_code": status_code,
                    }) + "\n")
            self.cancel.sleep(retry_delay)

    def close(self) -> None:
        self.client.close()
