"""Run the real executors on any OpenRouter model, for benchmarks only.

The runtime's z.ai client already speaks the OpenAI chat API; OpenRouter needs its own
reasoning switch instead of z.ai's `thinking` field, and the lowest-latency provider. `routed`
returns a Config that sends every language-model call there, so race, jev and direct run
unchanged on the model under test. The runtime itself never imports this module.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

import httpx

from brain.agent import model as runtime_model
from brain.config import Config

BASE_URL = "https://openrouter.ai/api/v1"


class _RouterClient(httpx.Client):
    """Rewrites z.ai's request body into OpenRouter's on the way out."""

    def __init__(self, effort: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reasoning = {"effort": effort} if effort else {"enabled": False}

    def post(self, url: str, *, json: dict | None = None, **kwargs: Any) -> httpx.Response:  # type: ignore[override]
        if json is not None:
            json = {k: v for k, v in json.items() if k != "thinking"}
            json |= {"reasoning": self.reasoning, "temperature": 0, "provider": {"sort": "latency"}}
        return super().post(url, json=json, **kwargs)


class RouterJSON(runtime_model.ZaiJSON):
    effort = ""  # set by `routed`: "" turns reasoning off, else "minimal" / "low" / ...

    def __init__(self, api_key: str, model: str, thinking: str | None, cancel, timeout_s: float, log_path,
                 base_url: str, fallback_model: str = "") -> None:
        super().__init__(api_key, model, thinking, cancel, timeout_s, log_path, base_url, fallback_model)
        self.client.close()
        self.client = _RouterClient(self.effort, base_url=base_url.rstrip("/") + "/", timeout=timeout_s,
                                    headers={"Authorization": f"Bearer {api_key}"})


def routed(cfg: Config, model: str, effort: str = "") -> Config:
    """A Config whose language model is `model` on OpenRouter (reasoning `effort`, "" = off)."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    RouterJSON.effort = effort
    runtime_model.ZaiJSON = RouterJSON  # language_model() looks the class up at call time
    return replace(cfg, zai_api_key=key, zai_base_url=BASE_URL, zai_model=model, zai_planner_model=model,
                   zai_fallback_model="", zai_thinking=effort or "", model_timeout_s=20.0)
