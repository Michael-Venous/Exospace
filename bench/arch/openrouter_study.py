"""Study 7: are newer low-latency models faster than Gemini 3.8 Flash, at equal accuracy?

Same two tasks as before, all models run concurrently so they share network conditions:
  triage     32 spoken segments: act (with a point) / plan / none   (scored on DOM boxes)
  first step 18 planning requests x 2 with the runtime prompt + schema, first action
             grounded by the same Jev pipeline and scored with the same rubric.
Streaming is used to measure time to first token; output tokens/s comes from usage.
"""

from __future__ import annotations

import base64
import json
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from dotenv import load_dotenv

from bench.arch.cases import CASES
from bench.arch.first_step_study import POINTER, first_action, judge, requests
from bench.arch.triage_study import LLM_PROMPT, LLM_SCHEMA, SCREENS, score
from bench.arch.zoom_study import refine
from brain.agent.jev import JevVision
from brain.agent.jev_flash import PROMPT, SCHEMA
from brain.cancel import CancelToken
from brain.config import Config

OUT = Path("work/arch/openrouter")
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
MODELS = [
    "google/gemini-3.8-flash",
    "openai/gpt-6-luna",
    "anthropic/claude-haiku-5.5",
    "xiaomi/mimo-v2.6-flash",
    "qwen/qwen3.8-omni-flash",
    "nvidia/nemotron-3.5-super-vl",
]
DIRECT = "gemini-3.8-flash (Google API)"
NO_FALLBACK = "\nDo not provide fallback_x or fallback_y: Jev handles grounding for this plan."


def parse_json(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


class OpenRouterJSON:
    """`effort` "" turns reasoning off; `fastest` routes to the provider with the lowest latency."""

    def __init__(self, model: str, key: str, effort: str = "low", fastest: bool = False) -> None:
        self.model = model
        self.reasoning = {"effort": effort} if effort else {"enabled": False}
        self.provider = {"sort": "latency"} if fastest else None
        self.client = httpx.Client(timeout=90, headers={"Authorization": f"Bearer {key}"})

    def call(self, image: bytes, prompt: str, system: str, schema: dict) -> tuple[dict, dict]:
        body = {
            "model": self.model, "stream": True, "temperature": 0,
            "reasoning": self.reasoning,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "reply", "schema": schema, "strict": False}},
            "messages": [
                {"role": "system", "content": system + "\nReply with a single JSON object only."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}},
                    {"type": "text", "text": prompt}]},
            ],
            "usage": {"include": True},
        }
        if self.provider:
            body["provider"] = self.provider
        started = time.monotonic()
        first = None
        text, usage = "", {}
        with self.client.stream("POST", OPENROUTER, json=body) as response:
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.read()[:200]!r}")
            for line in response.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content") or ""
                    if piece and first is None:
                        first = time.monotonic() - started
                    text += piece
        total = time.monotonic() - started
        out_tokens = usage.get("completion_tokens") or 0
        gen_time = max(1e-6, total - (first or total))
        return parse_json(text), {"ttft": first, "total": total, "out_tokens": out_tokens,
                                  "tok_s": out_tokens / gen_time if first else None, "cost": usage.get("cost")}


class DirectGemini:
    """Baseline through the Google API (no streaming), as the runtime calls it."""

    def __init__(self, cfg: Config, model: str = "gemini-3.8-flash", thinking: str | None = "low") -> None:
        from google import genai
        from google.genai import types

        self.model, self.thinking = model, thinking
        self.types = types
        self.client = genai.Client(api_key=cfg.gemini_api_key, http_options=types.HttpOptions(timeout=90_000))

    def call(self, image: bytes, prompt: str, system: str, schema: dict) -> tuple[dict, dict]:
        t = self.types
        config = t.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                         response_json_schema=schema,
                                         thinking_config=t.ThinkingConfig(thinking_level=self.thinking) if self.thinking else None)
        started = time.monotonic()
        response = self.client.models.generate_content(
            model=self.model, contents=[t.Part.from_bytes(data=image, mime_type="image/jpeg"), prompt], config=config)
        total = time.monotonic() - started
        out = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
        return json.loads(response.text or "{}"), {"ttft": None, "total": total, "out_tokens": out, "tok_s": None}


def main() -> None:
    load_dotenv()
    cfg = Config.load()
    OUT.mkdir(parents=True, exist_ok=True)
    boxes = json.loads((SCREENS / "boxes.json").read_text())
    images = {s: (SCREENS / f"{s}.jpg").read_bytes() for s in boxes}
    key = os.environ["OPENROUTER_API_KEY"]
    clients = {DIRECT: DirectGemini(cfg), **{m: OpenRouterJSON(m, key) for m in MODELS}}
    jev = JevVision(cfg.typesafe_api_key, CancelToken(), OUT / "jev.jsonl", cfg.jev_model, 30)

    jobs = []
    for name in clients:
        jobs += [(name, "triage", c, None) for c in CASES]
        jobs += [(name, "first_step", (s, t, rub), r) for r in range(2) for s, t, rub in requests()]

    def run(job):
        name, task, item, repeat = job
        try:
            if task == "triage":
                data, timing = clients[name].call(images[item.screen], LLM_PROMPT.format(utterance=item.utterance),
                                                  "You control a device for a user.", LLM_SCHEMA)
                kind = data.get("kind")
                point = (data["x"] / 1000, data["y"] / 1000) if kind == "act" and "x" in data and "y" in data else None
                decision = kind if kind != "act" or point else "plan"
                outcome = score(decision, point, item, boxes)
                return {"model": name, "task": task, "utterance": item.utterance, "outcome": outcome, **timing}
            screen, text, rubric = item
            data, timing = clients[name].call(images[screen], f"Goal: {text}" + NO_FALLBACK, PROMPT, SCHEMA)
            action = first_action(data)
            point = None
            if action and action.get("type") in POINTER and action.get("target"):
                located = jev.locate(images[screen], action["target"])
                if located:
                    zoom = refine(jev, images[screen], action["target"], located)
                    point = zoom["point"] or located
            outcome = judge(data, action, point, boxes[screen], rubric)
            return {"model": name, "task": task, "utterance": text[:50], "repeat": repeat,
                    "outcome": outcome, "action": action, **timing}
        except Exception as exc:
            return {"model": name, "task": task, "outcome": "error", "error": f"{type(exc).__name__}: {exc}"[:200]}

    with ThreadPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(run, jobs))
    jev.close()
    (OUT / "rows.json").write_text(json.dumps(rows, indent=1))
    report(rows)


def report(rows: list[dict]) -> None:
    print("| Model | Triage correct | Triage unsafe | First step correct | wrong | errors | total p50 | TTFT p50 | out tok/s p50 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name in dict.fromkeys(r["model"] for r in rows):
        r = [x for x in rows if x["model"] == name]
        tri = [x for x in r if x["task"] == "triage"]
        fs = [x for x in r if x["task"] == "first_step"]
        timed = [x for x in r if x.get("total")]
        ttft = [x["ttft"] for x in timed if x.get("ttft")]
        tps = [x["tok_s"] for x in timed if x.get("tok_s")]
        med = lambda v: f"{statistics.median(v):.2f}" if v else "–"  # noqa: E731
        print(f"| {name} | {sum(x['outcome'] == 'correct' for x in tri)}/{len(tri)} | "
              f"{sum(x['outcome'] in ('false_execute', 'wrong_target') for x in tri)} | "
              f"{sum(x['outcome'] == 'ok' for x in fs)}/{len(fs)} | {sum(x['outcome'] in ('wrong', 'bad') for x in fs)} | "
              f"{sum(x['outcome'] == 'error' for x in r)} | {med([x['total'] for x in timed])} s | {med(ttft)} s | {med(tps)} |")
    errors = [x for x in rows if x["outcome"] == "error"]
    for e in errors[:8]:
        print("  error:", e["model"], e["error"])


if __name__ == "__main__":
    import sys

    if "--report" in sys.argv:
        report(json.loads((OUT / "rows.json").read_text()))
    else:
        main()
