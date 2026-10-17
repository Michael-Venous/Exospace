"""Study 2: how soon can execution start on long multi-step requests?

Compares Flash returning one complete JSON plan (current) against Flash
streaming one stage per line (NDJSON), where stage 1 can run while the rest is
still being written. Also compares thinking levels. Stages are validated with
the runtime's own `validate_steps`.
"""

from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google import genai
from google.genai import types

from brain.agent.jev_flash import PROMPT as RUNTIME_PROMPT
from brain.agent.jev_flash import SCHEMA as RUNTIME_SCHEMA
from brain.agent.jev_flash import validate_steps
from brain.config import Config

OUT = Path("work/arch/stream")
SCREENS = Path("work/arch/screens")
MODEL = "gemini-3.8-flash"

REQUESTS = [
    ("desktop", "Open up the browser, open up the email, go back, go to youtube, search up test on google, "
                "copy this text, paste it here, draft me an email to sam@contoso.com, drag this file over, send it"),
    ("bios", "check the bios, find the memory settings, apply xmp, disable secure boot, save and reset"),
    ("phone", "open the play store, install duolingo, open it, and log in with autofill"),
    ("inbox", "click on janet's email that has the report info, draft up the report from it and reply to john with it"),
    ("desktop", "open up blender and make me a house"),
]

STREAM_RULES = """
OUTPUT FORMAT (streaming): write NDJSON, one JSON object per line, no prose, no code fences.
Each stage line: {"ready": "...", "expected": "...", "actions": [...]} using the action rules above.
Write the FIRST stage immediately, as soon as you know it; keep later stages coming one per line.
The LAST line is {"completion": "..."} describing the visible proof the whole goal is done.
"""


def first_line_stage(line: str) -> dict | None:
    try:
        obj = json.loads(line)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) and "actions" in obj else None


def run_stream(client: genai.Client, image: bytes, request: str, thinking: str | None) -> dict:
    config = types.GenerateContentConfig(
        system_instruction=RUNTIME_PROMPT + STREAM_RULES,
        thinking_config=types.ThinkingConfig(thinking_level=thinking) if thinking else None,
    )
    contents = [types.Part.from_bytes(data=image, mime_type="image/jpeg"), f"Goal: {request}"]
    started = time.monotonic()
    first_token = first_stage = None
    buffer, stages, invalid = "", [], 0
    for chunk in client.models.generate_content_stream(model=MODEL, contents=contents, config=config):
        text = chunk.text or ""
        if text and first_token is None:
            first_token = time.monotonic() - started
        buffer += text
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip().strip("`")
            if not line:
                continue
            stage = first_line_stage(line)
            if stage is None:
                continue
            try:
                validate_steps({"steps": [stage], "completion": "x"})
                stages.append(stage)
                if first_stage is None:
                    first_stage = time.monotonic() - started
            except ValueError:
                invalid += 1
    if buffer.strip() and (stage := first_line_stage(buffer.strip())):
        stages.append(stage)
        first_stage = first_stage or time.monotonic() - started
    return {"first_token": first_token, "first_stage": first_stage, "total": time.monotonic() - started,
            "stages": len(stages), "invalid_lines": invalid, "first": stages[0] if stages else None}


def run_whole(client: genai.Client, image: bytes, request: str, thinking: str | None) -> dict:
    config = types.GenerateContentConfig(
        system_instruction=RUNTIME_PROMPT, response_mime_type="application/json",
        response_json_schema=RUNTIME_SCHEMA,
        thinking_config=types.ThinkingConfig(thinking_level=thinking) if thinking else None,
    )
    contents = [types.Part.from_bytes(data=image, mime_type="image/jpeg"), f"Goal: {request}"]
    started = time.monotonic()
    response = client.models.generate_content(model=MODEL, contents=contents, config=config)
    total = time.monotonic() - started
    data = json.loads(response.text or "{}")
    try:
        steps = validate_steps(data)
    except ValueError:
        steps = []
    return {"first_token": total, "first_stage": total if steps else None, "total": total,
            "stages": len(steps), "invalid_lines": 0 if steps else 1, "first": steps[0] if steps else None}


STREAM_SCHEMA = {**RUNTIME_SCHEMA, "propertyOrdering": ["steps", "completion", "reasoning"]}


def complete_stages(text: str) -> list[dict]:
    """Extract every fully closed object inside the top-level "steps" array of a
    partially streamed JSON document."""
    start = text.find('"steps"')
    if start < 0 or (bracket := text.find("[", start)) < 0:
        return []
    out, depth, obj_start, in_str, esc = [], 0, None, False, False
    for i in range(bracket + 1, len(text)):
        ch = text[i]
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            elif ch != "\\":
                esc = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                out.append(json.loads(text[obj_start:i + 1]))
        elif ch == "]" and depth == 0:
            break
    return out


NO_FALLBACK = "\nDo not provide fallback_x or fallback_y: Jev handles grounding for this plan."
LOOKAHEAD = """
LOOKAHEAD: plan as far ahead as the workflow is predictable (up to 12 stages), including stages on screens
you expect to appear next. Jev checks each stage's `ready` condition on a fresh screenshot before running it
and triggers a replan on any mismatch, so optimistic lookahead is safe. Put the immediately executable stage first."""


def run_stream_schema(client: genai.Client, image: bytes, request: str, thinking: str | None,
                      lookahead: bool = False) -> dict:
    config = types.GenerateContentConfig(
        system_instruction=RUNTIME_PROMPT + (LOOKAHEAD if lookahead else ""), response_mime_type="application/json",
        response_json_schema=STREAM_SCHEMA,
        thinking_config=types.ThinkingConfig(thinking_level=thinking) if thinking else None,
    )
    contents = [types.Part.from_bytes(data=image, mime_type="image/jpeg"), f"Goal: {request}" + NO_FALLBACK]
    started = time.monotonic()
    first_token = first_stage = None
    text, seen, invalid = "", 0, 0
    for chunk in client.models.generate_content_stream(model=MODEL, contents=contents, config=config):
        piece = chunk.text or ""
        if piece and first_token is None:
            first_token = time.monotonic() - started
        text += piece
        stages = complete_stages(text)
        for stage in stages[seen:]:
            try:
                validate_steps({"steps": [stage], "completion": "x"})
                if first_stage is None:
                    first_stage = time.monotonic() - started
            except ValueError:
                invalid += 1
        seen = len(stages)
    final = complete_stages(text)
    return {"first_token": first_token, "first_stage": first_stage, "total": time.monotonic() - started,
            "stages": len(final), "invalid_lines": invalid, "first": final[0] if final else None}


def main() -> None:
    cfg = Config.load()
    OUT.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=cfg.gemini_api_key,
                          http_options=types.HttpOptions(timeout=90_000))
    lookahead = lambda c, i, r, t: run_stream_schema(c, i, r, t, lookahead=True)  # noqa: E731
    modes = [("stream_schema_low", run_stream_schema, "low"), ("stream_lookahead_low", lookahead, "low")]
    jobs = [(m, fn, th, s, req, rep) for m, fn, th in modes for s, req in REQUESTS for rep in range(2)]
    rows = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [(m, s, req, pool.submit(fn, client, (SCREENS / f"{s}.jpg").read_bytes(), req, th))
                   for m, fn, th, s, req, _ in jobs]
        for m, s, req, fut in futures:
            try:
                rows.append({"mode": m, "screen": s, "request": req[:60], **fut.result()})
            except Exception as exc:
                rows.append({"mode": m, "screen": s, "request": req[:60], "error": f"{type(exc).__name__}: {exc}"[:200]})
    (OUT / "rows_lookahead.json").write_text(json.dumps(rows, indent=1))
    for m, *_ in modes:
        ok = [r for r in rows if r["mode"] == m and "error" not in r and r["first_stage"]]
        errs = sum(1 for r in rows if r["mode"] == m and "error" in r)
        if not ok:
            print(m, "no successful rows; errors", errs)
            continue
        med = lambda k: round(statistics.median(r[k] for r in ok), 2)  # noqa: E731
        print(f"{m}: ok {len(ok)}/{sum(1 for r in rows if r['mode'] == m)} errors {errs} "
              f"first_stage p50 {med('first_stage')}s total p50 {med('total')}s stages p50 {med('stages')} "
              f"invalid lines {sum(r['invalid_lines'] for r in ok)}")
    for r in rows:
        if r["mode"] == "stream_lookahead_low":
            print(r["screen"], "| first", r.get("first_stage"), "| total", round(r.get("total", 0), 2), "| stages", r.get("stages"), "|", json.dumps(r.get("first"))[:160])


if __name__ == "__main__":
    main()
