"""Study 3: which LLM should propose steps that Jev can't take alone?

For every request that needs planning, each LLM proposes a semantic plan with the
runtime prompt/schema. The first action is grounded by the same Jev pipeline
(locate + zoom refine) and scored against a rubric of acceptable first moves.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from google import genai
from google.genai import types

from bench.arch.cases import CASES
from bench.arch.stream_study import REQUESTS
from bench.arch.triage_study import SCREENS, hit
from bench.arch.zoom_study import refine
from brain.agent.jev import JevVision
from brain.agent.jev_flash import PROMPT, SCHEMA
from bench.experiments.live_experiment import LiveJSON
from brain.agent.model import GeminiJSON
from brain.cancel import CancelToken
from brain.config import Config
from brain.keys import canonical_combo

OUT = Path("work/arch/first_step")
POINTER = {"click", "double_click", "right_click", "hover", "drag"}
ANY = "*"

# utterance -> acceptable first moves: ("ptr", ids) | ("key", None or tuple of keys) | (kind,)
RUBRIC: dict[str, list[tuple]] = {
    "open up blender and make me a house": [("ptr", ("blender",))],
    "empty the recycle bin": [("ptr", ("recycle",)), ("ask_user",)],
    "draft up the report real quick and send it": [("ptr", ("reply", "janet")), ("ask_user",)],
    "delete all of these": [("ask_user",)],
    "no wait, the other one": [("ask_user",)],
    "reply to john saying I'll have it by five": [("ptr", ("reply", "john"))],
    "search for lofi hip hop": [("ptr", ("search",))],
    "go to the boot tab": [("ptr", ("boot_tab",)), ("key", None)],
    "make the usb boot first then save and exit": [("ptr", ("boot_tab",)), ("key", None)],
    "what's my cpu temp": [("answer", "54")],
    "text mom that I'm running late": [("ptr", ("messages",))],
    "make me a house": [("ptr", ("scripting_ws", "modeling_ws", "cube", "outliner_cube")), ("key", None),
                        ("type_text",), ("skill",)],
    "delete the cube": [("ptr", ("cube", "outliner_cube")), ("key", None)],
}
LONG_RUBRIC = {
    "desktop:Open up the browser": [("ptr", ("chrome",)), ("ask_user",)],
    "bios:check the bios": [("ptr", ("advanced_tab",)), ("key", None)],
    "phone:open the play store": [("ptr", (ANY,)), ("drag",), ("skill",)],
    "inbox:click on janet's email": [("ptr", ("janet",))],
    "desktop:open up blender and make me a house": [("ptr", ("blender",))],
}


def requests() -> list[tuple[str, str, list[tuple]]]:
    out = [(c.screen, c.utterance, RUBRIC[c.utterance]) for c in CASES if c.kind == "plan"]
    for screen, text in REQUESTS:
        key = next(k for k in LONG_RUBRIC if k.split(":", 1)[0] == screen and text.startswith(k.split(":", 1)[1]))
        out.append((screen, text, LONG_RUBRIC[key]))
    return out


def first_action(data: dict) -> dict | None:
    steps = data.get("steps") or []
    return (steps[0].get("actions") or [None])[0] if steps else None


def judge(data: dict, action: dict | None, point, boxes: dict, rubric: list[tuple]) -> str:
    if action is None:
        answer_ok = any(r[0] == "answer" and r[1] in json.dumps(data) for r in rubric)
        return "ok" if answer_ok else "no_step"
    kind = action.get("type")
    for spec in rubric:
        if spec[0] == "ptr" and kind in POINTER:
            if point is None:
                return "ungrounded"
            if spec[1] == (ANY,) or hit(point, spec[1], boxes):
                return "ok"
        elif spec[0] == "key" and kind == "hotkey":
            try:
                canonical_combo(action.get("keys") or [])
                return "ok"
            except ValueError:
                return "bad"
        elif spec[0] == kind:
            return "ok"
    return "wrong"


TEXT_ONLY_NOTE = ("\nNO SCREENSHOT is attached (for speed). Plan from the goal alone: describe each target by "
                  "the label or appearance it will have on screen. Jev sees the real screen, checks every "
                  "stage's ready condition and locates targets; it replans with a screenshot on mismatch.")


class TextOnly:
    """Same call shape as GeminiJSON.generate, but the image is never sent."""

    def __init__(self, api_key: str, model: str) -> None:
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60_000))
        self.model = model

    def generate(self, image: bytes, prompt: str, system: str, schema: dict, *, role: str) -> dict:
        config = types.GenerateContentConfig(
            system_instruction=system + TEXT_ONLY_NOTE, response_mime_type="application/json",
            response_json_schema=schema, thinking_config=types.ThinkingConfig(thinking_level="low"))
        response = self.client.models.generate_content(model=self.model, contents=[prompt], config=config)
        return json.loads(response.text or "{}")

    def close(self) -> None:
        self.client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()
    global OUT
    if args.text_only:
        OUT = OUT.with_name("first_step_text")
    cfg = Config.load()
    OUT.mkdir(parents=True, exist_ok=True)
    boxes = json.loads((SCREENS / "boxes.json").read_text())
    images = {s: (SCREENS / f"{s}.jpg").read_bytes() for s in boxes}
    cancel = CancelToken()
    log = OUT / "calls.jsonl"
    jev = JevVision(cfg.typesafe_api_key, cancel, log, cfg.jev_model, 30)
    if args.text_only:
        llms = {"flash-3.8-text": TextOnly(cfg.gemini_api_key, "gemini-3.8-flash"),
                "flash-lite-3.5-text": TextOnly(cfg.gemini_api_key, "gemini-3.5-flash-lite")}
    else:
        live = LiveJSON(cfg, cancel, PROMPT, SCHEMA)
        live.warm()
        llms = {
            "flash-3.8": GeminiJSON(cfg.gemini_api_key, "gemini-3.8-flash", "low", cancel, 60, log),
            "flash-lite-3.5": GeminiJSON(cfg.gemini_api_key, "gemini-3.5-flash-lite", "low", cancel, 60, log),
            "live-3.8": live,
        }
    rows = []
    for repeat in range(2):
        for screen, text, rubric in requests():
            prompt = f"Goal: {text}\nDo not provide fallback_x or fallback_y: Jev handles grounding for this plan."
            for name, llm in llms.items():
                started = time.monotonic()
                try:
                    data = llm.generate(images[screen], prompt, PROMPT, SCHEMA, role=name)
                except Exception as exc:
                    rows.append({"llm": name, "screen": screen, "request": text[:50], "outcome": "error",
                                 "error": f"{type(exc).__name__}: {exc}"[:160]})
                    continue
                llm_s = time.monotonic() - started
                action = first_action(data)
                point, jev_s = None, 0.0
                if action and action.get("type") in POINTER and action.get("target"):
                    t0 = time.monotonic()
                    first = jev.locate(images[screen], action["target"])
                    if first:
                        refined = refine(jev, images[screen], action["target"], first)
                        point = refined["point"] or first
                    jev_s = time.monotonic() - t0
                outcome = judge(data, action, point, boxes[screen], rubric)
                rows.append({"llm": name, "screen": screen, "request": text[:50], "repeat": repeat,
                             "outcome": outcome, "llm_s": llm_s, "jev_s": jev_s,
                             "action": action, "completion": data.get("completion", "")[:120]})
                print(name, outcome, round(llm_s, 2), "|", screen, text[:40], "|", json.dumps(action)[:90], flush=True)
    jev.close()
    for llm in llms.values():
        llm.close()
    (OUT / "rows.json").write_text(json.dumps(rows, indent=1))
    for name in llms:
        r = [x for x in rows if x["llm"] == name]
        good = [x for x in r if x["outcome"] == "ok"]
        timed = [x["llm_s"] + x["jev_s"] for x in r if "llm_s" in x]
        print(f"{name}: ok {len(good)}/{len(r)} outcomes {sorted({o: sum(x['outcome'] == o for x in r) for o in {x['outcome'] for x in r}}.items())} "
              f"step p50 {round(statistics.median(timed), 2)}s p90 {round(sorted(timed)[int(len(timed) * .9)], 2)}s")


if __name__ == "__main__":
    main()
