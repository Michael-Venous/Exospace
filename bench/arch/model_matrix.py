"""Study 9: which model for which job, call by call, all candidates interleaved on the same screens.

  ground   point at a described control (DOM boxes), on the six fixture screens as drawn and on a
           2560 px canvas where every control is half as big (a 4K monitor's small targets).
           Jev's runtime locate + two zooms is the baseline.
  rewrite  the vague requests of `vague.py` (and four from the triage cases) rewritten into Jev's
           lines (`rewrite.py` prompt), scored on what the lines must and must not say.
  plan     Flash's first stage (study 3's 18 x 2 requests, runtime prompt and schema), grounded by Jev.
Reports hit rate, latency p50 / p90, time to first token and cost per model. Needs OPENROUTER_API_KEY;
GEMINI_API_KEY adds the Google API rows and TYPESAFE_API_KEY the Jev rows and the plan task.

  PYTHONPATH=. .venv/bin/python -m bench.arch.model_matrix --render      # screens only, no API calls
  PYTHONPATH=. .venv/bin/python -m bench.arch.model_matrix --pilot     # ~8 calls a model: availability, cost
  PYTHONPATH=. .venv/bin/python -m bench.arch.model_matrix [--tasks ground rewrite] [--models ...]
  PYTHONPATH=. .venv/bin/python -m bench.arch.model_matrix --report
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from bench.arch import fixtures, rewrite, vague
from bench.arch.first_step_study import POINTER, first_action, judge, requests
from bench.arch.openrouter_study import NO_FALLBACK, DirectGemini, OpenRouterJSON
from brain.agent.jev import JevVision
from brain.agent.jev_flash import PROMPT, SCHEMA
from brain.agent.triage import refine
from brain.cancel import CancelToken
from brain.config import Config
from brain.imaging import shrink

OUT = Path("work/arch/matrix")
SCREENS = OUT / "screens"
# (OpenRouter id, reasoning effort; "" = off). Fast vision models listed on 2026-10-20, plus study 7's finalists.
CANDIDATES = [
    ("google/gemini-3.8-flash", "low"), ("google/gemini-3.8-flash", ""),
    ("google/gemini-3.7-flash", ""), ("google/gemini-3.5-flash-lite", ""), ("google/gemini-3.1-flash-lite", ""),
    ("openai/gpt-6-luna", ""), ("anthropic/claude-haiku-5.5", ""), ("z-ai/glm-5.3-flashx", ""),
    ("inclusionai/ling-3.0-flash-vl", ""), ("deepseek/deepseek-v4.1-flash-1017", ""),
    ("google/gemma-4-26b-a4b-it", ""), ("qwen/qwen3.8-omni-flash", ""),
    ("moondream/moondream3.1-9b-a2b", ""), ("perceptron/perceptron-mk1.5", ""),
]
GOOGLE = [("gemini-3.8-flash", "low"), ("gemini-3.5-flash-lite", None)]
JEV = "jev (locate + 2 zooms)"

# (screen, description as a planner names it, acceptable box ids)
GROUND = [
    ("desktop", "the Google Chrome desktop icon", ("chrome",)), ("desktop", "the Blender 4.2 desktop icon", ("blender",)),
    ("desktop", 'the "Q3 report draft.docx" file icon', ("report_doc",)), ("desktop", "the Recycle Bin icon", ("recycle",)),
    ("desktop", "the Start button at the left of the taskbar", ("start",)),
    ("inbox", "the email from Janet Park in the message list", ("janet",)),
    ("inbox", "the email from HR Team about benefits enrollment", ("hr",)),
    ("inbox", "the Sent Items folder in the left pane", ("sent",)), ("inbox", "the Reply button", ("reply",)),
    ("inbox", "the browser's back arrow button", ("back",)),
    ("youtube", "the YouTube search box", ("search",)), ("youtube", "the first video's thumbnail", ("v_lofi",)),
    ("youtube", "the video about fresh pasta", ("v_pasta",)), ("youtube", "the Starship launch replay video", ("v_rocket",)),
    ("youtube", "the guitar chords video", ("v_guitar",)),
    ("bios", "the Boot tab in the top menu", ("boot_tab",)), ("bios", "the Save & Exit tab", ("exit_tab",)),
    ("bios", "the Security tab", ("security_tab",)),
    ("phone", "the WhatsApp app icon", ("whatsapp",)), ("phone", "the Camera app icon", ("camera",)),
    ("phone", "the Gmail app icon", ("gmail",)), ("phone", "the Settings app icon", ("settings",)),
    ("blender", "the Render menu in the top bar", ("render_menu",)),
    ("blender", "the Scripting workspace tab", ("scripting_ws",)), ("blender", "the Modeling workspace tab", ("modeling_ws",)),
    ("blender", "the Camera entry in the outliner", ("outliner_camera",)),
    ("blender", "the Cube entry in the outliner", ("outliner_cube",)),
]
GROUND_PROMPT = ('Find {target}. Reply with JSON {{"found": true|false, "x": 0-1000, "y": 0-1000}}: the centre of that '
                 "control, x from the left edge and y from the top edge of the image; found false if it isn't visible.")
GROUND_SCHEMA = {"type": "object", "properties": {"found": {"type": "boolean"}, "x": {"type": "number"},
                                                  "y": {"type": "number"}}, "required": ["found"]}

QUOTED = r'"[^"]*{}[^"]*"'
# (start, spoken request, patterns the lines must hold, patterns they must not, True if it must ask)
REWRITES = [
    ("inbox", "um the one from HR, open that", [r"benefit|hr"], [r"\bsend\b|delete"], False),
    ("inbox", "what's janet saying about churn, pull it up", [r"janet"], [r"\bsend\b"], False),
    ("inbox", "tell john I'll get it to him by six", [r"john", QUOTED.format("(6|six)")], [r"\bsend\b"], False),
    ("alex", "write back to alex, tacos sound great, see you at noon", [r"repl", QUOTED.format("taco")],
     [r"\bsend\b"], False),
    ("inbox", "put on something chill I can study to", [r"youtube", r"lofi|study|chill|relax"], [], False),
    ("inbox", "find that starship launch video and play it", [r"youtube", r"starship"], [], False),
    ("inbox", "search youtube for pasta recipes and play the first one", [r"youtube", r"pasta", r"first"], [], False),
    ("lofi", "get me back to my mail and pull up the benefits thing", [r"mail|outlook|inbox", r"benefit"], [], False),
    ("inbox", "no wait, the other one", [], [], True),
    ("inbox", "draft up a quick reply to john saying the report is attached", [r"repl", QUOTED.format("report")],
     [r"\bsend\b"], False),
    ("lofi", "turn this off and find me a cat video instead", [r"cat"], [], False),
    ("inbox", "go to my sent folder and then back to the inbox", [r"sent", r"inbox"], [], False),
]
STARTS = {"inbox": vague.INBOX, "alex": vague.at("email:alex", ["inbox"]),
          "lofi": vague.at("watch:lofi", ["inbox", "youtube"], watched="lofi")}


def render() -> dict[str, dict]:
    """The six screens at 1x and on a 2560 px canvas (`<name>@small`), and the rewrite start screens."""
    from playwright.sync_api import sync_playwright

    SCREENS.mkdir(parents=True, exist_ok=True)
    boxes: dict[str, dict] = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for name, (html, viewport) in fixtures.SCREENS.items():
            for tag, scale in (("", 1), ("@small", 2)):
                if scale > 1 and name == "phone":
                    continue  # a phone's screen doesn't grow
                w, h = viewport["width"] * scale, viewport["height"] * scale
                page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
                page.set_content(html)
                page.screenshot(path=str(SCREENS / f"{name}{tag}.jpg"), type="jpeg", quality=90)
                boxes[name + tag] = {el.get_attribute("data-gt"): [b["x"] / w, b["y"] / h, (b["x"] + b["width"]) / w,
                                                                   (b["y"] + b["height"]) / h]
                                     for el in page.query_selector_all("[data-gt]") if (b := el.bounding_box())}
                page.close()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.set_content(vague.HTML)
        for name, state in STARTS.items():
            page.evaluate("s => load(s)", state)
            page.screenshot(path=str(SCREENS / f"start-{name}.jpg"), type="jpeg", quality=90)
        browser.close()
    (SCREENS / "boxes.json").write_text(json.dumps(boxes, indent=1))
    return boxes


def score_rewrite(data: dict, must: list[str], never: list[str], asks: bool) -> str:
    lines = " \n".join(str(x) for x in data.get("lines") or [])
    if asks:
        return "ok" if data.get("ask") and not lines else "no_question"
    if not lines:
        return "asked" if data.get("ask") else "empty"
    text = lines.lower()
    if any(re.search(p, text) for p in never):
        return "unsafe"
    return "ok" if all(re.search(p, text) for p in must) else "missing"


def inside(point: tuple[float, float] | None, ids: tuple[str, ...], boxes: dict) -> bool:
    return point is not None and any(b[0] <= point[0] <= b[2] and b[1] <= point[1] <= b[3]
                                     for b in (boxes[i] for i in ids))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="render the screens and stop (no API calls)")
    parser.add_argument("--report", action="store_true", help="report saved rows (no API calls)")
    parser.add_argument("--tasks", nargs="+", default=["ground", "rewrite", "plan"])
    parser.add_argument("--models", nargs="*", help="OpenRouter ids (effort off) instead of CANDIDATES")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--pilot", action="store_true",
                        help="one repeat of a few items per task, to check models answer and what calls cost")
    args = parser.parse_args()
    if args.report:
        report(json.loads((OUT / "rows.json").read_text()))
        return
    boxes = render()
    if args.render:
        print(f"rendered {len(boxes)} screens to {SCREENS}")
        return
    load_dotenv()
    cfg = Config.load()
    key = os.environ.get("OPENROUTER_API_KEY", "")
    clients: dict[str, object] = {}
    if key:
        for model, effort in [(m, "") for m in args.models] if args.models else CANDIDATES:
            clients[f"{model} [{effort or 'off'}]"] = OpenRouterJSON(model, key, effort, fastest=True)
    if cfg.gemini_api_key:
        for model, thinking in GOOGLE:
            clients[f"{model} (Google API) [{thinking or 'default'}]"] = DirectGemini(cfg, model, thinking)
    jev = JevVision(cfg.typesafe_api_key, CancelToken(), OUT / "jev.jsonl", cfg.jev_model, 30) \
        if cfg.typesafe_api_key else None
    if not clients and jev is None:
        raise SystemExit("No keys: fill OPENROUTER_API_KEY, GEMINI_API_KEY or TYPESAFE_API_KEY in .env")
    images = {p.stem: p.read_bytes() for p in SCREENS.glob("*.jpg")}

    ground = [(s + tag, t, ids) for s, t, ids in GROUND for tag in ("", "@small") if s + tag in boxes]
    rewrites, plans = REWRITES, requests()
    if args.pilot:  # a small target and a big one, a composed reply and a question, two plans
        ground, rewrites, plans, args.repeats = ground[:2] + ground[-2:], [REWRITES[2], REWRITES[8]], plans[:2], 1
    jobs: list[tuple] = []
    for r in range(args.repeats):
        if "ground" in args.tasks:
            jobs += [(name, "ground", item, r) for name in clients for item in ground]
            jobs += [(JEV, "ground", item, r) for item in ground] if jev else []
        if "rewrite" in args.tasks:
            jobs += [(name, "rewrite", item, r) for name in clients for item in rewrites]
        if "plan" in args.tasks and jev:
            jobs += [(name, "plan", item, r) for name in clients for item in plans]
    random.Random(9).shuffle(jobs)  # interleaved: API latency drifts 2x over an hour

    def run(job: tuple) -> dict:
        name, task, item, repeat = job
        row = {"model": name, "task": task, "repeat": repeat}
        try:
            if name == JEV:
                screen, target, ids = item
                started = time.monotonic()
                point = jev.locate(images[screen], target)
                point = (refine(jev, images[screen], target, point) or point) if point else None
                return {**row, "screen": screen, "item": target, "total": time.monotonic() - started,
                        "outcome": "hit" if inside(point, ids, boxes[screen]) else "miss" if point else "not_found"}
            client = clients[name]
            if task == "ground":
                screen, target, ids = item
                data, timing = client.call(shrink(images[screen]), GROUND_PROMPT.format(target=target),
                                           "You locate controls in screenshots.", GROUND_SCHEMA)
                point = (float(data["x"]) / 1000, float(data["y"]) / 1000) if data.get("found") and "x" in data else None
                outcome = "hit" if inside(point, ids, boxes[screen]) else "miss" if point else "not_found"
                return {**row, "screen": screen, "item": target, "outcome": outcome, **timing}
            if task == "rewrite":
                start, said, must, never, asks = item
                prompt = f"Goal: {said}\nDevice OS: linux; input: mouse\nExecuted steps: (none)"
                data, timing = client.call(shrink(images[f"start-{start}"]), prompt, rewrite.PROMPT, rewrite.SCHEMA)
                return {**row, "item": said, "outcome": score_rewrite(data, must, never, asks), "reply": data, **timing}
            screen, text, rubric = item
            data, timing = client.call(shrink(images[screen]), f"Goal: {text}" + NO_FALLBACK, PROMPT, SCHEMA)
            action = first_action(data)
            point = None
            if action and action.get("type") in POINTER and action.get("target"):
                point = jev.locate(images[screen], action["target"])
                point = (refine(jev, images[screen], action["target"], point) or point) if point else None
            return {**row, "item": text[:60], "outcome": judge(data, action, point, boxes[screen], rubric),
                    "action": action, **timing}
        except Exception as exc:  # noqa: BLE001 - every failure is a row in the report
            return {**row, "outcome": "error", "error": f"{type(exc).__name__}: {exc}"[:200]}

    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(run, jobs))
    if jev:
        jev.close()
    (OUT / ("pilot.json" if args.pilot else "rows.json")).write_text(json.dumps(rows, indent=1))
    report(rows)


def _pct(values: list[float], q: float) -> str:
    if not values:
        return "–"
    values = sorted(values)
    return f"{values[min(len(values) - 1, int(len(values) * q))]:.2f}"


def report(rows: list[dict]) -> None:
    good = {"ground": {"hit"}, "rewrite": {"ok"}, "plan": {"ok"}}
    print("| Model | Ground 1x | Ground small | Rewrite ok / unsafe | Plan ok / wrong | errors | p50 s | p90 s | "
          "TTFT p50 | $ / 100 calls |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name in dict.fromkeys(r["model"] for r in rows):
        mine = [r for r in rows if r["model"] == name]

        def rate(task: str, small: bool | None = None) -> str:
            sel = [r for r in mine if r["task"] == task and r["outcome"] != "error"
                   and (small is None or r.get("screen", "").endswith("@small") == small)]
            return f"{sum(r['outcome'] in good[task] for r in sel)}/{len(sel)}" if sel else "–"

        rewrites = [r for r in mine if r["task"] == "rewrite"]
        plans = [r for r in mine if r["task"] == "plan"]
        totals = [r["total"] for r in mine if r.get("total")]
        ttft = [r["ttft"] for r in mine if r.get("ttft")]
        costs = [r["cost"] for r in mine if r.get("cost") is not None]
        print(f"| {name} | {rate('ground', False)} | {rate('ground', True)} | "
              f"{rate('rewrite')} / {sum(r['outcome'] == 'unsafe' for r in rewrites)} | "
              f"{rate('plan')} / {sum(r['outcome'] in ('wrong', 'bad') for r in plans)} | "
              f"{sum(r['outcome'] == 'error' for r in mine)} | {_pct(totals, .5)} | {_pct(totals, .9)} | "
              f"{_pct(ttft, .5)} | {f'{100 * statistics.mean(costs):.3f}' if costs else '–'} |")
    for r in [r for r in rows if r["outcome"] == "error"][:10]:
        print("  error:", r["model"], r["task"], r["error"])


if __name__ == "__main__":
    main()
