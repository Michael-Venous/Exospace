"""Study 11: an option scan before optimizing one path. Where does a model call's time go, and which
fast models or inputs could make speech-to-command, planning or grounding answer in well under a second?

  floor    a tiny JSON reply with no image, and one about a screenshot: network + queue + first token,
           per model and provider, OpenRouter vs Google's own API for the same model; Jev's floor too
  command  the 32 triage segments of study 1 as act (with a target) / plan / none, from
             screen  the screenshot (vision models)
             dom     a numbered list of the screen's labelled controls with their text and position, taken
                     from the page itself: an oracle inventory, the BEST case for any element-list design
             ocr     the same list built by CPU OCR from the pixels (RapidOCR): what a parser would give
  plan     study 3's 18 planning requests: the first action, choosing an element from the dom list
  ground   study 9's 100 described controls, placed by OCR text matching alone (no model)
Scored like studies 1, 3 and 9 (DOM boxes). Element lists are synthetic-page inventories, not real-screen
perception: a parser on a real screen sees less than `dom` does.

  .venv/bin/pip install rapidocr-onnxruntime                     # the CPU OCR (bench only, not a project dependency)
  PYTHONPATH=. .venv/bin/python -m bench.arch.option_scan [--tasks floor command plan ground] [--repeats 2]
  PYTHONPATH=. .venv/bin/python -m bench.arch.option_scan --report
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from bench.arch import fixtures
from bench.arch.cases import CASES
from bench.arch.first_step_study import first_action, judge, requests
from bench.arch.model_matrix import GROUND, SCREENS as MATRIX_SCREENS, inside
from bench.arch.openrouter_study import DirectGemini, OpenRouterJSON
from bench.arch.triage_study import score
from brain.agent.jev import JevVision
from brain.cancel import CancelToken
from brain.config import Config
from brain.imaging import shrink

OUT = Path("work/arch/options")
# (label, OpenRouter id, provider pinned or "" for the fastest, reasoning effort, takes images)
CANDIDATES = [
    ("Mercury 3 (Inception)", "inception/mercury-3", "Inception", "", False),
    ("gpt-oss-120b (Cerebras)", "openai/gpt-oss-120b", "Cerebras", "low", False),
    ("gpt-oss-120b (Groq)", "openai/gpt-oss-120b", "Groq", "low", False),
    ("gpt-oss-20b (Groq)", "openai/gpt-oss-20b", "Groq", "low", False),
    ("Llama 3.1 8B (Groq)", "meta-llama/llama-3.1-8b-instruct", "Groq", "", False),
    ("Llama 3.3 70B (Groq)", "meta-llama/llama-3.3-70b-instruct", "Groq", "", False),
    ("Qwen 3.8 27B (Cerebras)", "qwen/qwen3.8-27b", "Cerebras", "", True),
    ("Gemma 4 31B (SambaNova)", "google/gemma-4-31b-it", "SambaNova", "", True),
    ("DeepSeek V4.1 Flash (fastest)", "deepseek/deepseek-v4.1-flash-1017", "", "", True),
    ("Gemini 3.5 Flash-Lite (OpenRouter)", "google/gemini-3.5-flash-lite", "", "minimal", True),
    ("Gemini 3.8 Flash (OpenRouter)", "google/gemini-3.8-flash", "", "low", True),
]
GOOGLE = [("Gemini 3.5 Flash-Lite (Google API)", "gemini-3.5-flash-lite", None),
          ("Gemini 3.8 Flash (Google API)", "gemini-3.8-flash", "low")]
JEV = "Jev (one noul question)"

FLOOR_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
SCREEN_SCHEMA = {"type": "object", "properties": {"app": {"type": "string"}}, "required": ["app"]}
COMMAND_PROMPT = """You control a computer for a user who is speaking to you. {screen}
Classify the user's latest utterance:
- act: ONE visible control should be clicked now and that alone satisfies the request: give its {target}.
- plan: needs typing, keys, several steps, content, an answer, a destructive action, or the target is
  ambiguous or not visible.
- none: a remark or context; no action requested.
Utterance: {utterance}"""
COMMAND_SCHEMA = {"type": "object", "properties": {
    "kind": {"type": "string", "enum": ["act", "plan", "none"]}, "element": {"type": "string"},
    "x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["kind"]}
PLAN_PROMPT = """You control a computer for a user. The screen shows these elements (id: text @ x,y of 1000):
{elements}
Goal: {goal}
Give only the FIRST action toward the goal: type is click, double_click, right_click, type_text (text),
hotkey (keys, e.g. ["CTRL","L"]), scroll (dy), ask_user (question) or fail. Clicks name an element id."""
PLAN_SCHEMA = {"type": "object", "properties": {
    "type": {"type": "string"}, "element": {"type": "string"}, "text": {"type": "string"},
    "keys": {"type": "array", "items": {"type": "string"}}, "dy": {"type": "integer"},
    "question": {"type": "string"}}, "required": ["type"]}
STOP = {"the", "a", "an", "of", "in", "on", "to", "and", "icon", "button", "tab", "menu", "app", "entry", "video",
        "top", "bar", "left", "pane", "list", "message", "desktop", "file", "about", "first", "s"}


def inventories() -> tuple[dict, dict, dict]:
    """Per screen: the dom list [(id, text, (x, y))], the ocr list the same way, and gt boxes."""
    from playwright.sync_api import sync_playwright
    from rapidocr_onnxruntime import RapidOCR

    dom, boxes = {}, json.loads((MATRIX_SCREENS / "boxes.json").read_text())
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for name, (html, viewport) in fixtures.SCREENS.items():
            page = browser.new_page(viewport=viewport)
            page.set_content(html)
            items = page.evaluate("""() => [...document.querySelectorAll('[data-gt]')].map(e => {
                const b = e.getBoundingClientRect();
                return [e.innerText.replace(/\\s+/g, ' ').trim().slice(0, 60), (b.x + b.width / 2) / innerWidth,
                        (b.y + b.height / 2) / innerHeight]})""")
            order = list(range(len(items)))
            random.Random(name).shuffle(order)  # ids say nothing about which control is which
            dom[name] = [(f"e{i + 1}", items[j][0] or "(no text)", (items[j][1], items[j][2])) for i, j in enumerate(order)]
            page.close()
        browser.close()
    ocr, engine = {}, RapidOCR()
    for name in fixtures.SCREENS:
        path = MATRIX_SCREENS / f"{name}.jpg"
        import cv2
        h, w = cv2.imread(str(path)).shape[:2]
        found, _ = engine(str(path), use_cls=False)
        ocr[name] = [(f"t{i + 1}", text, ((min(p[0] for p in quad) + max(p[0] for p in quad)) / 2 / w,
                                           (min(p[1] for p in quad) + max(p[1] for p in quad)) / 2 / h))
                     for i, (quad, text, _) in enumerate(found or [])]
    return dom, ocr, boxes


def listing(elements: list) -> str:
    return "\n".join(f"{i}: {text} @ {round(x * 1000)},{round(y * 1000)}" for i, text, (x, y) in elements)


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP}


def ocr_point(target: str, elements: list) -> tuple[float, float] | None:
    """The OCR line sharing most words with the description; None on no overlap or a tie."""
    want = _words(target)
    scored = sorted(((len(want & _words(text)), xy) for _, text, xy in elements), key=lambda s: -s[0])
    if not scored or scored[0][0] == 0 or (len(scored) > 1 and scored[1][0] == scored[0][0]):
        return None
    return scored[0][1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["floor", "command", "plan", "ground"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        report(json.loads((OUT / "rows.json").read_text()))
        return
    load_dotenv()
    cfg = Config.load()
    OUT.mkdir(parents=True, exist_ok=True)
    dom, ocr, boxes = inventories()
    images = {name: (MATRIX_SCREENS / f"{name}.jpg").read_bytes() for name in fixtures.SCREENS}
    key = os.environ["OPENROUTER_API_KEY"]
    clients = {label: (OpenRouterJSON(model, key, effort, only=only, schema=False), vision)
               for label, model, only, effort, vision in CANDIDATES}
    clients |= {label: (DirectGemini(cfg, model, thinking), True) for label, model, thinking in GOOGLE}
    jev = JevVision(cfg.typesafe_api_key, CancelToken(), OUT / "jev.jsonl", cfg.jev_model, 30)

    jobs: list[tuple] = []
    for r in range(args.repeats):
        for name, (_, vision) in clients.items():
            if "floor" in args.tasks:
                jobs += [(name, "floor-text", None, r), *([(name, "floor-screen", None, r)] if vision else [])]
            if "command" in args.tasks:
                sources = ["dom", "ocr", *(["screen"] if vision else [])]
                jobs += [(name, f"command-{src}", case, r) for src in sources for case in CASES]
            if "plan" in args.tasks:
                jobs += [(name, "plan-dom", item, r) for item in requests()]
        if "floor" in args.tasks:
            jobs += [(JEV, "floor-screen", None, r)]
    random.Random(11).shuffle(jobs)

    def run(job: tuple) -> dict:
        row = _run(job)
        if row["outcome"] == "error" and re.search(r"HTTP (429|5\d\d)|ReadError|Timeout|ServerError", row["error"]):
            time.sleep(2.0)  # one retry, as in the matrix
            row = {**_run(job), "retried": True}
        return row

    def _run(job: tuple) -> dict:
        name, task, item, repeat = job
        row = {"model": name, "task": task, "repeat": repeat}
        try:
            started = time.monotonic()
            if name == JEV:
                jev.evaluate(images["inbox"], {"q": {"type": "noul", "instructions": "Is an email app open in `screen`?"}})
                return {**row, "outcome": "ok", "total": time.monotonic() - started}
            client, _ = clients[name]
            if task == "floor-text":
                _, timing = client.call(None, 'Reply {"ok": true}.', "You answer in JSON.", FLOOR_SCHEMA)
                return {**row, "outcome": "ok", **timing, "total": timing["total"]}
            if task == "floor-screen":
                _, timing = client.call(shrink(images["inbox"]), "Which app is in front?", "You answer in JSON.",
                                        SCREEN_SCHEMA)
                return {**row, "outcome": "ok", **timing}
            if task.startswith("command"):
                src, case = task.split("-")[1], item
                elements = {"dom": dom, "ocr": ocr}.get(src, {}).get(case.screen, [])
                screen = ("The screen shows these elements (id: text @ x,y of 1000):\n" + listing(elements)
                          if elements else "The screenshot shows the screen.")
                target = "element id" if src != "screen" else "x,y (0-1000) centre"
                data, timing = client.call(shrink(images[case.screen]) if src == "screen" else None,
                                           COMMAND_PROMPT.format(screen=screen, target=target, utterance=case.utterance),
                                           "You control a device for a user.", COMMAND_SCHEMA)
                kind = data.get("kind") if data.get("kind") in ("act", "plan", "none") else "plan"
                if src == "screen":
                    point = (data["x"] / 1000, data["y"] / 1000) if "x" in data and "y" in data else None
                else:
                    point = dict((i, xy) for i, _, xy in elements).get(str(data.get("element", "")))
                decision = kind if kind != "act" or point else "plan"
                return {**row, "item": case.utterance, "outcome": score(decision, point, case, boxes), **timing}
            screen, goal, rubric = item
            data, timing = client.call(None, PLAN_PROMPT.format(elements=listing(dom[screen]), goal=goal),
                                       "You control a device for a user.", PLAN_SCHEMA)
            point = dict((i, xy) for i, _, xy in dom[screen]).get(str(data.get("element", "")))
            action = {k: v for k, v in data.items() if k != "element"}
            if action.get("type") in ("click", "double_click", "right_click"):
                action["target"] = data.get("element")
            plan = {"steps": [{"actions": [action]}]}
            return {**row, "item": goal[:60], "outcome": judge(plan, first_action(plan), point, boxes[screen], rubric),
                    **timing}
        except Exception as exc:  # noqa: BLE001 - every failure is a row
            return {**row, "outcome": "error", "error": f"{type(exc).__name__}: {exc}"[:200]}

    rows: list[dict] = []
    if "ground" in args.tasks:  # no model: OCR text matching, timed per screen
        for screen, target, ids in GROUND:
            point = ocr_point(target, ocr[screen])
            rows.append({"model": "OCR text match (CPU)", "task": "ground", "item": target,
                         "outcome": "hit" if inside(point, ids, boxes[screen]) else "miss" if point else "not_found"})
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows += list(pool.map(run, jobs))
    jev.close()
    (OUT / "rows.json").write_text(json.dumps(rows, indent=1))
    report(rows)


def _p(values: list[float], q: float) -> str:
    values = sorted(v for v in values if v is not None)
    return f"{values[min(len(values) - 1, int(len(values) * q))]:.2f}" if values else "–"


def report(rows: list[dict]) -> None:
    good = {"command": "correct", "plan": "ok", "ground": "hit"}
    unsafe = {"command": ("false_execute", "wrong_target"), "plan": ("wrong", "bad"), "ground": ("miss",)}
    print("| Model | floor text p50 (TTFT) | floor screen p50 (TTFT) | command dom ok / unsafe | command ocr | "
          "command screen | command p50 s | plan ok / wrong | errors |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name in dict.fromkeys(r["model"] for r in rows):
        mine = [r for r in rows if r["model"] == name]

        def cell(task: str) -> str:
            sel = [r for r in mine if r["task"] == task and r["outcome"] != "error"]
            if not sel:
                return "–"
            kind = task.split("-")[0]
            return f"{sum(r['outcome'] == good[kind] for r in sel)}/{len(sel)} / {sum(r['outcome'] in unsafe[kind] for r in sel)}"

        def floor(task: str) -> str:
            sel = [r for r in mine if r["task"] == task and r["outcome"] == "ok"]
            return f"{_p([r['total'] for r in sel], .5)} ({_p([r.get('ttft') for r in sel], .5)})" if sel else "–"

        cmd = [r["total"] for r in mine if r["task"].startswith("command") and r.get("total")]
        print(f"| {name} | {floor('floor-text')} | {floor('floor-screen')} | {cell('command-dom')} | "
              f"{cell('command-ocr')} | {cell('command-screen')} | {_p(cmd, .5)} | {cell('plan-dom')} | "
              f"{sum(r['outcome'] == 'error' for r in mine)} |")
    ground = [r for r in rows if r["task"] == "ground"]
    if ground:
        print(f"\nOCR text match: {sum(r['outcome'] == 'hit' for r in ground)}/{len(ground)} hit, "
              f"{sum(r['outcome'] == 'miss' for r in ground)} wrong, {sum(r['outcome'] == 'not_found' for r in ground)} refused")
    for r in [r for r in rows if r["outcome"] == "error"][:12]:
        print("  error:", r["model"], r["task"], r.get("error"))
    providers = {}
    for r in rows:
        if r.get("provider"):
            providers.setdefault(r["model"], set()).add(r["provider"])
    print("\nserved by:", {k: sorted(v) for k, v in providers.items()})


if __name__ == "__main__":
    main()
