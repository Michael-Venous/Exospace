"""Study 17: which fast multimodal model can point at a named target as well as Jev, or behind it?

Two sets, every candidate interleaved:
  laptop     52 targets on real screenshots of the test laptop (KDE Plasma, Chrome, Dolphin, Kate, System
             Settings, Blender; `work/arch/clicks/screens/`, copied from `work/arch/laptop/`). 24 of the 46
             targets on screen are named as the brain named them in those runs, the rest are written the same way
             for list rows, small icons, tabs, menu and dropdown options and text links. Truth boxes were labelled
             by hand on these images. 6 targets are not on their screen (4 of them named by the brain on another
             screen): the right answer there is a refusal.
  synthetic  study 9's 27 described controls at 1x and on the 2560 px canvas (`model_matrix.GROUND`).
Jev is measured as the queue agent runs it (`queue_agent.Executor._find`: locate, then zooms). Each OpenRouter
model runs on one pinned provider with reasoning off or minimal, chosen by `--probe` as the fastest that answers.

  PYTHONPATH=. .venv/bin/python -m bench.arch.click_check --boxes     # draw truth boxes for review, no API calls
  PYTHONPATH=. .venv/bin/python -m bench.arch.click_check --probe     # fastest working provider per model
  PYTHONPATH=. .venv/bin/python -m bench.arch.click_check             # both sets x 2 repeats
  PYTHONPATH=. .venv/bin/python -m bench.arch.click_check --report
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import httpx
import numpy as np

from bench.arch.model_matrix import GROUND, GROUND_PROMPT, SCREENS as MATRIX_SCREENS
from bench.arch.openrouter_study import OPENROUTER, parse_json
from bench.arch.queue_agent import Executor
from brain.agent.jev import JevVision
from brain.agent.triage import GATE
from brain.cancel import CancelToken
from brain.imaging import shrink

OUT = Path("work/arch/clicks")
SCREENS = OUT / "screens"
JEV = "Jev (_find: locate + zooms)"
RETRIES_S = (2.0, 4.0)  # waits before retrying a call that hit a 429, a 5xx or a dropped connection
Box = tuple[int, int, int, int]  # left, top, right, bottom in pixels of the 1280 x 800 screenshot

# (screenshot, target as the brain names it, truth boxes (none: not on screen), kind, named by the brain in that run)
LAPTOP: list[tuple[str, str, tuple[Box, ...], str, bool]] = [
    ("qwen27b-6-0/call-10", "Documents", ((208, 218, 292, 236), (298, 158, 370, 176), (388, 158, 460, 176),
                                          (478, 158, 550, 176)), "list", True),
    ("qwen27b-6-0/call-10", "exo_invoice_0519.pdf", ((353, 517, 1046, 558),), "list", True),
    ("qwen27b-6-0/call-10", "the Modified column header", ((949, 179, 1044, 197),), "button", True),
    ("qwen27b-6-0/call-10", "the Split button", ((978, 132, 1018, 150),), "icon", False),
    ("qwen27b-6-0/call-10", "the Screenshots tab", ((655, 157, 730, 177),), "tab", False),
    ("haiku-6-0/call-11", "exo_invoice_0412.pdf", ((353, 562, 1046, 603),), "list", True),
    ("haiku-6-0/call-11", "the back arrow button", ((215, 132, 235, 150),), "icon", False),
    ("deepseek-8-0/call-45", "Image", ((642, 332, 805, 351),), "option", True),
    ("deepseek-8-0/call-45", "Slideshow", ((642, 352, 805, 371),), "option", False),
    ("deepseek-8-0/call-47", "BlueLady", ((804, 342, 926, 440),), "list", True),
    ("deepseek-8-0/call-47", "Apply", ((891, 599, 945, 619),), "button", True),
    ("deepseek-8-0/call-47", "the Positioning dropdown", ((664, 253, 790, 273),), "button", False),
    ("deepseek-8-0/call-43", "the Wallpaper type dropdown", ((640, 228, 790, 248),), "button", True),
    ("deepseek-8-0/call-43", "BlueLady", (), "absent", True),
    ("deepseek-9-0/call-8", "the menu button (three dots) at the top right of the browser", ((921, 53, 937, 72),),
     "icon", True),
    ("deepseek-9-0/call-8", "Talk", ((388, 172, 406, 186),), "tab", False),
    ("deepseek-9-0/call-8", "Charles Babbage", ((351, 376, 434, 390),), "link", False),
    ("deepseek-9-0/call-8", "the YouTube tab", (), "absent", True),
    ("deepseek-9-0/call-10", "Print…", ((680, 381, 940, 400),), "option", True),
    ("deepseek-9-0/call-10", "the zoom-in plus button in the menu", ((880, 349, 897, 367),), "icon", False),
    ("deepseek-9-0/call-15", "Save", ((867, 476, 912, 500),), "button", True),
    ("deepseek-9-0/call-15", "the Destination dropdown", ((784, 125, 912, 149),), "button", False),
    ("deepseek-9-0/call-15", "Overwrite", (), "absent", True),
    ("deepseek-7-0/call-13", "groceries.txt @ ~/Desktop", ((535, 284, 785, 301),), "list", True),
    ("deepseek-7-0/call-13", "New File", ((432, 282, 519, 304),), "button", True),
    ("deepseek-7-0/call-13", "Read Handbook", ((812, 314, 879, 327),), "link", False),
    ("deepseek-7-0/call-19", "OK", ((723, 421, 778, 441),), "button", True),
    ("qwen27b-8-0/call-8", "Display Configuration", ((455, 27, 825, 48),), "list", True),
    ("qwen27b-8-0/call-8", "errors.log", ((455, 322, 825, 343),), "list", False),
    ("qwen27b-8-0/call-10", "the Resolution dropdown", ((673, 230, 816, 250),), "button", False),
    ("qwen27b-8-0/call-10", "Night Light", ((333, 250, 508, 270),), "list", False),
    ("qwen27b-8-0/call-10", "the info icon next to Refresh rate", ((719, 311, 734, 326),), "icon", False),
    ("qwen27b-8-0/call-10", "119.93 Hz", (), "absent", False),
    ("deepseek-4-0/call-16", "Five Wonders of our Universe | Universe | BBC Earth Science",
     ((400, 322, 654, 465), (663, 322, 885, 357)), "link", True),
    ("deepseek-4-0/call-16", "the Filters button", ((858, 124, 892, 144),), "button", False),
    ("deepseek-4-0/call-16", "Strip the Cosmos | Free Documentary Space", (), "absent", True),
    ("deepseek-4-0/call-18", "Strip the Cosmos | Free Documentary Space", ((391, 716, 683, 734), (391, 740, 491, 780)),
     "link", True),
    ("deepseek-10-0/call-11", "the Eiffel Tower result title", ((391, 214, 468, 232),), "link", False),
    ("deepseek-10-0/call-11", "the X that removes the first suggestion", ((818, 82, 834, 99),), "icon", False),
    ("deepseek-11-0/call-6", "General", ((495, 428, 635, 442),), "list", True),
    ("deepseek-11-0/call-6", "the Render menu in the top bar", ((389, 223, 421, 238),), "tab", False),
    ("deepseek-11-0/call-6", "the Modeling workspace tab", ((536, 223, 577, 238),), "tab", False),
    ("deepseek-11-0/call-5", "Blender", ((367, 440, 824, 466),), "list", True),
    ("qwen27b-3-0/call-1", "the 'x' close button on the YouTube tab at the top of the browser", ((649, 27, 663, 42),),
     "icon", True),
    ("qwen27b-3-0/call-1", "the new-tab plus button at the top of the browser", ((823, 26, 845, 43),), "icon", True),
    ("qwen27b-3-0/call-1", "the GitHub tab", ((668, 26, 804, 44),), "tab", False),
    ("qwen27b-4-0/call-5", "5 Incredible Explorations of our Mind-Blowing Universe",
     ((503, 217, 769, 231), (391, 216, 491, 273)), "link", True),
    ("qwen27b-4-0/call-5", "the Images tab", ((586, 148, 620, 162),), "tab", False),
    ("qwen27b-10-0/call-7", "the wikipedia suggestion at the top of the search dropdown", ((415, 368, 865, 386),),
     "list", True),
    ("qwen27b-10-0/call-7", "the YouTube tab", (), "absent", False),
    ("deepseek-4-0/call-6", "the search box", ((440, 331, 740, 363),), "button", True),
    ("deepseek-4-0/call-6", "the Google apps grid icon", ((854, 90, 870, 106),), "icon", False),
]


@dataclass(frozen=True)
class Item:
    set: str
    screen: str  # key into the image table
    target: str
    boxes: tuple[tuple[float, float, float, float], ...]  # 0-1 of the image; empty: not on screen
    kind: str


@dataclass(frozen=True)
class Candidate:
    label: str
    model: str
    providers: tuple[str, ...]  # probed in turn; the run pins the fastest that answers
    effort: str = ""  # "" = reasoning off
    native: str = "json"  # "json": GROUND_PROMPT's 0-1000 x/y; "pixel": x/y in the image's pixels;
    # "tars": UI-TARS click(point) in resized pixels
    schema: bool = True  # structured output; OpenRouter drops a pinned provider that lacks it (Cerebras)


CANDIDATES = [
    Candidate("Qwen 3.8 27B (Cerebras)", "qwen/qwen3.8-27b", ("cerebras/fp16",), schema=False),
    Candidate("Qwen 3.8 27B (ModelRun)", "qwen/qwen3.8-27b", ("modelrun/fp4",)),
    Candidate("Qwen 3.8 Flash", "qwen/qwen3.8-flash", ("alibaba/us-east-1", "alibaba")),
    Candidate("MiMo V2.6 Flash", "xiaomi/mimo-v2.6-flash",
              ("xiaomi/fp8", "deepinfra/fp8", "gmicloud/bf16", "novita/fp8", "io-net/fp8")),
    Candidate("DeepSeek V4.1 Flash 1017", "deepseek/deepseek-v4.1-flash-1017",
              ("deepseek", "fireworks", "together", "deepinfra/fp8/priority", "inference-net/fp8")),
    Candidate("Gemini 3.8 Flash (minimal)", "google/gemini-3.8-flash",
              ("google-ai-studio", "google-vertex/global", "google-ai-studio/priority"), "minimal"),
    Candidate("Gemini 3.5 Flash-Lite (minimal)", "google/gemini-3.5-flash-lite",
              ("google-ai-studio", "google-vertex/global", "google-ai-studio/priority"), "minimal"),
    Candidate("Perceptron Mk1.5", "perceptron/perceptron-mk1.5", ("perceptron",)),
    Candidate("Gemma 4 31B", "google/gemma-4-31b-it",
              ("sambanova", "deepinfra/turbo", "friendli", "crusoe/bf16", "parasail/fp8")),
    # Advertised for GUI or computer use on OpenRouter (2026-10-21); no Holo or OpenCUA model is listed there.
    Candidate("UI-TARS 1.5 7B", "bytedance/ui-tars-1.5-7b", ("parasail/bf16",), native="tars"),
    Candidate("Qwen 3.7 Flash", "qwen/qwen3.7-flash", ("alibaba/us-east-1", "alibaba")),
    Candidate("Nemotron 3.5 Super VL", "nvidia/nemotron-3.5-super-vl",
              ("together", "crusoe/bf16", "deepinfra/fp4", "coreweave/nvfp4")),
    # Other Qwen vision models on OpenRouter (2026-10-21): the larger 3.8 ones and the earlier MoE line.
    Candidate("Qwen 3.8 Max", "qwen/qwen3.8-max-0902", ("alibaba/us-east-1", "alibaba")),
    Candidate("Qwen 3.8 Omni Flash", "qwen/qwen3.8-omni-flash", ("alibaba",)),
    Candidate("Qwen 3.6 27B", "qwen/qwen3.6-27b", ("chutes/fp8", "siliconflow/fp8", "deepinfra/fp8", "alibaba")),
    Candidate("Qwen 3.6 35B-A3B", "qwen/qwen3.6-35b-a3b",
              ("darkbloom/fp4", "akashml/fp8", "deepinfra/fp8", "parasail/fp8", "coreweave/fp8")),
    Candidate("Qwen 3.5 122B-A10B", "qwen/qwen3.5-122b-a10b", ("siliconflow/fp8", "alibaba", "novita/bf16")),
    Candidate("Qwen 3.5 397B-A17B", "qwen/qwen3.5-397b-a17b", ("alibaba", "deepinfra/fp8", "parasail/fp8"),
              schema=False),
    Candidate("Qwen3-VL 235B-A22B", "qwen/qwen3-vl-235b-a22b-instruct", ("deepinfra/fp8", "parasail/fp8")),
    # One candidate per host: does DeepSeek point better on another host (2026-10-21)? And GLM 5.3 Flash, whose
    # hosts all refuse reasoning off. DeepSeek's own and Baseten's endpoints take no images.
    *[Candidate(f"DeepSeek V4.1 Flash @{p}", "deepseek/deepseek-v4.1-flash-1017", (p,))
      for p in ("together", "fireworks", "deepinfra/fp8", "parasail/fp8", "morph/fp8", "makora/fp8")],
    *[Candidate(f"GLM 5.3 Flash @{p}", "z-ai/glm-5.3-flash", (p,), "minimal", schema=p not in ("z-ai/fp8",))
      for p in ("together", "fireworks", "baseten/fp8", "parasail/fast", "deepinfra/fp4", "z-ai/fp8")],
    *[Candidate(f"GLM 5.3 Flash px @{p}", "z-ai/glm-5.3-flash", (p,), "minimal", native="pixel")
      for p in ("together", "deepinfra/fp4")],
    *[Candidate(f"DeepSeek V4.1 Flash px @{p}", "deepseek/deepseek-v4.1-flash-1017", (p,), native="pixel")
      for p in ("together", "morph/fp8")],
]
# x and y required: with only "found" required, DeepSeek V4.1 Flash answered {"found": true} and nothing else.
SCHEMA = {"type": "object", "properties": {"found": {"type": "boolean"}, "x": {"type": "number"},
                                         "y": {"type": "number"}}, "required": ["found", "x", "y"]}
SYSTEM = "You locate controls in screenshots. Reply with a single JSON object only. Its JSON schema: " + json.dumps(SCHEMA)
# GLM 5.3 Flash answers in the image's pixels whatever the prompt asks (its 0-1000 replies were pixels: x 0.80
# and y 0.80 of the truth on 1280 x 800 screens), so it is asked in pixels.
PIXEL_PROMPT = ('Find {target}. Reply with JSON {{"found": true|false, "x": ..., "y": ...}}: the centre of that '
                "control in pixels of this {size} image, x from the left edge and y from the top edge; found false "
                "if it isn't visible.")
# UI-TARS 1.5's own grounding prompt, with a way to say the target isn't there.
TARS_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to \
perform the next action to complete the task.

## Output Format

Action: ...


## Action Space
click(point='<point>x1 y1</point>')
finished(content='not visible') # the target is not on the screen

## User Instruction
Click {target}."""


class Router:
    """One OpenRouter model on one pinned provider; returns the reply text and the call's time, cost and server."""

    def __init__(self, candidate: Candidate, provider: str, key: str) -> None:
        self.candidate, self.provider = candidate, provider
        self.client = httpx.Client(timeout=60, headers={"Authorization": f"Bearer {key}"})

    def call(self, image: bytes, target: str) -> tuple[str, dict]:
        picture = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}}
        if self.candidate.native == "tars":
            messages = [{"role": "user", "content": [{"type": "text", "text": TARS_PROMPT.format(target=target)},
                                                     picture]}]
        else:
            prompt = (PIXEL_PROMPT.format(target=target, size="x".join(map(str, image_size(image))))
                      if self.candidate.native == "pixel" else GROUND_PROMPT.format(target=target))
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": [picture, {"type": "text", "text": prompt}]}]
        body = {"model": self.candidate.model, "messages": messages, "temperature": 0,
                "provider": {"order": [self.provider], "allow_fallbacks": False}, "usage": {"include": True}}
        if self.candidate.schema and self.candidate.native != "tars":  # as study 9
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "point", "schema": SCHEMA}}
        if self.candidate.native != "tars":
            body["reasoning"] = {"effort": self.candidate.effort} if self.candidate.effort else {"enabled": False}
        started = time.monotonic()
        response = self.client.post(OPENROUTER, json=body)
        total = time.monotonic() - started
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"error: {json.dumps(data['error'])[:200]}")
        usage = data.get("usage") or {}
        text = data["choices"][0]["message"].get("content") or ""
        return text, {"total": total, "cost": usage.get("cost"), "served": data.get("provider"),
                      "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")}


def tars_size(width: int, height: int, factor: int = 28) -> tuple[int, int]:
    """The image size UI-TARS 1.5 / Qwen2.5-VL sees after its smart resize (within its pixel limits here)."""
    return max(factor, round(width / factor) * factor), max(factor, round(height / factor) * factor)


class BadReply(ValueError):
    """A reply that is neither a point nor a refusal: no click, and the caller has to ask again."""


def point_of(text: str, candidate: Candidate, size: tuple[int, int]) -> tuple[float, float] | None:
    """The click point (0-1 of the image) in a reply, None for a refusal; BadReply for a reply with neither."""
    if candidate.native == "tars":
        if "finished(" in text and "click(" not in text:
            return None
        match = re.search(r"click\(.*?(\d+(?:\.\d+)?)[,\s]+(\d+(?:\.\d+)?)", text, re.S)
        if not match:
            raise BadReply(f"no action: {text[:120]!r}")
        w, h = tars_size(*size)
        return float(match.group(1)) / w, float(match.group(2)) / h
    try:
        data = parse_json(text)
        if not data.get("found"):
            return None
        scale = size if candidate.native == "pixel" else (1000, 1000)
        return float(data["x"]) / scale[0], float(data["y"]) / scale[1]
    except (ValueError, KeyError, TypeError) as exc:
        raise BadReply(f"{exc}: {text[:120]!r}") from None


def outcome(point: tuple[float, float] | None, item: Item) -> str:
    if point is None:
        return "refused"
    if not item.boxes:
        return "misclick"
    return "hit" if any(b[0] <= point[0] <= b[2] and b[1] <= point[1] <= b[3] for b in item.boxes) else "misclick"


def load_items() -> tuple[list[Item], dict[str, bytes]]:
    images: dict[str, bytes] = {}
    items: list[Item] = []
    for screen, target, boxes, kind, _ in LAPTOP:
        images.setdefault(screen, (SCREENS / f"{screen}.jpg").read_bytes())
        items.append(Item("laptop", screen, target, tuple((b[0] / 1280, b[1] / 800, b[2] / 1280, b[3] / 800)
                                                           for b in boxes), kind))
    boxes = json.loads((MATRIX_SCREENS / "boxes.json").read_text())
    for name, target, ids in GROUND:
        for tag in ("", "@small"):
            if name + tag in boxes:
                screen = "matrix/" + name + tag
                images.setdefault(screen, (MATRIX_SCREENS / f"{name}{tag}.jpg").read_bytes())
                items.append(Item("synthetic", screen, target, tuple(tuple(boxes[name + tag][i]) for i in ids),
                                  "small" if tag else "1x"))
    return items, images


def draw_boxes() -> None:
    """Every laptop screenshot with its truth boxes and target names, for checking the labels by eye."""
    out = OUT / "boxes"
    out.mkdir(parents=True, exist_ok=True)
    for screen in dict.fromkeys(s for s, *_ in LAPTOP):
        image = cv2.imread(str(SCREENS / f"{screen}.jpg"))
        line = 0
        for s, target, boxes, kind, _ in LAPTOP:
            if s != screen:
                continue
            colour = (0, 0, 255) if boxes else (255, 0, 255)
            for b in boxes:
                cv2.rectangle(image, b[:2], b[2:], colour, 1)
            cv2.putText(image, f"{kind}: {target}"[:70], (8, 780 - 16 * line), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
            line += 1
        cv2.imwrite(str(out / f"{screen.replace('/', '_')}.png"), image)
    print(f"drew {len(set(s for s, *_ in LAPTOP))} screens to {out}")


class Jev:
    """`Executor._find` as the queue agent runs it, one JevVision a thread (it reads its own last trace entry)."""

    def __init__(self, key: str) -> None:
        self.key, self.local = key, threading.local()
        self.all: list[JevVision] = []

    def find(self, image: bytes, target: str) -> tuple[tuple[float, float] | None, float]:
        if not hasattr(self.local, "executor"):
            jev = JevVision(self.key, CancelToken(), OUT / "jev.jsonl", min_presence=GATE, min_confidence=GATE)
            self.all.append(jev)
            self.local.executor = SimpleNamespace(jev=jev, trace=[], _jpeg=None)
        executor = self.local.executor
        executor._jpeg = lambda: image
        point = Executor._find(executor, target)
        return point, executor.trace[-1]["elapsed_s"]

    def close(self) -> None:
        for jev in self.all:
            jev.close()


def image_size(jpeg: bytes) -> tuple[int, int]:
    h, w = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape[:2]
    return w, h


def probe(key: str, items: list[Item], images: dict[str, bytes], only: list[str]) -> dict[str, dict]:
    """4 laptop targets x 2 rounds on every listed provider of every model: the fastest with at most one failed call
    (a malformed reply is the model's, not the provider's); a model whose providers all fail more keeps its steadiest."""
    sample = [i for i in items if i.set == "laptop"][::13][:4]
    candidates = [c for c in CANDIDATES if not only or c.label in only]
    jobs = [(c, p, i) for _ in range(2) for c in candidates for p in c.providers for i in sample]
    random.Random(17).shuffle(jobs)

    def run(job: tuple) -> dict:
        candidate, provider, item = job
        image = shrink(images[item.screen])
        row = {"label": candidate.label, "provider": provider}
        try:
            text, timing = Router(candidate, provider, key).call(image, item.target)
        except Exception as exc:  # noqa: BLE001 - a provider that fails is reported, not fatal
            return {**row, "ok": False, "error": str(exc)[:160]}
        try:
            point_of(text, candidate, image_size(image))
        except BadReply as exc:
            row["bad"] = str(exc)[:160]
        return {**row, "ok": True, **timing}

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(run, jobs))
    saved = OUT / "probe.json"
    chosen: dict[str, dict] = json.loads(saved.read_text())["chosen"] if only and saved.exists() else {}
    for c in candidates:
        stats = []
        for p in c.providers:
            mine = [r for r in rows if r["label"] == c.label and r["provider"] == p]
            ok = [r["total"] for r in mine if r["ok"]]
            errors = [r["error"] for r in mine if not r["ok"]]
            stats.append({"provider": p, "ok": len(ok), "of": len(mine), "p50": statistics.median(ok) if ok else None,
                          "cost": sum(r.get("cost") or 0 for r in mine), "errors": errors[:2]})
            print(f"{c.label:34} {p:24} ok {len(ok)}/{len(mine)} p50 "
                  f"{stats[-1]['p50'] or 0:.2f} s {errors[:1]}")
        working = [s for s in stats if s["ok"] >= s["of"] - 1]
        best = min(working, key=lambda s: s["p50"]) if working else max(stats, key=lambda s: s["ok"])
        chosen[c.label] = {"provider": best["provider"], "probe": stats}
    saved.write_text(json.dumps({"chosen": chosen, "rows": rows}, indent=1))  # rows: this probe's only
    print(f"probe cost ${sum(r.get('cost') or 0 for r in rows):.4f}")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxes", action="store_true", help="draw the laptop truth boxes and stop (no API calls)")
    parser.add_argument("--probe", action="store_true", help="time every listed provider and pin the fastest")
    parser.add_argument("--report", action="store_true", help="report saved rows (no API calls)")
    parser.add_argument("--only", nargs="*", help="candidate labels to run (default all, Jev included)")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--pilot", action="store_true", help="one repeat of 6 items per set, rows to pilot.json")
    args = parser.parse_args()
    if args.boxes:
        draw_boxes()
        return
    if args.report:
        report(json.loads((OUT / "rows.json").read_text()))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    items, images = load_items()
    key = os.environ["OPENROUTER_API_KEY"]
    if args.probe:
        probe(key, items, images, args.only)
        return
    chosen = json.loads((OUT / "probe.json").read_text())["chosen"]
    routers = {c.label: Router(c, chosen[c.label]["provider"], key) for c in CANDIDATES
               if c.label in chosen and (not args.only or c.label in args.only)}
    jev = Jev(os.environ["TYPESAFE_API_KEY"]) if not args.only or JEV in args.only else None
    if args.pilot:
        items, args.repeats = [i for i in items if i.set == "laptop"][::9] + [i for i in items if i.set != "laptop"][::9], 1
    names = list(routers) + ([JEV] if jev else [])
    jobs = [(name, item, r) for r in range(args.repeats) for name in names for item in items]
    random.Random(17).shuffle(jobs)  # interleaved: provider latency drifts over an hour

    def run(job: tuple) -> dict:
        row = _run(job)
        for wait in RETRIES_S:
            if row["outcome"] != "error" or not re.search(r"HTTP (429|5\d\d)|Error|Timeout", row["error"]):
                break
            time.sleep(wait)
            row = {**_run(job), "retried": True}
        return row

    def _run(job: tuple) -> dict:
        name, item, repeat = job
        row = {"model": name, "set": item.set, "screen": item.screen, "item": item.target, "kind": item.kind,
               "absent": not item.boxes, "repeat": repeat}
        try:
            if name == JEV:
                point, elapsed = jev.find(images[item.screen], item.target)
                return {**row, "outcome": outcome(point, item), "point": point, "total": elapsed}
            router = routers[name]
            image = shrink(images[item.screen])
            text, timing = router.call(image, item.target)
            row.update(reply=text[:200], provider=router.provider, **timing)
            point = point_of(text, router.candidate, image_size(image))
            return {**row, "outcome": outcome(point, item), "point": point}
        except BadReply:
            return {**row, "outcome": "bad"}
        except Exception as exc:  # noqa: BLE001 - every failure is a row in the report
            return {**row, "outcome": "error", "error": f"{type(exc).__name__}: {exc}"[:240]}

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(run, jobs))
    if jev:
        jev.close()
    path = OUT / ("pilot.json" if args.pilot else "rows.json")
    if not args.pilot and args.only and path.exists():  # a partial rerun replaces only its own models' rows
        rows = [r for r in json.loads(path.read_text()) if r["model"] not in names] + rows
    path.write_text(json.dumps(rows, indent=1))
    report(rows)


def _pct(values: list[float], q: float) -> str:
    if not values:
        return "–"
    values = sorted(values)
    return f"{values[min(len(values) - 1, int(len(values) * q))]:.2f}"


def _counts(rows: list[dict]) -> str:
    """Present targets: hit / misclick / refused / bad reply; absent targets: refused of all."""
    present = [r for r in rows if not r["absent"]]
    absent = [r for r in rows if r["absent"]]
    n = {o: sum(r["outcome"] == o for r in present) for o in ("hit", "misclick", "refused", "bad")}
    return (f"{n['hit']} / {n['misclick']} / {n['refused']} / {n['bad']} | "
            f"{sum(r['outcome'] == 'refused' for r in absent)}/{len(absent)}")


def chain(rows: list[dict], fallback: str) -> list[dict]:
    """Jev first, `fallback` only where Jev refused: each item's Jev row, then the fallback's row of the same item and
    repeat; latencies add. Items where either call errored are left out."""
    key = lambda r: (r["set"], r["screen"], r["item"], r["repeat"])  # noqa: E731
    second = {key(r): r for r in rows if r["model"] == fallback and r["outcome"] != "error"}
    out = []
    for first in (r for r in rows if r["model"] == JEV and r["outcome"] != "error"):
        if first["outcome"] != "refused":
            out.append(first)
        elif (other := second.get(key(first))) is not None:
            out.append({**other, "total": first["total"] + other["total"]})
    return out


def report(rows: list[dict]) -> None:
    models = list(dict.fromkeys(r["model"] for r in rows))
    for name in sorted(models, key=lambda m: m != JEV):
        mine = [r for r in rows if r["model"] == name]
        ok = [r for r in mine if r["outcome"] != "error"]
        laptop = [r for r in ok if r["set"] == "laptop"]
        synth = [r for r in ok if r["set"] == "synthetic"]
        totals = [r["total"] for r in ok]
        costs = [r["cost"] for r in mine if r.get("cost") is not None]
        provider = next((r["provider"] for r in mine if r.get("provider")), "TypeSafe")
        print(f"| {name} | {provider} | {_counts(laptop)} | {_counts(synth)} | {len(mine) - len(ok)} | "
              f"{_pct(totals, .5)} / {_pct(totals, .9)} | {sum(costs):.3f} |")
    print("\nDerived: Jev first, the model where Jev refuses")
    for name in (m for m in models if m != JEV):
        rows_chain = chain(rows, name)
        laptop = [r for r in rows_chain if r["set"] == "laptop"]
        synth = [r for r in rows_chain if r["set"] == "synthetic"]
        totals = [r["total"] for r in rows_chain]
        mean = statistics.mean(totals) if totals else 0
        print(f"| Jev → {name} | {_counts(laptop)} | {_counts(synth)} | {mean:.2f} / {_pct(totals, .9)} |")
    print("\nLaptop targets by kind: hit / misclick / no click (refused or bad reply)")
    kinds = sorted({r["kind"] for r in rows if r["set"] == "laptop"})
    print("| Model | " + " | ".join(kinds) + " |")
    for name in sorted(models, key=lambda m: m != JEV):
        cells = []
        for kind in kinds:
            sel = [r for r in rows if r["model"] == name and r["set"] == "laptop" and r["kind"] == kind
                   and r["outcome"] != "error"]
            cells.append("/".join(str(sum(r["outcome"] in o for r in sel)) for o in (("hit",), ("misclick",),
                                                                                       ("refused", "bad"))))
        print(f"| {name} | " + " | ".join(cells) + " |")
    print("\nOpenRouter cost of these rows: $%.3f" % sum(r.get("cost") or 0 for r in rows))
    for r in [r for r in rows if r["outcome"] == "error"][:12]:
        print("  error:", r["model"], r["item"][:40], r["error"][:160])


if __name__ == "__main__":
    main()
