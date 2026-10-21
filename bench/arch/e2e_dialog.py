"""Study 14: multi-turn requests on the simulated desktop (`pc.py`), through the real Agent loop.

A scripted user says each turn in order, the way a person would answer the agent's questions; a turn the
agent didn't ask for still comes, as when someone adds detail. Judged by the desktop's final state. Per
dialog: success, unsafe (an action the dialog forbids, or the admin prompt accepted before any question),
questions asked, wall time over all turns, first input of the first turn, model calls and their seconds.

Executors: the runtime race on Gemini 3.8 Flash; the same race with its language model swapped for a fast
image model on OpenRouter (`race@<model>`); and the race with the fast list planner at the hand-off over a
CPU-OCR list, blind or with the screenshot too (`fast_list.py`). Interleaved, one dialog at a time.

  PYTHONPATH=. .venv/bin/python -m bench.arch.e2e_dialog [--dialogs 0 4] [--repeats 2] [--executors ...]
  PYTHONPATH=. .venv/bin/python -m bench.arch.e2e_dialog --web [...]      # real Chrome, public sites (study 15)
  PYTHONPATH=. .venv/bin/python -m bench.arch.e2e_dialog --ocr-recall      # OCR against the page's elements
  PYTHONPATH=. .venv/bin/python -m bench.arch.e2e_dialog --report
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import cv2

from bench.arch import pc, web
from bench.arch.e2e_study import KeptTrace
from bench.arch.fast_list import DOM_ELEMENTS, HybridOcrExecutor, HybridOcrImageExecutor
from bench.arch.router import routed
from bench.arch.xdesk import XTarget, start_display
from brain.agent.loop import Agent
from brain.agent.memory import Memory
from brain.agent.race import RaceExecutor
from brain.browser import BrowserTarget
from brain.cancel import CancelToken
from brain.capture import active_rect
from brain.config import Config
from brain.controller import Controller
from brain.telemetry import Telemetry
from brain.types import DeviceProfile

OUT = Path("work/arch/dialog")
MAX_STEPS = 20  # a turn's steps; the printer turn alone needs a dozen
# Fast image models (study 11: a screenshot question in <= ~1 s), as the race's language model.
FAST = {"deepseek": ("deepseek/deepseek-v4.1-flash-1017", ""), "qwen27b": ("qwen/qwen3.8-27b", ""),
        "gemma31b": ("google/gemma-4-31b-it", ""), "flashlite": ("google/gemini-3.5-flash-lite", "minimal")}


def race_on(model: str, effort: str):
    return lambda cfg, token: RaceExecutor(routed(cfg, model, effort), token)


EXECUTORS = {"race": RaceExecutor, **{f"race@{k}": race_on(*v) for k, v in FAST.items()},
             "hybrid-ocr": HybridOcrExecutor, "hybrid-ocr+image": HybridOcrImageExecutor}


def browser() -> BrowserTarget:
    target = BrowserTarget("about:blank")
    target._keymap.update({f"F{i}": f"F{i}" for i in range(1, 13)})  # the dev backend lowercases F4 (Alt+F4)
    target.page.set_content(pc.HTML)
    return target


class Desktop:
    """The simulated desktop page in headless Chromium (pc.py)."""

    out, dialogs = OUT, pc.DIALOGS

    def open(self, state: dict):
        target = browser()
        target.page.evaluate("s => load(s)", state)
        return target

    def state(self, target) -> dict:
        return target.page.evaluate("() => window.S")

    def asked(self, target) -> None:
        target.page.evaluate("() => { S.asked = true }")

    def screenshot(self, target, path: Path) -> None:
        target.page.screenshot(path=str(path))


class Web:
    """Real Chrome on the virtual display, public sites (xdesk.py, web.py)."""

    out, dialogs = OUT.with_name("web-dialog"), web.DIALOGS

    def open(self, urls: list[str]):
        return XTarget(urls)

    def state(self, target) -> dict:
        return {"tabs": target.tabs()}

    def asked(self, target) -> None:
        pass

    def screenshot(self, target, path: Path) -> None:
        cv2.imwrite(str(path), target.grab())


def run_dialog(name: str, index: int, repeat: int, base: Config, fixture: Desktop | Web) -> dict:
    title, state, turns, oracle, unsafe = fixture.dialogs[index]
    folder = fixture.out / f"{name.replace('+', '_').replace('@', '_')}-{index}-{repeat}"
    folder.mkdir(parents=True, exist_ok=True)
    cfg = replace(base, gemini_thinking="low", model_log=folder / "model_calls.jsonl", bench_csv=folder / "results.csv")
    token, events, said = CancelToken(), [], []
    with ExitStack() as cleanup:
        target = fixture.open(state)
        cleanup.callback(target.close)
        executor = EXECUTORS[name](cfg, token)
        cleanup.callback(executor.close)
        trace: list[dict] = []
        executor.jev.trace = KeptTrace(trace)
        send = target.send
        clock = [time.monotonic()]

        def timed_send(command: dict) -> None:
            if command["op"] not in {"release_all", "ping"}:
                events.append(time.monotonic() - clock[0])
            send(command)

        target.send = timed_send
        agent = Agent(executor, Controller(target, target, token), Memory(folder / "memory.json"),
                      Telemetry(cfg.bench_csv, name), MAX_STEPS)
        profile = DeviceProfile("linux", active_rect(target.grab()), "mouse", time.time())
        started = time.monotonic()
        for k, utterance in enumerate(turns):
            clock[0], before = time.monotonic(), len(events)
            try:
                result = agent.run(utterance, profile, f"d{index}t{k}")
                status, message = result.status, result.message
            except Exception as exc:  # noqa: BLE001 - a crashed turn is a turn, the next one still comes
                status, message = "error", f"{type(exc).__name__}: {exc}"[:200]
            if status == "ask_user":
                fixture.asked(target)
            said.append({"turn": utterance, "status": status, "message": message[:200],
                         "wall_s": round(time.monotonic() - clock[0], 2),
                         "first_input_s": round(events[before], 2) if len(events) > before else None})
        wall = time.monotonic() - started
        final = fixture.state(target)
        fixture.screenshot(target, folder / "after.png")
    (folder / "trace.json").write_text(json.dumps(trace, indent=1, default=str))
    calls = [json.loads(line) for line in cfg.model_log.read_text().splitlines()] if cfg.model_log.exists() else []
    fast = [t for t in trace if t.get("label") == "fast-list"]
    return {"executor": name, "dialog": index, "title": title, "repeat": repeat, "success": bool(oracle(final)),
            "unsafe": bool(unsafe(final)), "questions": sum(t["status"] == "ask_user" for t in said),
            "final": final if isinstance(fixture, Web) else None,
            "wall_s": round(wall, 1), "turns": said,
            "llm_calls": sum(c["role"] != "jev" for c in calls) + len(fast),
            "llm_s": round(sum(c["elapsed_s"] for c in calls if c["role"] != "jev")
                           + sum(t["model_s"] for t in fast), 1),
            "perceive_s": round(sum(t["perceive_s"] for t in fast), 1),
            "jev_calls": sum(c["role"] == "jev" for c in calls)}


def ocr_recall() -> None:
    """How many of the page's interactive elements a CPU-OCR list contains, on the dialogs' screens."""
    from rapidocr_onnxruntime import RapidOCR

    engine, found, total, icons, seconds = RapidOCR(), 0, 0, 0, []
    norm = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())  # noqa: E731
    target = browser()
    screens = [pc.DIALOGS[i][1] for i in range(len(pc.DIALOGS))] + [
        pc.chrome_on("web.whatsapp.com"), {**pc.chrome_on("hp.com/drivers"), "downloads": ["x"]},
        {**pc._base(), "open": ["files"], "front": "files", "files": {"folder": "Downloads", "sel": None}},
        {**pc._base(), "open": ["settings"], "front": "settings", "settings": "Printers & scanners"}]
    for state in screens:
        target.page.evaluate("s => load(s)", state)
        elements = target.page.evaluate(DOM_ELEMENTS)
        frame = target.grab()
        started = time.monotonic()
        lines, _ = engine(frame, use_cls=False)
        seconds.append(time.monotonic() - started)
        text = " ".join(norm(t) for _, t, _ in lines or [])
        for label, *_ in elements:
            total += 1
            key = norm(label)[:24]
            if label == "(no text)" or len(key) < 2:
                icons += 1
            elif key in text:
                found += 1
    target.close()
    print(f"OCR (RapidOCR, CPU) on {len(screens)} screens, {total} interactive elements: {found} of the {total - icons} "
          f"with text found; {icons} are icons or symbols (taskbar, ✕, +, ←) that text can't find; "
          f"{statistics.median(seconds):.2f} s a screen (median)")


def report(rows: list[dict]) -> None:
    print("| Executor | Success | Unsafe | Questions | Dialog wall p50 / sum s | First input p50 s | "
          "LLM calls / s | Perception s |")
    print("|---|---|---|---|---|---|---|---|")
    for name in dict.fromkeys(r["executor"] for r in rows):
        mine = [r for r in rows if r["executor"] == name]
        firsts = [r["turns"][0]["first_input_s"] for r in mine if r["turns"][0]["first_input_s"]]
        print(f"| {name} | {sum(r['success'] for r in mine)}/{len(mine)} | {sum(r['unsafe'] for r in mine)} | "
              f"{sum(r['questions'] for r in mine)} | {statistics.median(r['wall_s'] for r in mine):.0f} / "
              f"{sum(r['wall_s'] for r in mine):.0f} | {statistics.median(firsts) if firsts else '–'} | "
              f"{sum(r['llm_calls'] for r in mine)} / {sum(r['llm_s'] for r in mine):.0f} | "
              f"{sum(r['perceive_s'] for r in mine):.0f} |")
    print("\nper dialog (success of repeats, mean wall s):")
    for d in sorted({r["dialog"] for r in rows}):
        title = next(r["title"] for r in rows if r["dialog"] == d)
        cells = []
        for name in dict.fromkeys(r["executor"] for r in rows):
            mine = [r for r in rows if r["executor"] == name and r["dialog"] == d]
            cells.append(f"{name} {''.join('Y' if r['success'] else ('U' if r['unsafe'] else 'n') for r in mine)} "
                         f"{statistics.mean(r['wall_s'] for r in mine):.0f}s")
        print(f"  {title}: " + " | ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--web", action="store_true", help="real Chrome on the virtual display (xdesk.py, web.py)")
    parser.add_argument("--dialogs", nargs="*", type=int)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--executors", nargs="+", default=list(EXECUTORS))
    parser.add_argument("--ocr-recall", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.ocr_recall:
        ocr_recall()
        return
    fixture = Web() if args.web else Desktop()
    if args.web:
        start_display()
    path = fixture.out / "report.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    if args.report:
        report(rows)
        return
    fixture.out.mkdir(parents=True, exist_ok=True)
    base, done = Config.load(), {(r["executor"], r["dialog"], r["repeat"]) for r in rows}
    for repeat in range(args.repeats):
        for index in args.dialogs if args.dialogs is not None else range(len(fixture.dialogs)):
            for name in args.executors if repeat % 2 == 0 else args.executors[::-1]:
                if (name, index, repeat) in done:
                    continue
                try:
                    row = run_dialog(name, index, repeat, base, fixture)
                except Exception as exc:  # noqa: BLE001 - one retry (a TLS context failed to load once, study 12)
                    print(f"retrying {name} dialog {index}: {type(exc).__name__}: {exc}"[:200], flush=True)
                    time.sleep(2.0)
                    row = run_dialog(name, index, repeat, base, fixture)
                rows.append(row)
                path.write_text(json.dumps(rows, indent=1))
                print(json.dumps({k: row[k] for k in ("executor", "title", "success", "unsafe", "questions", "wall_s",
                                                      "llm_calls")}), flush=True)
    report(rows)


if __name__ == "__main__":
    main()
