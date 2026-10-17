"""Study 8: does the triage call pick the right pointer gesture?

Runs the runtime `brain.agent.triage.triage` on direct requests whose correct gesture is
known: desktop icons need a double-click, "right-click"/options requests a right-click,
everything else (buttons, links, menus, sidebar items, phone apps) a single click.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench.arch.verify_study import heldout_screens, render
from brain.agent.jev import JevVision
from brain.agent.triage import triage
from brain.cancel import CancelToken
from brain.config import Config

OUT = Path("work/arch/gesture")
CASES = [
    ("desktop", "open up chrome", "double_click"),
    ("desktop", "open blender", "double_click"),
    ("desktop", "open that q3 report draft", "double_click"),
    ("desktop", "right click the recycle bin", "right_click"),
    ("desktop", "show me the options for the q3 report file", "right_click"),
    ("blender", "right-click the cube", "right_click"),
    ("blender", "click the render menu", "click"),
    ("inbox", "open john's email", "click"),
    ("inbox", "go to my sent folder", "click"),
    ("youtube", "play the first video", "click"),
    ("phone", "open whatsapp", "click"),
    ("phone", "open the camera", "click"),
    ("s_downloads", "open the pictures folder", "click"),
    ("s_kb_not_added", "add it to my cart", "click"),
]


def main() -> None:
    cfg = Config.load()
    OUT.mkdir(parents=True, exist_ok=True)
    shots = render(heldout_screens())
    for name in ("desktop", "inbox", "youtube", "phone", "blender"):
        shots[name] = (Path("work/arch/screens") / f"{name}.jpg").read_bytes()
    jev = JevVision(cfg.typesafe_api_key, CancelToken(), OUT / "calls.jsonl", cfg.jev_model, 30)

    def run(case):
        screen, utterance, expected = case
        return case, triage(jev, shots[screen], utterance, CancelToken())

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(run, [c for c in CASES for _ in range(3)]))
    jev.close()
    rows = [{"screen": s, "utterance": u, "expected": e, "mode": r.mode, "gesture": r.gesture} for (s, u, e), r in results]
    (OUT / "rows.json").write_text(json.dumps(rows, indent=1))
    acted = [r for r in rows if r["mode"] == "act"]
    right = sum(r["gesture"] == r["expected"] for r in acted)
    print(f"acted on {len(acted)}/{len(rows)}; gesture correct {right}/{len(acted)}")
    print("by expected gesture:", {g: f"{sum(r['gesture'] == g for r in acted if r['expected'] == g)}/"
                                      f"{sum(1 for r in acted if r['expected'] == g)}"
                                   for g in ("click", "double_click", "right_click")})
    for (screen, utterance, expected) in CASES:
        got = Counter(f"{r['mode']}:{r['gesture']}" for r in rows if r["utterance"] == utterance)
        print(f"  {expected:12} {dict(got)}  | {utterance}")


if __name__ == "__main__":
    main()
