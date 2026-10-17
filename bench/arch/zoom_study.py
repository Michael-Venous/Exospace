"""Study 1b: does a zoomed second Jev pass fix small-target misses?

Takes every first-pass triage point (from triage raw.json), crops a window around
it, enlarges it, and asks Jev to box the same utterance's target again.
"""

from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from bench.arch.cases import CASES
from bench.arch.triage_study import SCREENS, box_center, hit, jev_raw
from brain.agent.jev import JevVision
from brain.cancel import CancelToken
from brain.config import Config

OUT = Path("work/arch/zoom")
WINDOW = 0.30  # crop side as a fraction of the longer screen side


def crop_around(jpeg: bytes, point: tuple[float, float]) -> tuple[bytes, tuple[float, float, float, float]]:
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    side = WINDOW * max(w, h)
    half_w, half_h = min(side, w) / 2, min(side * 0.66, h) / 2
    cx = min(max(point[0] * w, half_w), w - half_w)
    cy = min(max(point[1] * h, half_h), h - half_h)
    x0, y0, x1, y1 = int(cx - half_w), int(cy - half_h), int(cx + half_w), int(cy + half_h)
    crop = cv2.resize(image[y0:y1, x0:x1], None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return buf.tobytes(), (x0 / w, y0 / h, x1 / w, y1 / h)


def refine(jev: JevVision, jpeg: bytes, utterance: str, point: tuple[float, float]) -> dict:
    crop, (x0, y0, x1, y1) = crop_around(jpeg, point)
    questions = {
        "present": {"type": "noul", "instructions":
                    "This is a zoomed-in part of a screen. Is the control the `utterance` asks to click, tap or open visible here?"},
        "target": {"type": "box", "instructions":
                   "This is a zoomed-in part of a screen. Locate exactly the control the `utterance` asks to click, tap or open."},
    }
    answers, size, padded, elapsed = jev_raw(jev, crop, utterance, questions)
    local = box_center(answers["target"], size, padded)
    full = None if local is None else (x0 + local[0] * (x1 - x0), y0 + local[1] * (y1 - y0))
    return {"elapsed": elapsed, "present": answers["present"].get("noul"),
            "box_conf": answers["target"].get("confidence"), "point": full}


def main() -> None:
    cfg = Config.load()
    OUT.mkdir(parents=True, exist_ok=True)
    boxes = json.loads((SCREENS / "boxes.json").read_text())
    images = {s: (SCREENS / f"{s}.jpg").read_bytes() for s in boxes}
    lookup = {(c.screen, c.utterance): c for c in CASES}
    first = [r for r in json.loads(Path("work/arch/triage/raw.json").read_text())["jev_triage"]
             if r["choice"] == "act_now" and r["point"]]
    jev = JevVision(cfg.typesafe_api_key, CancelToken(), OUT / "calls.jsonl", cfg.jev_model, 30)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(r, pool.submit(refine, jev, images[r["screen"]], r["utterance"], tuple(r["point"]))) for r in first]
        rows = []
        for r, fut in futures:
            case = lookup[(r["screen"], r["utterance"])]
            z = fut.result()
            ids = case.act_ids
            rows.append({"utterance": r["utterance"], "screen": r["screen"],
                         "first_hit": hit(tuple(r["point"]), ids, boxes[case.screen]) if ids else None,
                         "first_conf": r["box_conf"], "zoom_hit": hit(z["point"], ids, boxes[case.screen]) if ids else None,
                         "zoom_conf": z["box_conf"], "zoom_present": z["present"], "zoom_s": z["elapsed"]})
    jev.close()
    (OUT / "rows.json").write_text(json.dumps(rows, indent=1))
    scored = [r for r in rows if r["first_hit"] is not None]
    print("n", len(scored))
    print("first-pass hits", sum(r["first_hit"] for r in scored), "zoom hits", sum(r["zoom_hit"] for r in scored))
    for gate in (0.0, 0.5, 0.6, 0.7):
        g = [r for r in scored if r["zoom_conf"] >= gate and r["zoom_present"] >= 0.5]
        print(f"zoom gate>={gate}: acted {len(g)} hits {sum(r['zoom_hit'] for r in g)} misses {sum(not r['zoom_hit'] for r in g)}")
    print("zoom p50", round(statistics.median(r["zoom_s"] for r in rows), 3))
    for r in rows:
        if not r["zoom_hit"] or not r["first_hit"]:
            print(r)


if __name__ == "__main__":
    main()
