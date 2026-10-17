"""Probe Jev solo skills on a screenshot: what triage decides and what it would type. Sends
no input, so it's safe on any machine. Needs TYPESAFE_API_KEY in .env.

    python -m bench.solo_probe --image shot.png --os windows "open spotify" "go to youtube"
    python -m bench.solo_probe --local --os macos "launch the text editor"   # this machine's screen

Each line: the intent Jev chose with its probability, the words it picked, the decision
(keys / act / plan / none), and the keyboard actions a "keys" decision would send.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from brain.agent.jev import JevError, JevVision
from brain.agent.solo import solo_actions, spans
from brain.agent.triage import decide, triage_questions
from brain.cancel import CancelToken
from brain.config import Config

TOP_SPANS = 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="a screenshot (PNG or JPEG)")
    source.add_argument("--local", action="store_true", help="grab this machine's screen (needs `mss`)")
    parser.add_argument("--os", default="windows", help="target OS for the keyboard actions")
    parser.add_argument("utterances", nargs="+")
    args = parser.parse_args()

    cfg = Config.load()
    cancel = CancelToken()
    jev = JevVision(cfg.typesafe_api_key, cancel, cfg.model_log, cfg.jev_model, cfg.jev_timeout_s)
    jpeg = _screenshot(args)
    try:
        for utterance in args.utterances:
            options = spans(utterance)
            started = time.monotonic()
            try:
                answers, size, padded = jev.evaluate(jpeg, triage_questions(options), state={"utterance": utterance})
            except JevError as exc:
                hint = " (check TYPESAFE_API_KEY and the account's credit)" if "401" in str(exc) else ""
                raise SystemExit(f"{exc}{hint}") from None
            elapsed = time.monotonic() - started
            result = decide(jev, jpeg, utterance, answers, size, padded, options, cancel)
            _report(utterance, answers, options, result, args.os, elapsed)
    finally:
        jev.close()
    return 0


def _screenshot(args: argparse.Namespace) -> bytes:
    if args.local:
        import mss  # optional dependency: pip install -e '.[local]'

        with mss.mss() as sct:
            image = cv2.cvtColor(np.asarray(sct.grab(sct.monitors[1])), cv2.COLOR_BGRA2BGR)
    else:
        image = cv2.imread(str(args.image))
        if image is None:
            raise SystemExit(f"cannot read {args.image}")
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def _report(utterance: str, answers: dict, options: dict, result, os: str, elapsed: float) -> None:
    intent = answers["intent"]
    probs = intent.get("probabilities") or {}
    print(f"\n“{utterance}”  ({elapsed:.2f} s, one Jev call)")
    print("  intent:  " + "  ".join(f"{k} {v:.2f}" for k, v in sorted(probs.items(), key=lambda kv: -kv[1])))
    span = answers.get("name_span")
    if span:
        ranked = sorted((span.get("probabilities") or {}).items(), key=lambda kv: -kv[1])[:TOP_SPANS]
        print("  words:   " + "  ".join(f"“{options.get(k, k)}” {v:.2f}" for k, v in ranked))
        print(f"  browser in front: {answers['browser_front'].get('noul', 0):.2f}")
    print(f"  irreversible: {answers['irreversible'].get('noul', 0):.2f}")
    print(f"  decision: {result.mode}" + (f" → {result.skill} “{result.text}”" if result.mode == "keys" else ""))
    if result.mode == "keys":
        actions = solo_actions(result.skill, result.text, os) or []
        print("  would send: " + (", ".join(a.brief() for a in actions) or f"nothing (no keyboard route on {os})"))


if __name__ == "__main__":
    raise SystemExit(main())
