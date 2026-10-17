"""Learned cache (experimental, JEV_LEARNED_CACHE=1): replay decisions that worked.

After a Jev triage decision (a click, or a solo skill) visibly worked, it's remembered under
the OS, the request's exact wording and a fingerprint of the screen it was decided on. The
same request on a near-identical screen replays at once: no Jev call, no planner. Skills
that don't depend on the screen (open an app, press a key, scroll) match on any screen.
A replay that changes nothing is forgotten. Irreversible decisions are never stored.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from brain.agent.triage import Triage

FINGERPRINT_W, FINGERPRINT_H = 17, 8  # difference hash: 16x8 = 128 bits
MATCH_BITS = 10  # at most this many differing bits counts as the same screen (a clock or cursor may change)
MAX_ENTRIES = 500  # least recently used are dropped beyond this
ANY_SCREEN_SKILLS = frozenset({"open_app", "press_key", "scroll"})


def fingerprint(jpeg: bytes) -> int:
    """A 128-bit difference hash of the screen: neighbouring-pixel brightness comparisons on a
    tiny grayscale copy, so small changes flip few bits and different screens flip many."""
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("invalid image")
    small = cv2.resize(image, (FINGERPRINT_W, FINGERPRINT_H), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if bit else "0" for bit in bits), 2)


def normalize(utterance: str) -> str:
    """Case, punctuation and spacing don't change what was asked."""
    return " ".join(re.sub(r"[^\w\s.]", " ", utterance.lower()).split())


class LearnedCache:
    def __init__(self, path: Path, max_entries: int = MAX_ENTRIES) -> None:
        self.path, self.max_entries = path, max_entries
        self.entries: list[dict] = self._load()

    def lookup(self, os_name: str, utterance: str, jpeg: bytes) -> Triage | None:
        entry = self._find(os_name, utterance, jpeg)
        if entry is None:
            return None
        entry["used"], entry["hits"] = time.time(), entry.get("hits", 0) + 1
        self._save()
        decision = dict(entry["decision"])
        if decision.get("point") is not None:
            decision["point"] = tuple(decision["point"])
        return Triage(**decision)

    def store(self, os_name: str, utterance: str, jpeg: bytes, decision: Triage) -> None:
        if decision.irreversible or decision.mode not in ("act", "keys"):
            return
        self.forget(os_name, utterance, jpeg)
        self.entries.append({"os": os_name, "utterance": normalize(utterance), "screen": fingerprint(jpeg),
                             "decision": asdict(decision), "used": time.time(), "hits": 0})
        self.entries = sorted(self.entries, key=lambda e: e["used"])[-self.max_entries:]
        self._save()

    def forget(self, os_name: str, utterance: str, jpeg: bytes) -> None:
        entry = self._find(os_name, utterance, jpeg)
        if entry is not None:
            self.entries.remove(entry)
            self._save()

    def _find(self, os_name: str, utterance: str, jpeg: bytes) -> dict | None:
        wanted, screen = normalize(utterance), None
        for entry in reversed(self.entries):  # newest first
            if entry["os"] != os_name or entry["utterance"] != wanted:
                continue
            if entry["decision"].get("skill") in ANY_SCREEN_SKILLS:
                return entry
            screen = fingerprint(jpeg) if screen is None else screen
            if bin(entry["screen"] ^ screen).count("1") <= MATCH_BITS:
                return entry
        return None

    def _load(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return []
        return [e for e in data if isinstance(e, dict) and {"os", "utterance", "screen", "decision"} <= e.keys()] \
            if isinstance(data, list) else []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(self.entries))
        os.replace(temp, self.path)  # atomic: a crash never leaves half a file
