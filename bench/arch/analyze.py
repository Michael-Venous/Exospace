"""Final analysis across both end-to-end scenarios.

Success rates with Wilson 95% intervals, per-category breakdown, latency, LLM calls,
and an exact paired sign test between two architectures on identical
(scenario, segment, repeat) runs.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

REPORTS = {"workstation": Path("work/arch/e2e/report.json"), "studio": Path("work/arch/e2e_studio/report.json")}
# names differ between runs; map them onto one label per architecture
# Only the interleaved finalist runs, so each architecture appears once per scenario.
ALIASES = {"jev-flash (current)": "current", "final-concrete": "race+concrete", "final-steps-aware": "race+steps-aware"}


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z, p = 1.96, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial p-value on discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load() -> list[dict]:
    rows = []
    for scenario, path in REPORTS.items():
        if not path.exists():
            continue
        for r in json.loads(path.read_text()):
            label = ALIASES.get(r["executor"])
            if label:
                rows.append({**r, "scenario": scenario, "arch": label})
    return rows


def category(r: dict) -> str:
    return {"act": "direct", "none": "context", "plan": "planned", "confirm": "destructive"}.get(r.get("kind", ""), "?")


def main() -> None:
    rows = load()
    archs = ["current", "race+concrete", "race+steps-aware"]
    print("| Architecture | Success (95% CI) | Unsafe | Direct first input p50 | Segment p50 | LLM calls/segment |")
    print("|---|---|---|---|---|---|")
    for a in archs:
        r = [x for x in rows if x["arch"] == a]
        if not r:
            continue
        k, n = sum(x["success"] for x in r), len(r)
        lo, hi = wilson(k, n)
        direct = [x["first_input_s"] for x in r if category(x) == "direct" and x["first_input_s"] is not None]
        print(f"| {a} | {k}/{n} = {k / n:.0%} ({lo:.0%}–{hi:.0%}) | {sum(x['unsafe'] for x in r)} | "
              f"{statistics.median(direct):.2f} s | {statistics.median(x['wall_s'] for x in r):.1f} s | "
              f"{sum(x['llm_calls'] for x in r) / n:.2f} |")
    print("\nBy category (success):")
    for cat in ["direct", "context", "planned", "destructive"]:
        cells = []
        for a in archs:
            r = [x for x in rows if x["arch"] == a and category(x) == cat]
            if r:
                cells.append(f"{a} {sum(x['success'] for x in r)}/{len(r)}")
        print(f"  {cat:12} " + " | ".join(cells))
    print("\nBy scenario (success):")
    for sc in REPORTS:
        print(f"  {sc:12} " + " | ".join(
            f"{a} {sum(x['success'] for x in rows if x['arch'] == a and x['scenario'] == sc)}/"
            f"{sum(1 for x in rows if x['arch'] == a and x['scenario'] == sc)}" for a in archs))

    def paired(a: str, b: str) -> None:
        key = lambda x: (x["scenario"], x["segment"], x["repeat"])  # noqa: E731
        ra = {key(x): x["success"] for x in rows if x["arch"] == a}
        rb = {key(x): x["success"] for x in rows if x["arch"] == b}
        common = ra.keys() & rb.keys()
        wins = sum(rb[k] and not ra[k] for k in common)
        losses = sum(ra[k] and not rb[k] for k in common)
        print(f"  {b} vs {a}: {len(common)} paired runs, {b} better on {wins}, worse on {losses}, "
              f"sign-test p = {sign_test(wins, losses):.3g}")

    print("\nPaired comparisons (same scenario/segment/repeat):")
    paired("current", "race+steps-aware")
    paired("current", "race+concrete")
    paired("race+concrete", "race+steps-aware")


if __name__ == "__main__":
    main()
