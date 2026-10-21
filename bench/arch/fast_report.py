"""Studies 12-13's report: the fast element-list planner against the runtime race (no API calls).

Reads `work/arch/e2e*-fast/report.json` and each run's `trace.json`: success, unsafe input, false "done",
wall time, and for the candidates the time each step spent building its element list (perception) and
in the fast model, next to race's Flash plans.

  PYTHONPATH=. .venv/bin/python -m bench.arch.fast_report [folder, default work/arch] [tag, default fast]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path("work/arch")  # the latest round; earlier ones are moved to subfolders


def _p(values: list[float], q: float) -> str:
    values = sorted(values)
    return f"{values[min(len(values) - 1, int(len(values) * q))]:.1f}" if values else "–"


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT
    tag = sys.argv[2] if len(sys.argv) > 2 else "fast"
    for report in sorted(root.glob(f"e2e*-{tag}/report.json")):
        rows = json.loads(report.read_text())
        print(f"\n{report.parent.name}: {len(rows)} runs")
        print("| Executor | Success | Unsafe | False done | Planned ok | Planned wall sum / p50 / p90 s | "
              "Steps' perception s | Model s (calls) | Runs that called a planner model |")
        print("|---|---|---|---|---|---|---|---|---|")
        for name in dict.fromkeys(r["executor"] for r in rows):
            mine = [r for r in rows if r["executor"] == name]
            planned = [r for r in mine if r["kind"] == "plan"]
            perceive = model = calls = 0.0
            handed = 0
            for r in mine:
                folder = report.parent / f"{name.split()[0].replace('+', '_')}-{r['segment']}-{r['repeat']}"
                trace = json.loads((folder / "trace.json").read_text()) if (folder / "trace.json").exists() else []
                handed += any(t.get("label") in ("fast-list", "plan") for t in trace)
                for t in trace:
                    if t.get("label") == "fast-list":
                        perceive += t["perceive_s"]
                        model += t["model_s"]
                        calls += 1
                    elif t.get("label") == "plan":
                        model += t.get("elapsed_s") or 0
                        calls += 1
            walls = [r["wall_s"] for r in planned]
            print(f"| {name} | {sum(r['success'] for r in mine)}/{len(mine)} | {sum(r['unsafe'] for r in mine)} | "
                  f"{sum(r['status'] == 'done' and not r['success'] for r in mine)} | "
                  f"{sum(r['success'] for r in planned)}/{len(planned)} | "
                  f"{sum(walls):.0f} / {_p(walls, .5)} / {_p(walls, .9)} | "
                  f"{perceive:.0f} | {model:.0f} ({calls:.0f}) | {handed} |")
        print("misses:", {name: [(r["segment"], r["repeat"], r["status"], r["final_view"])
                                 for r in rows if r["executor"] == name and not r["success"]]
                          for name in dict.fromkeys(r["executor"] for r in rows)})
        firsts = {name: [r["first_input_s"] for r in rows if r["executor"] == name and r["first_input_s"]]
                  for name in dict.fromkeys(r["executor"] for r in rows)}
        print("first input p50:", {k: round(statistics.median(v), 2) for k, v in firsts.items() if v})


if __name__ == "__main__":
    main()
