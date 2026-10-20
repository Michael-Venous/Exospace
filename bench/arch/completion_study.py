"""Study 10: the completion check on the calls real runs made, labelled by the fixture's state at the time.

Input: the `completion.json` + `completion-<n>.jpg` that e2e runs record (`completion_log.py`) under
`work/arch/*-cl*/`. Each call is labelled done / not done by its scenario's own oracle on `window.S` when
the screen was taken. Only segments whose oracle is the goal itself count (kinds plan and act).

  logged   the runtime's answer as it was given, at the gate its caller used: 0.5 for a planner's
           check (it passes `expected_evidence`), 0.55 after a direct click (race.DIRECT_GATE)
  replay   the same screens and state again (`--replay`, Jev calls): the runtime question set, with
           one generic extra question asked in the same call ("outcome", below), and the plain
           question without the planner's evidence; `--repeats` times each, for noise and medians

  PYTHONPATH=. .venv/bin/python -m bench.arch.completion_study            # logged calls only, no API calls
  PYTHONPATH=. .venv/bin/python -m bench.arch.completion_study --replay   # + Jev replays
  PYTHONPATH=. .venv/bin/python -m bench.arch.completion_study --study5   # study 5's screens, both gates
"""

from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench.arch import studio, vague, workstation
from brain.agent.jev import COMPLETION_HINTED, COMPLETION_QUESTION, EVIDENCE_SUFFICIENT, JevVision, probability
from brain.agent.jev_flash import COMPLETION_GATE
from brain.agent.race import DIRECT_GATE
from brain.cancel import CancelToken
from brain.config import Config

ROOT = Path("work/arch")
OUT = ROOT / "completion"
SCENARIOS = {"e2e_vague": vague, "e2e_studio": studio, "e2e": workstation}
# Generic, asked of every goal alike: the logged false passes were a page on the way to the goal (YouTube's
# home page after a typed search, nothing playing) read as the goal itself.
OUTCOME = ("Does `screen` show the end result the user's `goal` asks for, what the user wanted to see or have "
           "happen, rather than a page, list or field on the way to it?")
GATES = [round(0.3 + 0.05 * i, 2) for i in range(13)]


def calls() -> list[dict]:
    """Every logged completion call of a goal-oracle segment, labelled."""
    out = []
    for log in sorted(ROOT.glob("*-cl*/*/completion.json")):
        study = log.parent.parent.name.split("-cl")[0]
        executor, segment, repeat = log.parent.name.rsplit("-", 2)
        utterance, _, oracle, kind = SCENARIOS[study].SEGMENTS[int(segment)]
        if kind not in ("plan", "act"):
            continue
        trace = [t for t in json.loads((log.parent / "trace.json").read_text()) if t.get("label") == "completion"]
        for call in json.loads(log.read_text()):
            answers = trace[call["n"]]["answers"] if call["n"] < len(trace) else {}
            out.append({**call, "run": f"{log.parent.parent.name}/{log.parent.name}", "utterance": utterance,
                        "done": bool(oracle(call["state"])), "gate": COMPLETION_GATE if call["evidence"] else DIRECT_GATE,
                        "answers": {k: v.get("noul") for k, v in answers.items()},
                        "image": str(log.parent / f"completion-{call['n']}.jpg")})
    return out


def errors(rows: list[dict], score: str, gate: float | None = None) -> tuple[int, int, int, int]:
    """(false passes, not-done calls, finished work rejected, done calls) at each call's own gate or `gate`."""
    passed = [(r[score] >= (gate if gate is not None else r["gate"])) for r in rows]
    false_pass = sum(p and not r["done"] for p, r in zip(passed, rows))
    rejected = sum(not p and r["done"] for p, r in zip(passed, rows))
    return false_pass, sum(not r["done"] for r in rows), rejected, sum(r["done"] for r in rows)


def replay(rows: list[dict], repeats: int) -> None:
    cfg = Config.load()
    jev = JevVision(cfg.typesafe_api_key, CancelToken(), OUT / "jev.jsonl", cfg.jev_model, 30)

    def ask(job: tuple[int, int]) -> tuple[int, dict, float]:
        i, _ = job
        r = rows[i]
        image = Path(r["image"]).read_bytes()
        state = {"goal": r["goal"], "executed_steps": r["executed"]}
        runtime = {"complete": {"type": "noul", "instructions": COMPLETION_QUESTION}}
        if r["evidence"]:  # the runtime's own question set (JevVision.completion)
            state["expected_evidence"] = r["evidence"]
            runtime = {"complete": {"type": "noul", "instructions": COMPLETION_QUESTION +
                                    " Ignore `expected_evidence` for this question."},
                       "hinted": {"type": "noul", "instructions": COMPLETION_HINTED},
                       "sufficient": {"type": "noul", "instructions": EVIDENCE_SUFFICIENT}}
        answers, _, _ = jev.evaluate(image, {**runtime, "outcome": {"type": "noul", "instructions": OUTCOME}},
                                     state=state, label="completion-replay")
        plain, _, _ = jev.evaluate(image, {"complete": {"type": "noul", "instructions": COMPLETION_QUESTION}},
                                   state={"goal": r["goal"], "executed_steps": r["executed"]}, label="completion-plain")
        value = {k: probability(v.get("noul")) for k, v in answers.items()}
        score = value["hinted"] if r["evidence"] and value["sufficient"] >= 0.5 else value["complete"]
        return i, {"runtime": score, "outcome": value["outcome"]}, probability(plain["complete"].get("noul"))

    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, scores, plain in pool.map(ask, [(i, k) for k in range(repeats) for i in range(len(rows))]):
            rows[i].setdefault("replays", []).append({**scores, "plain": plain})
    jev.close()


def report(rows: list[dict]) -> None:
    done = [r for r in rows if r["done"]]
    print(f"{len(rows)} labelled calls from {len({r['run'] for r in rows})} runs: {len(done)} on finished work, "
          f"{len(rows) - len(done)} not")
    fp, nd, rj, dn = errors(rows, "score")
    print(f"logged, each caller's gate: false passes {fp}/{nd}, finished work rejected {rj}/{dn}")
    for name in ("score",):
        print(f"\n| gate | false pass / {nd} | rejected / {dn} |\n|---|---|---|")
        for g in GATES:
            fp, _, rj, _ = errors(rows, name, g)
            print(f"| {g} | {fp} | {rj} |")
    print("\nnot-done calls scoring >= 0.3:")
    for r in sorted((r for r in rows if not r["done"] and r["score"] >= 0.3), key=lambda r: -r["score"]):
        print(f"  {r['score']:.2f} gate {r['gate']} {r['utterance'][:40]!r} view={r['state'].get('view') or r['state'].get('app')} "
              f"last={r['executed'][-1:]} {r['answers']}")
    print("\nfinished work scoring < 0.55:")
    for r in sorted((r for r in rows if r["done"] and r["score"] < 0.55), key=lambda r: r["score"]):
        print(f"  {r['score']:.2f} gate {r['gate']} {r['utterance'][:40]!r} last={r['executed'][-1:]} {r['answers']}")
    replayed = [r for r in rows if r.get("replays")]
    if not replayed:
        return
    for r in replayed:
        reps = r["replays"]
        r["runtime_med"] = statistics.median(x["runtime"] for x in reps)
        r["plain_med"] = statistics.median(x["plain"] for x in reps)
        r["both_med"] = statistics.median(min(x["runtime"], x["outcome"]) for x in reps)
        r["runtime_first"] = reps[0]["runtime"]
        r["both_first"] = min(reps[0]["runtime"], reps[0]["outcome"])
        r["flip"] = len({x["runtime"] >= r["gate"] for x in reps}) > 1
    print(f"\nreplayed {len(replayed)} calls x {len(replayed[0]['replays'])}; "
          f"runtime verdict changed between repeats on {sum(r['flip'] for r in replayed)}")
    fp0, nd, rj0, dn = errors(replayed, "score")
    print(f"| variant (each caller's gate) | false pass / {nd} | rejected / {dn} |\n|---|---|---|")
    print(f"| logged (as the run saw it) | {fp0} | {rj0} |")
    for label, key in (("runtime, one replay", "runtime_first"), ("runtime, median of replays", "runtime_med"),
                       ("plain question (no evidence), median", "plain_med"),
                       ("runtime AND outcome, one call", "both_first"), ("runtime AND outcome, median", "both_med")):
        fp, _, rj, _ = errors(replayed, key)
        print(f"| {label} | {fp} | {rj} |")
    print("\n| gate | runtime AND outcome: false pass | rejected |\n|---|---|---|")
    for g in GATES:
        fp, _, rj, _ = errors(replayed, "both_first", g)
        print(f"| {g} | {fp} | {rj} |")


def study5(repeats: int) -> None:
    """Study 5's finished-task and trap screens (and its held-out set) through the runtime's own check."""
    from bench.arch import verify_study

    cfg = Config.load()
    jev = JevVision(cfg.typesafe_api_key, CancelToken(), OUT / "study5.jsonl", cfg.jev_model, 30)
    sets = {"study 5": (verify_study.CASES, verify_study.render()),
            "study 5 held-out": (verify_study.HELDOUT, verify_study.render(verify_study.heldout_screens()))}
    for name, (cases, shots) in sets.items():
        jobs = [(case, r) for case in cases for r in range(repeats)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            scores = list(pool.map(lambda job: jev.completion(shots[job[0][1]], job[0][0], job[0][2],
                                                              evidence=job[0][3]), jobs))
        for gate in (COMPLETION_GATE, 0.6):
            accepted = sum(s >= gate for (case, _), s in zip(jobs, scores) if case[4])
            traps = sum(s >= gate for (case, _), s in zip(jobs, scores) if not case[4])
            print(f"{name}, gate {gate}: accepts finished {accepted}/{sum(c[4] for c, _ in jobs)}, "
                  f"false done {traps}/{sum(not c[4] for c, _ in jobs)}")
        near = sorted(round(s, 2) for (case, _), s in zip(jobs, scores) if case[4] and s < 0.7)
        print(f"  finished scores below 0.7: {near}")
    jev.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true", help="ask Jev again on the saved screens")
    parser.add_argument("--study5", action="store_true", help="study 5's screens through the runtime check")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.study5:
        study5(args.repeats)
        return
    saved = OUT / "rows.json"
    rows = calls()
    if args.replay:
        replay(rows, args.repeats)
        saved.write_text(json.dumps(rows, indent=1))
    elif saved.exists():
        old = {(r["run"], r["n"]): r for r in json.loads(saved.read_text())}
        rows = [{**r, **({"replays": old[(r["run"], r["n"])]["replays"]} if "replays" in old.get((r["run"], r["n"]), {})
                         else {})} for r in rows]
    report(rows)


if __name__ == "__main__":
    main()
