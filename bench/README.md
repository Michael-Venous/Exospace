# Benchmarks and studies

**Owner:** Models

Every speed or accuracy claim in the pitch comes from here. All of these make real API calls except where
noted; the offline test suite in `tests/` makes none. Synthetic pages render in the isolated browser backend,
and each study scores outcomes with its own oracle (DOM state or labels), never the agent's own claim.

| Path | What it holds |
|---|---|
| [`arch/`](arch/README.md) | **The architecture study behind the current runtime** (race + steps-aware completion): triage, planner choice, completion, irreversible-action and gesture studies, end-to-end and held-out runs, results and limits |
| `experiments/` | Earlier executor experiments (routines, recipes, Live, routing, overlap); opt-in, never change the runtime defaults |
| `interpreter_*.py` | Fast-interpreter study: compact text interpretation and screenshot grounding, with a frozen held-out set |
| `guard_cost.py` | Cost of the loop's mid-batch screen check (run it on the Pi) |
| `results.csv` | One row per agent step (PLAN.md §5.7), written by every runtime run |
| `model_calls.jsonl` | Per-call timings (role, model, attempt, outcome); no prompts, images or keys |

## Running

```sh
.venv/bin/python -m pip install -e '.[browser]' && .venv/bin/python -m playwright install chromium
.venv/bin/python -m bench.arch.e2e_study              # end-to-end on the mail/workstation fixture
.venv/bin/python -m bench.arch.triage_study           # triage accuracy on six screens
.venv/bin/python -m bench.arch.analyze                # pooled results and sign test (no API calls)
.venv/bin/python -m bench.experiments.probe --runs 3  # one synthetic-image call per model
.venv/bin/python -m bench.experiments.local_benchmark --runs 3   # calendar task with a DOM oracle
```

Rules for new studies:
- Write the oracle before the run; keep failed runs in the report.
- Use a fresh output folder under `work/` when prompts or code change.
- Interleave the configurations you compare: API latency drifts by 2× over an hour.
- Browser fixtures are not the capture card, the Pico or a phone. Re-measure there before claiming hardware numbers.
