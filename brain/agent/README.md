# Agent

**Owner:** Agent

- **Planner** (frontier model): goal → subtasks. Called on a new request, a device change, or repeated failure.
- **Executor:** screenshot + subtask → next action (PLAN.md §5.3). The baseline is the frontier model; the candidate is UI-TARS on the B200s.
- **Memory:** `session.json` (§5.5), which persists across target swaps.
- **Verification and recovery:** check the fresh screen after meaningful actions, and re-plan after 3 failures.

Full design (orchestration, voice flow, events for the projector, interfaces): [DESIGN.md](DESIGN.md).
