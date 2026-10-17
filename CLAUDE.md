# Exospace — agent instructions

1. **Read [PLAN.md](PLAN.md) before doing anything.** It defines the project, what not to build, the architecture, owners and checkpoints.
2. **Contracts in PLAN.md §5 are shared across four teammates.** Do not change the UART protocol, coordinate convention, action schema, device profile, memory format, cancellation path or benchmark log format without your human's explicit approval. If one changes, update PLAN.md in the same commit.
3. **Stay in your owner's folders** (PLAN.md §7). Touching another area means coordinating with its owner first.
4. **`main` must always run.** Keep each change small and working. Never break the tagged fallback demo.
5. **Measure before optimizing.** Speed claims need numbers in `bench/results.csv`.
6. **Don't build the non-goals** in PLAN.md §3.
7. **Code quality is judged:** keep it clean, typed and in small modules. No dead code. Never commit secrets (use `.env`).
