# Exospace — agent instructions

Follow [CLAUDE.md](CLAUDE.md); the same rules apply to every coding agent. Then read [HANDOFF.md](HANDOFF.md) for
the current state, the Pi rig and what's next.

- **No git:** no commits, pushes or pulls unless the user asks. Push to `origin` is disabled on purpose; never
  re-enable it.
- **Never copy `.env` to the Pi**, and never print or commit its keys.
- **Check the Pi agent is idle before restarting a service:** the user may be mid-run.
