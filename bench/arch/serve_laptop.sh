#!/usr/bin/env bash
# The Qwen queue's agent server on the laptop rig (EXO_TARGET=laptop), run ON the laptop, with a front end:
#   serve_laptop.sh          hold Home anywhere to talk (bench.arch.home_talk: system mic, spoken answers)
#   serve_laptop.sh --page   the voice page instead: open http://127.0.0.1:8780, hold the orb or Space
# The Pi keeps its own path: python -m bench.arch.serve_agent (EXO_TARGET=pi) and brain.voice.layer.
set -euo pipefail
script_path=$(readlink -f -- "${BASH_SOURCE[0]}")  # through a symlink too
root=$(cd -- "$(dirname -- "$script_path")/../.." && pwd)
cd "$root"
export PYTHONPATH=. EXO_TARGET=laptop
python="$HOME/exo-bench/.venv/bin/python"
keys="$HOME/.cache/exo-bench/withkeys.py"
mkdir -p "$HOME/exo-bench/talk-logs"
log="$HOME/exo-bench/talk-logs/page-$(date +%Y%m%d-%H%M%S).txt"
ln -sf "$log" "$HOME/exo-bench/talk-logs/latest.txt"
"$python" -u "$keys" bench.arch.serve_agent > >(tee "$log") 2>&1 &
agent=$!
trap 'kill "$agent" 2>/dev/null; wait "$agent" 2>/dev/null' EXIT
for _ in $(seq 1 60); do  # the agent takes a few seconds to start its laptop target
  curl -s -o /dev/null http://127.0.0.1:8770/ && break
  kill -0 "$agent" 2>/dev/null || exit 1
  sleep 0.5
done
if [ "${1:-}" = "--page" ]; then
  "$python" -u "$keys" brain.voice.layer --agent http://127.0.0.1:8770
else
  "$python" -u "$keys" bench.arch.home_talk
fi
