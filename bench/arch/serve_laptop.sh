#!/usr/bin/env bash
# The voice page driving the Qwen queue on the laptop rig. Run ON the laptop, then open http://127.0.0.1:8780 and
# hold the orb (or Space) to talk. The Pi keeps its own path: python -m bench.arch.serve_agent (EXO_TARGET=pi).
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
"$python" -u "$keys" brain.voice.layer --agent http://127.0.0.1:8770
