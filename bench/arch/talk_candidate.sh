#!/usr/bin/env bash
# Opt-in launcher for the measured bench candidate. Run ON the configured laptop.
set -euo pipefail
script_path=$(readlink -f -- "${BASH_SOURCE[0]}")
task_root=$(cd -- "$(dirname -- "$script_path")/../.." && pwd)
mkdir -p "$HOME/exo-bench/talk-logs"
log="$HOME/exo-bench/talk-logs/candidate-$(date +%Y%m%d-%H%M%S).txt"
ln -sf "$log" "$HOME/exo-bench/talk-logs/latest.txt"
cd "$task_root"
export PYTHONPATH=.
export EXO_RELIABLE=1
"$HOME/exo-bench/.venv/bin/python" -u "$HOME/.cache/exo-bench/withkeys.py" bench.arch.talk 2>&1 | tee "$log"
