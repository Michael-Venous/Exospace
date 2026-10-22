#!/usr/bin/env bash
# Talk to the agent on this laptop: hold Home, speak, let go (bench/arch/talk.py). Ctrl+C stops.
# What it prints goes to ~/exo-bench/talk-logs/console-<time>.txt as well (latest.txt points at the newest).
mkdir -p ~/exo-bench/talk-logs
log=~/exo-bench/talk-logs/console-$(date +%Y%m%d-%H%M%S).txt
ln -sf "$log" ~/exo-bench/talk-logs/latest.txt
cd ~/exo-bench/Exospace && PYTHONPATH=. ~/exo-bench/.venv/bin/python -u ~/.cache/exo-bench/withkeys.py bench.arch.talk \
  2>&1 | tee "$log"
