#!/usr/bin/env bash
# Talk to the agent on this laptop: hold Home, speak, let go (bench/arch/talk.py). Ctrl+C stops.
# What it prints also goes to ~/exo-bench/talk-logs/latest.txt.
mkdir -p ~/exo-bench/talk-logs
cd ~/exo-bench/Exospace && PYTHONPATH=. ~/exo-bench/.venv/bin/python -u ~/.cache/exo-bench/withkeys.py bench.arch.talk \
  2>&1 | tee ~/exo-bench/talk-logs/latest.txt
