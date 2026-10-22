#!/usr/bin/env bash
# Talk to the agent on this laptop: hold Home, speak, let go (bench/arch/talk.py). Ctrl+C stops.
cd ~/exo-bench/Exospace && PYTHONPATH=. exec ~/exo-bench/.venv/bin/python ~/.cache/exo-bench/withkeys.py bench.arch.talk
