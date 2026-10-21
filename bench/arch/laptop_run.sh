#!/usr/bin/env bash
# Start a laptop test run from the workstation: sync the code, run the tests ON the laptop (no SSH in the loop),
# stream each result line back, then copy the run folders back into work/arch/laptop/.
#   bench/arch/laptop_run.sh --tests 1 3 --repeats 3
set -euo pipefail
HOST=mike@100.103.48.85  # the test laptop (Assistant hub MACHINES.md)
REMOTE=exo-bench/Exospace
cd "$(dirname "$0")/../.."
rsync -a --delete --exclude .venv --exclude work --exclude .git --exclude '*.egg-info' --exclude __pycache__ \
  ./ "$HOST:$REMOTE/"
ssh -o BatchMode=yes "$HOST" "cd $REMOTE && PYTHONPATH=. ~/exo-bench/.venv/bin/python ~/.cache/exo-bench/withkeys.py \
  bench.arch.laptop_tests $*" | grep --line-buffered -E '^\{"test"|Traceback|Error' || true
rsync -a "$HOST:$REMOTE/work/arch/laptop/" work/arch/laptop-native/
