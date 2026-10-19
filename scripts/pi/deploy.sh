#!/bin/sh
# Deploy this checkout to the Pi over SSH (run on the laptop).
#   scripts/pi/deploy.sh                 # code + venv + user services
#   PI=mike@host DEST=Exospace scripts/pi/deploy.sh
#   OS=linux scripts/pi/deploy.sh         # pin the target's OS (a black, empty screen can't be classified)
# Secrets are never copied: put .env on the Pi yourself (chmod 600). Nor is replay.json, the
# replays the Pi learned on its own target.
set -eu
PI=${PI:-mike@100.76.219.77}
DEST=${DEST:-Exospace}
cd "$(dirname "$0")/../.."

echo "Backing up $PI:$DEST (without its venv) to $DEST.backup.tgz"
# -h: $DEST may be a symlink (to /srv/exospace), and a backup of only the link restores nothing.
ssh "$PI" "if [ -d '$DEST' ]; then tar czhf '$DEST.backup.tgz' --exclude='$DEST/.venv' '$DEST'; fi"

echo "Syncing code"
rsync -az --delete \
  --exclude .git --exclude .venv --exclude .env --exclude work --exclude session.json \
  --exclude __pycache__ --exclude '*.egg-info' --exclude firmware/pico/lib \
  --exclude bench/results.csv --exclude bench/model_calls.jsonl --exclude replay.json \
  ./ "$PI:$DEST/"

ssh "$PI" "cd '$DEST' && OS='${OS:-}' sh scripts/pi/setup.sh"
