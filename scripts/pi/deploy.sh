#!/bin/sh
# Deploy this checkout to the Pi over SSH (run on the laptop).
#   scripts/pi/deploy.sh                 # code + venv + user services
#   PI=mike@host DEST=Exospace scripts/pi/deploy.sh
# Secrets are never copied: put .env on the Pi yourself (chmod 600).
set -eu
PI=${PI:-mike@100.76.219.77}
DEST=${DEST:-Exospace}
cd "$(dirname "$0")/../.."

echo "Backing up $PI:$DEST (without its venv) to $DEST.backup.tgz"
ssh "$PI" "if [ -d '$DEST' ]; then tar czf '$DEST.backup.tgz' --exclude='$DEST/.venv' '$DEST'; fi"

echo "Syncing code"
rsync -az --delete \
  --exclude .git --exclude .venv --exclude .env --exclude work --exclude session.json \
  --exclude __pycache__ --exclude '*.egg-info' --exclude firmware/pico/lib \
  ./ "$PI:$DEST/"

ssh "$PI" "cd '$DEST' && sh scripts/pi/setup.sh"
