#!/bin/sh
# Prepare the Pi (runs on the Pi, from the repo root; no sudo needed).
# Creates the venv, installs Exospace, installs user services, runs the offline
# tests and the hardware check, then lists the one-time steps that need sudo.
set -eu
cd "$(dirname "$0")/../.."
ROOT=$(pwd)

[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -e '.[voice]'

UNITS="$HOME/.config/systemd/user"
mkdir -p "$UNITS"
for unit in exospace-agent exospace-voice; do
  sed "s#@ROOT@#$ROOT#g" "scripts/pi/$unit.service" > "$UNITS/$unit.service"
  # OS=linux|macos|windows|...: pin the target's OS for the agent (detection can't read a black, empty screen)
  [ -n "${OS:-}" ] && [ "$unit" = exospace-agent ] && sed -i "s#-m brain #-m brain --os $OS #" "$UNITS/$unit.service"
done
systemctl --user daemon-reload 2>/dev/null || true
# Running services keep the old code (and hold the Pico UART) until restarted.
ACTIVE=""
for unit in exospace-agent exospace-voice; do
  systemctl --user is-active --quiet "$unit" 2>/dev/null && ACTIVE="$ACTIVE $unit"
done
[ -n "$ACTIVE" ] && systemctl --user stop $ACTIVE
echo "Installed user services: exospace-agent, exospace-voice."

echo "Running offline tests"
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -n 3

echo "Hardware check"
.venv/bin/python -m brain.doctor || true

if [ -n "$ACTIVE" ]; then
  systemctl --user start $ACTIVE && echo "Restarted:$ACTIVE"
fi
echo "Logs: journalctl --user -u exospace-agent -u exospace-voice -f"

cat <<'EOF'

One-time steps that need sudo (run on the Pi):
  1. Header UART for the Pico (GPIO14/15 -> /dev/ttyAMA0):
       sudo raspi-config nonint do_serial_cons 1   # serial console off
       sudo raspi-config nonint do_serial_hw 0     # UART hardware on
       echo 'dtparam=uart0=on' | sudo tee -a /boot/firmware/config.txt
  2. Access to the capture card and UART:  sudo usermod -aG video,dialout "$USER"
  3. Start services at boot without a login:  sudo loginctl enable-linger "$USER"
  Then reboot, copy .env (chmod 600), and run:
       .venv/bin/python -m brain.doctor
       systemctl --user enable --now exospace-agent exospace-voice
EOF
