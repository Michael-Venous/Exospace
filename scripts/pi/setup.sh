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
done
systemctl --user daemon-reload 2>/dev/null || true
echo "Installed user services: exospace-agent, exospace-voice (not started)."

echo "Running offline tests"
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -n 3

echo "Hardware check"
.venv/bin/python -m brain.doctor || true

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
