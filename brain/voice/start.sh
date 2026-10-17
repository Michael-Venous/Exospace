#!/bin/sh
# Start the server (exits harmlessly if one is already running), then open the page as its own window.
python3 -I "$(dirname "$0")/voice_tab.py" >/dev/null 2>&1 &
sleep 1
exec google-chrome --app=http://127.0.0.1:8766
