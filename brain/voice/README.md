# Voice

**Owner:** Demo

## Voice layer (`layer.py`, the product path)

The voice layer from PLAN.md §11: ears and mouth on top of the agent, never its hands.

```sh
.venv/bin/python -m pip install -e '.[voice]'
.venv/bin/python -m brain --display 8770                     # the agent, with its display server
.venv/bin/python -m brain.voice.layer --agent http://127.0.0.1:8770

# or both in one process: arguments the layer doesn't know go to the agent
.venv/bin/python -m brain.voice.layer --backend local --os linux --display 8770
```

Open http://127.0.0.1:8780 and press **Talk** (or Space, so a single accessibility switch works).
- Each pause sends that segment to the agent right away.
- Pressing Talk while the agent is busy stops it first, so what you say next is a correction.
- The page speaks the agent's questions (including "This can't be undone… Should I go ahead?"), plan stages and
  results, holds narration while you talk, and speaks it afterwards.
- Needs `DEEPGRAM_API_KEY`; the server relays the audio, so the key never reaches the browser.
- Browsers only allow the microphone on localhost or https, and both servers listen on localhost: open the page
  in Chromium on the Pi itself, with a USB mic and speaker (the Pi 5 has no headphone jack).

Measured through the real agent on the mail fixture: "open Janet's email" done in 1.8 s; a remark gets
"Noted." in 0.7 s with no input; "delete all of these emails" asks for confirmation in 0.9 s; "no" declines in 0.8 s.

## Laptop launcher demo (`start.sh`, older)

A standalone page at :8766: speak, Deepgram transcribes, Jev picks the app name and the OS launcher opens it.
It types through a Pico plugged straight into the laptop, running the laptop-only firmware in `pico/` (keyboard
commands over a second USB serial port, since a laptop has no UART). Needs `TYPESAFE_API_KEY` and
`DEEPGRAM_API_KEY`. It does not use the agent; use the voice layer above for the demo.
