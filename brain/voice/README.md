# Voice

**Owner:** Demo

- Push-to-talk on the GPIO button; pressing it during execution means barge-in, which triggers the shared cancel path (PLAN.md §5.6).
- Speech-to-text for requests and corrections.
- Short spoken status lines ("Opening Calendar…").
- The Pi 5 has no headphone jack, so use a USB headset or a USB mic plus a USB speaker.

## Laptop demo

`./start.sh` opens a white page with a bubble: tap it, speak, and Deepgram's live transcript shows underneath. Jev picks the app and the launcher opens it. Needs `TYPESAFE_API_KEY` and `DEEPGRAM_API_KEY` in the repo's `.env`. Under the transcript, a log shows what each command did; **Show details** adds Jev's choice and confidence and each on-screen check with its score and time.

The Pico does the typing, plugged straight into the laptop. Copy `pico/boot.py` and `pico/code.py` onto the `CIRCUITPY` drive, then unplug and replug it. This laptop-only firmware takes keyboard commands over a second USB serial port, because the laptop has no UART to reach [`firmware/pico/`](../../firmware/pico/), the real firmware.

## Voice layer (the real product path)

`layer.py` is the voice layer from PLAN.md §14: ears and mouth on top of the agent, never its hands.

```sh
.venv/bin/python -m pip install -e '.[voice]'
.venv/bin/python -m brain --display 8770            # the agent, with its display server
.venv/bin/python -m brain.voice.layer --agent http://127.0.0.1:8770
```

Open http://127.0.0.1:8780, press **Talk** (or Space, so a single accessibility switch works) and
speak. Each pause sends that segment to the agent right away; pressing Talk while the agent is busy
stops it first, so what you say next is a correction. The page speaks the agent's questions
(including "This can't be undone… Should I go ahead?"), plan stages and results, and stops talking
the moment you do. Needs `DEEPGRAM_API_KEY`; the key stays on the server.

Measured through the real agent on the mail fixture: "open Janet's email" done in 1.8 s; a remark
"Noted." in 0.7 s with no input; "delete all of these emails" asked for confirmation in 0.9 s;
"no" declined in 0.8 s.

