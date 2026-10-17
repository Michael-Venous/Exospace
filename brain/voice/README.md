# Voice

**Owner:** Demo

- Push-to-talk on the GPIO button; pressing it during execution means barge-in, which triggers the shared cancel path (PLAN.md §5.6).
- Speech-to-text for requests and corrections.
- Short spoken status lines ("Opening Calendar…").
- The Pi 5 has no headphone jack, so use a USB headset or a USB mic plus a USB speaker.

## Laptop demo

`./start.sh` opens a white page with a bubble: tap it, speak, and Deepgram's live transcript shows underneath. Jev picks the app and the launcher opens it. Needs `TYPESAFE_API_KEY` and `DEEPGRAM_API_KEY` in the repo's `.env`.

The Pico does the typing, plugged straight into the laptop. Copy `pico/boot.py` and `pico/code.py` onto the `CIRCUITPY` drive, then unplug and replug it. This laptop-only firmware takes keyboard commands over a second USB serial port, because the laptop has no UART to reach [`firmware/pico/`](../../firmware/pico/), the real firmware.
