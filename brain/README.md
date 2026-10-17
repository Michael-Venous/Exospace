# Brain (Pi 5 runtime)

Pi-side runtime: screen capture, Pico HID client, deterministic controller, agent loop. It can run on the MacBook during development.

| Module | Owner | Responsibility |
|---|---|---|
| `capture.py` | Hardware | Frames from the capture card, no-signal detection, active-rect detection |
| `hid.py` | Hardware | UART client for the Pico protocol (PLAN.md §5.1) |
| `controller.py` | Agent | Runs actions (§5.3), coordinate mapping (§5.2), settle detection, cancellation (§5.6) |
| `agent/` | Agent | Planner, executor loop, memory (§5.5), verification, recovery |
| `skills/` | Agent | Per-OS keyboard-shortcut skills |
| `voice/` | Demo | Push-to-talk, speech-to-text, spoken status |
