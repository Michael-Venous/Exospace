# Exospace

> Software runs in user space or kernel space. **Exospace runs outside the machine entirely.**

Exospace is a portable AI assistant that plugs into a computer or phone over one USB-C cable. It **sees the screen** through HDMI capture and **controls it** by acting as a USB keyboard, mouse and touchscreen. Press the button, say what you want, and it does it. Nothing is installed on the target, and it remembers your conversation and task as you move between devices.

Built at SwampHacks XII (October 16–18, 2026).

## How it works

```
 target ◄═ one USB-C ═► USB-C hub ─HDMI─► capture card ─USB─► Pi 5 ("brain") ◄──► hosted model APIs
                                └─USB-A─► Pico ("hands") ◄─UART─┘
```

- **Pi 5 (brain):** captures the screen, listens to the user, keeps memory, runs the agent loop and controller. It has its own power, so swapping targets never reboots it.
- **Pico (hands):** shows up on the target as a keyboard, absolute mouse and touchscreen, and runs commands from the Pi.
- **Models (race architecture):** for every spoken request, Jev 1.14 and Gemini 3.8 Flash start at once. If Jev confidently sees one control to click, it acts in about a second and Flash's plan is thrown away; remarks are remembered without touching anything; everything else uses Flash's plan, grounded and checked step by step by Jev. Anything irreversible (send, delete, pay, BIOS changes) waits for a spoken "yes". Measured: 89% task success vs 53% for a Flash-planner-only agent, with 0 unsafe actions ([bench/arch/README.md](bench/arch/README.md)). No separate slow planner or B200 is required. Isolated local-browser, QEMU (UEFI firmware) and Android emulator backends support testing before the hardware adapter is available.
- **Any screen:** the OS (Windows, macOS, Linux, ChromeOS, Android, iOS, or a BIOS/UEFI screen) is detected from the screen. A run survives restarts, so it can restart a PC into firmware setup and change settings there.
- **Voice:** a push-to-talk page (one button or the Space bar, so a single accessibility switch works) sends what you say to the agent and speaks back its questions and results.

See [PLAN.md](PLAN.md) for the full architecture, shared contracts, owners and checkpoints.

## Repository layout

| Path | What | Owner |
|---|---|---|
| `firmware/pico/` | CircuitPython HID firmware and the UART protocol | Hardware |
| `brain/` | Pi-side runtime: capture, HID client, controller | Hardware / Agent |
| `brain/agent/` | Planner, executor loop, memory, verification, recovery | Agent |
| `brain/skills/` | Per-OS keyboard-shortcut skills | Agent |
| `brain/voice/` | Voice layer (push-to-talk, narration) and the laptop voice demo | Demo |
| `server/` | Model serving on the B200s (deferred) | Models |
| `bench/arch/` | Architecture studies behind every design decision | Models |
| `bench/experiments/` | Earlier executor experiments and benchmarks | Models |
| `scripts/pi/` | Deploy to the Pi, user services, setup checklist | Hardware |
| `docs/` | Hardware bring-up, demo script, Devpost draft | Demo |
| `demo/` | Projector view and demo script | Demo |

## Setup

1. Copy `.env.example` to `.env` and fill in the keys: z.ai or Gemini (interchangeable, either one runs everything), TypeSafe/Jev, and Deepgram for voice.
2. Python 3.11+: `python3 -m venv .venv`, then `.venv/bin/python -m pip install -e '.[voice,browser]'`.
3. Try it without hardware, on a web page in an isolated browser:
   ```sh
   .venv/bin/python -m brain --backend browser --url https://example.com --os linux --display 8770
   .venv/bin/python -m brain.voice.layer --agent http://127.0.0.1:8770   # then open :8780
   ```
4. On the Pi: `scripts/pi/deploy.sh` from the laptop, then `.venv/bin/python -m brain.doctor`
   on the Pi once the capture card and Pico are plugged in ([docs/hardware-bringup.md](docs/hardware-bringup.md)).
5. Tests: `.venv/bin/python -m unittest discover -s tests` (offline, no API calls).

See [brain/README.md](brain/README.md) for the runtime and [docs/demo.md](docs/demo.md) for the demo.

## Prior work

Agent-driven KVM devices already exist, including [KaiVM](https://github.com/hardrave/KaiVM), Sipeed NanoKVM-Go and kvm-pilot. Exospace focuses on speed (with measured results), continuity across devices with automatic adaptation, and voice control you can interrupt.
