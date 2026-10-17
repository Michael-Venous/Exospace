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
- **Models:** with a z.ai key, GLM-4.6V-FlashX chooses and aims each step's actions in one hosted call. Without one, Jev 1.14 grounds screen targets and Gemini 3.8 Flash chooses actions, writes content and handles recovery. No separate slow planner or B200 is required. Isolated local-browser, QEMU (UEFI firmware) and Android emulator backends support testing before the hardware adapter is available.
- **Any screen:** the OS (Windows, macOS, Linux, ChromeOS, Android, iOS, or a BIOS/UEFI screen) is detected from the screen. A run survives restarts, so it can restart a PC into firmware setup and change settings there.

See [PLAN.md](PLAN.md) for the full architecture, shared contracts, owners and checkpoints.

## Repository layout

| Path | What | Owner |
|---|---|---|
| `firmware/pico/` | CircuitPython HID firmware and the UART protocol | Hardware |
| `brain/` | Pi-side runtime: capture, HID client, controller | Hardware / Agent |
| `brain/agent/` | Planner, executor loop, memory, verification, recovery | Agent |
| `brain/skills/` | Per-OS keyboard-shortcut skills | Agent |
| `brain/voice/` | Push-to-talk, speech-to-text, spoken status | Demo |
| `server/` | Model serving on the B200s | Models |
| `bench/` | Benchmark tasks, runner and results | Models |
| `demo/` | Projector view and demo script | Demo |

## Setup

1. Copy `.env.example` to `.env` and fill in the values.
2. Python 3.11+: `python3 -m venv .venv`, then `.venv/bin/python -m pip install -e .`.
3. Run `.venv/bin/python -m brain --backend pi` once capture and Pico are connected.
   See [brain/README.md](brain/README.md) for local development, model configuration,
   cancellation, offline tests and the live latency probe.

## Prior work

Agent-driven KVM devices already exist, including [KaiVM](https://github.com/hardrave/KaiVM), Sipeed NanoKVM-Go and kvm-pilot. Exospace focuses on speed (with measured results), continuity across devices with automatic adaptation, and voice control you can interrupt.
