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
- **Models (race architecture):** for every spoken request, Jev 1.14 and the language model (Gemini 3.8 Flash or z.ai GLM, interchangeable) start at once. If Jev confidently sees one control to click, it can act without waiting for the planner; remarks are remembered without touching anything; everything else uses Flash's plan, grounded and checked step by step by Jev. Anything irreversible (send, delete, pay, BIOS changes) waits for a spoken "yes". Measured on synthetic browser scenarios: 89% task success vs 53% for a Flash-planner-only agent, with 0 unsafe actions ([bench/arch/README.md](bench/arch/README.md)). No separate slow planner or B200 is required. An isolated browser backend supports testing before the hardware adapter is available.
- **Any screen, one path:** the same prompts and controller drive every device; the detected OS only picks touch or mouse input. A run survives a restart, replug or blank screen and continues the same goal.
- **Voice:** a push-to-talk page (one button or the Space bar, so a single accessibility switch works) sends what you say to the agent and speaks back its questions and results.

## Repository layout

| Path | What | Owner |
|---|---|---|
| `brain/` | Pi-side runtime: capture, HID client, controller, agent loop ([README](brain/README.md)) | Hardware / Agent |
| `brain/agent/` | Race executor, triage, planner, memory ([README](brain/agent/README.md)) | Agent |
| `brain/skills/` | Keyboard skills: `open_app`, `hover`, `wait` | Agent |
| `brain/voice/` | Voice layer: push-to-talk page, Deepgram relay, narration ([README](brain/voice/README.md)) | Demo |
| `brain/display/` | Touch display and projector page | Demo |
| `firmware/pico/` | CircuitPython HID firmware for the UART protocol ([README](firmware/pico/README.md)) | Hardware |
| `scripts/pi/` | Deploy to the Pi, setup, user services | Hardware |
| `bench/` | Architecture studies and benchmarks behind every claim ([README](bench/README.md)) | Models |
| `tests/` | Offline tests (no API calls) | All |

## Setup

1. Copy `.env.example` to `.env` and fill in the keys: z.ai or Gemini (interchangeable, either one runs everything), TypeSafe/Jev, and Deepgram for voice.
2. Python 3.11+: `python3 -m venv .venv`, then `.venv/bin/python -m pip install -e '.[voice,browser]'`.
3. Try it without hardware, on a web page in an isolated browser:
   ```sh
   .venv/bin/python -m brain --backend browser --url https://example.com --os linux --display 8770
   .venv/bin/python -m brain.voice.layer --agent http://127.0.0.1:8770   # then open :8780
   ```
4. On the Pi: `scripts/pi/deploy.sh` from the laptop, then follow [docs/hardware-bringup.md](docs/hardware-bringup.md).
5. Tests: `.venv/bin/python -m unittest discover -s tests` (offline, no API calls).

## Latency and regression checks

Run `.venv/bin/python -m bench.demo_smoke` for the compact routing, cancellation and voice-browser
regression gate. Run the full suite before deploying. Neither suite measures live ASR or API latency.
Step logs separate model, input, settle and capture/prepare time; raw timings live in `bench/results.csv`.
`CAPTURE_FPS` defaults to 60; `brain.doctor` reports actual throughput rather than trusting the requested rate.
The Pi prepares a newly connected target while idle so identification need not delay its first command.

The current physical rig is **not near-real-time**: a combined calculator launch/calculation took 24.6 s
on October 18, mostly in model calls. See [HANDOFF.md](HANDOFF.md) for deployment state and live failures.
Voice submits finalized segments while speech may continue; partial-transcript planning is not implemented.

## Documentation

| Read | For |
|---|---|
| [PLAN.md](PLAN.md) | Goals, non-goals, architecture, the shared contracts, owners, the model decision |
| [HANDOFF.md](HANDOFF.md) | Current status, known issues, what is waiting on hardware |
| [brain/README.md](brain/README.md) | Running the runtime, how a request flows, modules, touch display |
| [docs/hardware-bringup.md](docs/hardware-bringup.md) | Wiring, Pi setup, bring-up order, pre-demo checklist |
| [docs/demo.md](docs/demo.md) | The demo script and fallbacks |
| [docs/devpost.md](docs/devpost.md) | Devpost draft |
| [bench/arch/README.md](bench/arch/README.md) | The measurements behind the architecture |

## Prior work

Agent-driven KVM devices already exist, including [KaiVM](https://github.com/hardrave/KaiVM), Sipeed NanoKVM-Go and kvm-pilot. Exospace focuses on speed (with measured results), continuity across devices with automatic adaptation, and voice control you can interrupt.
