# Exospace — Team Plan (shared context for all agents)

> Read this fully before writing code. If your task conflicts with this file, stop and ask your human.
> Contracts in §5 are shared across teammates: **do not change them without telling the team.**

## 1. What we're building
A portable box that plugs into a computer or phone over one USB-C cable, **sees the screen** (HDMI capture) and **controls it** (pretends to be a USB keyboard/mouse/touchscreen). The user presses a button and speaks a goal; the box completes it. **Nothing is installed on the target.** It keeps memory, so you can unplug, move to another device, and continue the same task.

**Pitch:** "Plug it into any computer or phone, say what you want, it does it — and remembers you across devices."

**For:** people with limited hand mobility who can see the screen and speak (spinal cord injury, ALS, MS, tremor, arthritis, RSI). They already own an assistive switch; one button (or the Space bar) drives everything. Low-vision users are second. Speech-to-text still struggles with dysarthria, so say "designed for limited hand mobility," never "validated."

**Event:** SwampHacks XII, Oct 16–18, 2026. Team of 4. Tracks: Hardware (primary), AI, Overall, MLH Gemini.

## 2. How we're different (keep every decision serving these)
1. **Speed** — measured per step and per task. Prior art (KaiVM) works but is slow.
2. **Cross-device continuity** — hot-swap computer A → computer B → phone with automatic adaptation and retained context.
3. **Voice you can interrupt** — barge in, correct, continue.
4. **Recovery and safety** — a moved window or popup doesn't derail it; nothing irreversible happens without a spoken "yes".

We *complement* head mice and switches: they replace the mouse; we replace the steps.

## 3. Non-goals (do NOT build)
Scheduling, a large dashboard, fine-tuning, fully-offline claims, per-OS or per-device prompt hints and special cases, hard-coded command or phrase rules. These overlap with prior art, eat time, or break universality.

Firmware/BIOS screens are not a non-goal, but there is no firmware-specific logic for now: every device gets the same general path. We may return to it if Jev control generalizes to screens without a mouse.

## 4. Architecture
```
            ┌──────────── Pi 5 ("brain") — own power, never reboots ────────────┐
 user ─────►│ voice page (Talk / Space) + mic + speaker   memory (session.json) │◄──► hosted APIs
 target ───►│ capture card (/dev/video*)   race executor + agent loop + controller │   (Jev, Gemini or z.ai, Deepgram)
 screen     └───────────────┬───────────────────────────────────────────────────┘
                            │ UART JSON lines (§5.1)
                            ▼
                 Pico ("hands") ── USB via hub ──► target sees keyboard + abs mouse + touchscreen

 target ◄═ one USB-C ═► USB-C hub ─HDMI─► capture card ─USB─► Pi 5
                                └─USB-A─► Pico
```

**Layers (top → bottom):**
- **Voice layer** (`brain/voice/`): decides *what* the user wants, never *how*; never writes HID (§11).
- **Race executor** (`brain/agent/race.py`): every request starts Jev triage and a Jev + language-model plan at once (§10).
- **Agent loop + controller** (deterministic Python): the single HID writer; runs actions, skills, typing, settle detection, cancellation, coordinate mapping.
- **Pico firmware**: dumb, fast HID. No logic beyond the protocol.

## 5. Contracts

### 5.1 Pi ↔ Pico UART
- Wiring: Pi GPIO14/TX (pin 8) → Pico GP1/RX (pin 2); Pi GPIO15/RX (pin 10) ← Pico GP0/TX (pin 1); GND (Pi pin 6 ↔ Pico pin 3). Pico powered from Pi 5V (pin 2) → **1N5817 diode** → Pico VSYS (pin 39) so it survives target swaps.
- Pi: enable UART (`dtparam=uart0=on`, disable serial console) → `/dev/ttyAMA0`, 115200 baud.
- Newline-delimited JSON. Every command has an `id`; Pico replies `{"id":N,"ok":true}` or `{"id":N,"ok":false,"err":"..."}`.

| op | fields | notes |
|---|---|---|
| `ping` | — | health check; reply also carries `"mounted": bool` (USB host attached) |
| `key` | `keys: ["CTRL","L"]` | press together, then release |
| `type` | `text` (≤32 chars) | controller chunks long text so cancel is responsive |
| `move` | `x,y` (0–32767) | absolute mouse |
| `click` | `x,y,button: "left"/"right", count: 1/2` | absolute |
| `scroll` | `dy` (± wheel notches) | the controller caps one action at 15 |
| `tap` | `x,y` (0–32767) | touchscreen (Android) |
| `swipe` | `x1,y1,x2,y2,ms` | touchscreen |
| `release_all` | — | **must** release every key/button/touch |

- Pico → Pi unsolicited events: `{"ev":"mounted"}`, `{"ev":"unmounted"}` (USB host attach/detach), sent at once. Nothing is released toward a missing host: a new host starts with nothing pressed.
- HID devices: keyboard + absolute mouse (0–32767) + one-contact multitouch digitizer (so Android binds a touchscreen). CircuitPython; GP22 to GND at power-up = demo mode (HID only, boot-protocol keyboard), else maintenance mode (drive + console visible).

### 5.2 Coordinates
Model output (any space) → **normalized [0,1] within the active screen rect** of the captured frame → Pico 0–32767. Active rect is detected automatically (phone mirror = portrait image with black bars). **Never hard-code per-device calibration.**

### 5.3 Action schema (executor → controller)
```json
{"type":"click","x":0.42,"y":0.13}            // normalized in active rect
{"type":"double_click"|"right_click", "x":..,"y":..}
{"type":"tap"|"swipe", ...}                    // android
{"type":"type_text","text":"..."}
{"type":"hotkey","keys":["CMD","SPACE"]}
{"type":"skill","name":"open_app","args":{"name":"Notes"}}
{"type":"skill","name":"hover","args":{"x":..,"y":..}}   // pointer move, no button
{"type":"skill","name":"select","args":{"x":..,"y":..}}  // triple click: selects the line or paragraph there
{"type":"skill","name":"wait"}                 // until the screen changes, at most 10 s; sends no input
{"type":"scroll","dy":-3}                      // wheel notches
{"type":"remember","key":"...","value":...}
{"type":"done"} | {"type":"ask_user","question":"..."} | {"type":"fail","reason":"..."}
```
`wait` is offered to every executor; race plans it as its own stage (measured: a page that loads for 6 s, 2/2 passed with one click each).

### 5.4 Device profile (set on every connect)
```json
{"os":"windows|macos|linux|chromeos|android|ios|unknown","rect":[x0,y0,x1,y1],"input":"mouse|touch","connected_at":ts}
```
OS is detected **from the first frames** (USB can't reliably tell us). It is a label for the models plus the input mode (`touch` for Android, else `mouse`); no prompt or controller logic depends on it otherwise. `unknown` (boot or firmware screens, a failed classification) still works as keyboard + mouse. `--os` overrides detection.

### 5.5 Memory (`session.json`, persists across swaps)
```json
{"transcript":[...], "facts":{"event":{"title":"","date":"","time":"","location":""}},
 "task":{"goal":"","subtasks":[],"done":[]}, "prefs":{}}
```
The planner gets memory + device profile on every call.

### 5.6 Cancellation — ONE path for everything
Triggers: Pico `unmounted` · capture no-signal/black > 300 ms · Talk/STOP during execution (barge-in) · 3 failed verifications.
Effect: Pico `release_all` → drop action queue → (if disconnect) wait for new device → build profile → planner resumes from memory.
Resume: after a disconnect, restart or lost video, the run waits up to 180 s for live video, re-detects the profile if the connection changed and continues the same goal on the new screen.

### 5.7 Benchmark log (`bench/results.csv`)
`run_id, config, device, task_id, step, t_capture, t_upload, t_model, t_execute, t_settle, success, total_s`

## 6. Key behaviors
- **Settle detection, no fixed sleeps:** downscaled grayscale frame diff below threshold (timeout 2.5 s).
- **Don't call models every frame.** Call after settle, or on a change event.
- **Verify after meaningful actions** (not every keystroke).
- **Irreversible actions need spoken confirmation**, bound to that request and device.
- **Narrate intent and results**, not keystrokes ("Opening Janet's email… done").

## 7. Owners
| Role | Owns | Repo dirs |
|---|---|---|
| **Hardware** | Hub/capture/Pico firmware, UART, hot-swap events, phone validation, Pi deploy | `firmware/`, `brain/hid.py`, `brain/capture.py`, `scripts/pi/` |
| **Agent** | Executors, loop, memory, verification, recovery, skills | `brain/agent/`, `brain/skills/` |
| **Models** | Latency, model comparisons, **benchmark** | `bench/` |
| **Demo** | Voice, touch display, demo script, pitch, Devpost | `brain/voice/`, `brain/display/`, `docs/` |

Python 3.11 on the Pi. Secrets in `.env` (never commit). The brain runs on a laptop during development (`--backend browser`); Pi for the demo.

`main` must always run. Tag every working demo state (`fallback-v1` is the first). Cut scope rather than pivot.

## 8. Pitch honesty
Cite KaiVM and other agent KVMs as prior work. Say "hosted model APIs," not "offline." Say "designed for limited hand mobility," not "validated." Show real numbers only.

## 9. Code quality
AI judges read the repo: keep code clean, typed, small modules, a real README (architecture, how to run, benchmark results). No dead code, no committed secrets.

## 10. Model decision (2026-10-17, final)
**Race + steps-aware completion**, measured in [`bench/arch/README.md`](bench/arch/README.md): 89% end-to-end vs 53% for the previous runtime, 0 vs 5 unsafe actions (paired sign test p = 0.00002, two scenarios incl. a held-out one).
- Race is the default executor with either language-model key. **z.ai and Gemini are interchangeable** (user decision): `ZAI_API_KEY`, when set, serves every executor (GLM-4.6V-FlashX), otherwise Gemini 3.8 Flash; each has the same settings (model, thinking, fallback on 503/429, planner model).
- Every request starts **Jev triage** (one call: act_now / plan / no_action, target box, gesture, irreversible?) and **the language model's plan** (low thinking, with screenshot) at the same time.
- A confident Jev act (presence/uniqueness ≥ 0.5, zoom-refined box) wins and Flash's plan is discarded; no_action goes to memory with no input; otherwise Flash's plan runs, each stage grounded and checked by Jev.
- **Irreversible** requests (Jev ≥ 0.5: send, delete, pay, submit, format, BIOS changes, public account actions such as starring or following) never run without a spoken "yes", on both paths. The previous prompt-only rule deleted a folder in 3/3 runs. *(2026-10-18: a correction is judged with the request it corrects, and Jev scores every planner stage too, in the call it already makes: an unapproved send/delete/pay/submit stage is held as the same question. The demo's correction beat sent the reply 4/4 before and sent nothing in 3 segments after, on both planners; `bench/arch/README.md`.)*
- **Completion** is one Jev call judging goal + executed steps + screen (gate 0.3); it alone decides.
- **One commit gate** (`brain/arbiter.py`) is the only path to HID for pipelined voice: one winner per segment, spoken-order commit, generation bump on interruption, stale-screen proposals re-grounded.
- **Fallback:** a 503/429 retries once on the provider's fallback model (Flash-Lite 3.5 / GLM-4.6V). Lite planned worse, so this is degraded mode, not a default.
- **Options, not defaults:** `EXECUTOR=direct` (one call per step; a `done` claim after input is checked on screen), `jev-flash` (planner only), `legacy-dual`.
- **Measured:** race on Gemini, 89% (above). Race on z.ai alone, workstation study, one repeat: 7/10 by the fixture's oracle, 0 unsafe; GLM does not follow the stage schema as well (0/3 vs 1/3 for direct on the calendar task).
- **Rejected, with measurements in `bench/`:** Flash-Lite, Gemini Live and text-only planning as the planner (about 2× the wrong moves, no time saved); regex `direct_command` and recipe grammars; three-model routing (no accuracy gain, ~12% slower); GPT-6 Luna and Claude Haiku 5.5 (unsafe on messy speech). The B200 / self-hosted models are deferred.
- **Removed by user decision (2026-10-17), to keep one general path:** the firmware path (`boot_keys`, arrow-key menus, `bios` re-detection), per-OS prompt hints, the word-list quick start, `TARGET_OS`, and the QEMU and Android-emulator backends. They remain in git history.

## 11. Voice layer
The voice layer sits **on top** of the pipeline: it decides *what* the user wants, never *how*.
It never writes HID. Its only controls, all through the commit gate (§10):

| Control | Meaning |
|---|---|
| `submit_segment(text)` | a finished clause, sent as soon as the speaker pauses |
| `interrupt()` | barge-in / "stop" / correction: the §5.6 cancel path, bumps the generation |
| `confirm(action_id, yes/no)` | answers an irreversible-action prompt; bound to that action and generation, so a late "yes" can't approve anything else |
| `answer(question_id, text)` | replies to a planner `ask_user` |

Built: `brain/voice/layer.py`. Talk = `submit_segment`, talking while busy = `interrupt`, and confirmations/answers are the next utterance (the race executor checks approval with Jev). Later segments of one push-to-talk hold continue the same request ("go to youtube" … "and search lofi").

Output: it subscribes to the agent **event stream** (`brain/events.py`, shared with the touch display) and narrates intent and results. Speech never blocks execution; output stops when the user speaks. A running request is never silent: "On it." after 2.5 s, then progress every 10 s, and deep mode announces itself. The page's **Describe** toggle (for blind users) has Gemini Live (`gemini-3.8-live`) describe the screen after each request, streamed as audio (`brain/voice/describe.py`).
