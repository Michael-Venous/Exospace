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

## Typed-goal prototype

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
cp -n .env.example .env  # preserve existing credentials; set both API keys locally
.venv/bin/python -m brain --backend pi
```

For a local development target, install `-e '.[local]'` and run
`--backend local --os macos` (or `windows`/`linux`). This sends real keyboard and
mouse input to the local desktop after a two-second focus delay. macOS needs
Screen Recording and Accessibility permissions; PyAutoGUI is not a general
Wayland input backend.

Ctrl+C during a task cancels and releases input. The next line corrects the
previous goal; `/new GOAL` starts a different task. Ctrl+C at the prompt exits.
A run survives a target restart, replug or blank screen: it waits up to 180 s for
video, re-detects the device and continues on the new screen. Voice is not wired into
this prompt yet. `--max-steps` defaults to 30.

| Module | Responsibility |
|---|---|
| `keys.py` | Canonical keyboard names |
| `monitor.py` | Connection generation and disconnect cancellation |
| `imaging.py` | Active-screen crops and JPEG preparation |
| `telemetry.py` | Existing benchmark CSV contract |
| `device.py` | Screenshot-based OS identification, or explicit `--os` |
| `app.py` | Typed requests, corrections, lifecycle cleanup |
| `agent/model.py` | Bounded Gemini requests and per-model timing log |
| `agent/jev_flash.py` | Flash semantic actions + Jev visual grounding |
| `agent/jev.py` | Image normalization, bounded HTTP transport, presence/uniqueness gates |
| `browser.py` | Isolated screenshot/input development backend |
| `qemu.py` | Virtual PC (QEMU + OVMF UEFI) for firmware and restart tests |
| `android.py` | Android emulator or USB-debugging phone over adb, for phone tests |
| `agent/loop.py` | Single HID writer, verification, recovery and memory |

## Models and latency

With `ZAI_API_KEY` set, the default is `EXECUTOR=direct`: GLM-4.6V-FlashX returns each
step's actions and coordinates in one call (PLAN.md §13). GLM tends to claim `done` from
its own history, so once a run has sent input, a short second call (about 1.7 s per `done`
claim) must confirm that the screenshot shows the goal; if not, the run continues with what
is missing. A step's actions run as one batch, but a click or swipe after the first input
is skipped once the screen has changed (a tapped field opening the keyboard), since its
coordinates came from the old screen. A z.ai call is tried up to three times. Otherwise the default is
**Jev 1.14 + Gemini 3.8 Flash** (`EXECUTOR=jev-flash`).
Set `GEMINI_API_KEY` and `TYPESAFE_API_KEY` in the ignored `.env`. Flash chooses
semantic actions and writes content; Jev supplies pointer locations from the
same screenshot. There is no separate fast/slow language-model routing in this
mode, and no B200 dependency. The old dual-model path is opt-in only.

Jev has an explicit **execute / ask_flash** Choice. It can request Flash when
instructions are unclear, a new plan is needed, or the current screen contradicts
the plan. Its routing Choice is separate from localization confidence. Presence,
uniqueness and Box confidence still gate pointer input; a confident execute vote
cannot override an absent or ambiguous target. Clarity is also measured as
routing evidence. The thresholds are provisional and need held-out calibration.

Explicit single click/tap/hover commands can use Jev without Flash. Other goals
first route through Jev to Flash, which produces bounded semantic stages with
visible preconditions and expected results. Jev checks the previous result,
checks the next precondition, and grounds the next target together in one call.
The controller acknowledges actual execution before the cached plan advances;
stale/discarded actions cannot consume a stage. Corrections and device changes
invalidate the plan. Missing controls or mismatches trigger a bounded Flash replan;
genuine intent ambiguity produces a user question. Flash receives the previous
semantic stage and whether it actually executed, so it can recover without
blindly repeating the same input.

If Jev cannot localize a target, a bounded Flash fallback can supply coordinates
for one visible action. The fallback is allowed only following failed grounding;
its coordinates cannot be reused after the screenshot changes. Ambiguous user
intent still requires clarification. This is separate from optional Argon routing,
which is being considered but is not currently implemented.

Hover is an existing `skill` action backed by UART `move`, and never presses a
button. Semantic dragging maps to the existing `swipe` action. A hover and its
subsequent submenu click always use different screenshots. Shared action/UART
schemas are unchanged. Jev Box remains experimental, and the agent's successful
completion claim should be measured against an independent task outcome.

## Restarts and firmware setup

The `boot_keys` skill arms setup keys (F2, DELETE, ESC...) that the controller taps
every 0.2 s while the target restarts and for 5 s after its video returns. Taps start
only once the Pico sees the USB reset, so a screen that merely blanks gets none, and a
run that ends disarms them. Firmware screens get the `bios` profile. They have no
pointer, so a click presses DOWN/UP until the menu's highlight bar covers the clicked
row; right-click, scroll and swipe are rejected. A boot logo can read as firmware, so a
`bios` profile is detected again once its screen has changed a lot since it was
classified, also between goals. Test without hardware on a QEMU/OVMF virtual PC
(Debian/Ubuntu packages `qemu-system-x86` and `ovmf`); every run starts from fresh
firmware settings:

```sh
.venv/bin/python -m brain --backend qemu
```

Limits: Mac boot keys must be held, which taps cannot do. PCs with Fast Boot ignore USB
keyboards during POST, and a restart that never resets USB (no `unmounted` from the Pico)
gets no taps; use the OS's restart-to-firmware option there. Highlight
detection needs a solid bar, so menus that only recolor text need arrow keys, as do
one-row menus and a UEFI shell (UP/DOWN recall its history).

## Phone testing with an Android emulator

With one emulator (or a phone with USB debugging) listed by `adb devices`, the android
backend stands in for the capture card and Pico: `adb screencap` frames in, `adb shell
input` taps, swipes, keys and ASCII text out. It finds `adb` on `PATH` or in
`~/Android/Sdk/platform-tools`.

```sh
.venv/bin/python -m brain --backend android
```

## Local browser testing without the hardware adapter

The browser backend uses an isolated profile, real screenshots, and pixel clicks
and keyboard input. Models receive no DOM or accessibility tree. It works on
this Wayland laptop without controlling the user's desktop or logged-in browser.

```sh
.venv/bin/python -m pip install -e '.[browser]'
.venv/bin/python -m playwright install chromium
.venv/bin/python -m brain.local_benchmark --runs 3
.venv/bin/python -m brain.local_scenarios --case all
.venv/bin/python -m brain --backend browser --url file:///absolute/path/to/test.html --os linux --headed
```

If using an existing Chrome install, set `BROWSER_EXECUTABLE` to its executable
path instead of downloading Playwright Chromium. The benchmark creates a local
calendar fixture: read details, calculate an updated end time, fill six fields,
dismiss a popup in selected variants, and verify the saved summary. A separate
post-run DOM oracle checks every saved field; the agent itself sees pixels only.
Reports, screenshots, session state and timings go to `work/local-benchmark/`.
These are synthetic browser tasks, not proof of desktop, phone or Pico support.
Microphone, speaker and the hardware adapter are currently unavailable; typed
requests remain the working interaction path.

Model call timing records are stored separately in `MODEL_LOG` so the shared
benchmark CSV stays unchanged. The records contain role, model, run, step,
attempt, elapsed seconds and outcome; no prompts, screenshots or API keys.
`t_upload` in the existing CSV measures image preparation, while the model call
includes actual network upload. Canceled-call timings measure time until the
caller stops waiting, not remote compute termination.

A small opt-in probe sends only a synthetic image and measures complete JSON
response latency, not GUI task success or time to first token:

```sh
.venv/bin/python -m brain.probe --runs 3
FAST_MODEL=gemini-3.8-flash .venv/bin/python -m brain.probe --runs 3
```

Offline tests send no real input and make no model requests:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

## Platform independence and accessibility

The current cross-platform path is screenshot → normalized action → HID. The
planner works in objectives; the executor handles OS-specific visual grounding;
shortcut skills provide deterministic actions where available.

An optional target-side companion could normalize accessibility roles, names,
states, bounds and supported actions across OS APIs. That could improve semantic
control selection, but the current HDMI/USB-HID connection exposes pixels and
input only, not an accessibility tree. A companion would require installation and
permissions on each target, so it is a separate future mode, not a prerequisite
for the no-install hardware demo. No accessibility adapter is implemented here.

Verified documentation (2026-10-17):
- [Gemini 3.5 Flash-Lite](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite)
- [Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)
- [Gemini 4 Argon](https://ai.google.dev/gemini-api/docs/models/gemini-4-argon)
- [Windows UI Automation](https://learn.microsoft.com/en-us/windows/win32/winauto/entry-uiauto-win32)

### Earlier research notes (superseded by current default above)

Jev is TypeSafe's typed decision model, distinct from JEPA/V-JEPA. It selects
among supplied choices, scores a rubric, or estimates a yes/no probability.
The official docs currently list `jev-1.14.0` and `jev-1.13.0`; `jev-latest`
still resolves to 1.13 and `jev-preview` to 1.14. Pin a version for comparisons.
The ordinary model table says text input, but the separate image documentation
explicitly enables experimental images on both versions; Box localization
requires 1.14 and the image feature header. Do not confuse that experimental
path with established visual execution performance.

A candidate architecture is: screenshot parser (or optional accessibility
adapter) → controls with labels/bounds and permitted actions → Jev-style selector
→ existing controller. Argon supplies objectives and generated text and handles
uncertainty or unfamiliar workflows. Include an explicit no-match/escalate choice;
a probability distribution over bad candidates cannot create the missing action.
Jev Box always returns a box even for an absent target, so an independent presence
question and verification are required before using a location. Choice confidence
is a statistic derived from the distribution, not a measured end-to-end success
rate. Calibrate thresholds on held-out GUI tasks rather than copying a number.

The B200 is at UF HiPerGator; the user's friend is arranging API access. First
compare pretrained inference against the Gemini baseline. OmniParser is a
candidate for extracting structured controls from screenshots. AnyJev is an open
candidate for bounded decisions; its repository warns about calibration limits
and its vision backend is still on the roadmap. Neither has been installed or
benchmarked here. TypeSafe documents no customer fine-tuning/LoRA of Jev itself;
custom training would target an open model or a separate downstream selector.

Measure full perception + selection + network + action + verification time, task
success, wrong-action rate and escalation rate across held-out devices/apps.
Train only if a measured bottleneck and suitable examples justify it. Preserve
the Gemini fallback until a candidate wins at comparable task success.

Sources:
- [Jev models and customization](https://docs.typesafe.ai/models)
- [Experimental images](https://docs.typesafe.ai/images)
- [Box localization and absence behavior](https://docs.typesafe.ai/primitives/box)
- [Confidence semantics](https://docs.typesafe.ai/confidence)
- [OmniParser](https://github.com/microsoft/OmniParser)
- [AnyJev](https://github.com/nokia-applied-research/AnyJev)
- [AnyJev roadmap](https://github.com/nokia-applied-research/AnyJev/blob/main/ROADMAP.md)
