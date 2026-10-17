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
A disconnected target is re-detected before the next request. If the screen blanks
mid-task while the Pico stays connected (display sleep, mode switch), the run waits up
to 10 s and continues the same goal; an unplug ends the run, since the next device may
be another one. `--max-steps` defaults to 30.

| Module | Responsibility |
|---|---|
| `keys.py` | Canonical keyboard names |
| `monitor.py` | Connection generation and disconnect cancellation |
| `imaging.py` | Active-screen crops and JPEG preparation |
| `telemetry.py` | Existing benchmark CSV contract |
| `device.py` | Screenshot-based OS identification, or explicit `--os` |
| `app.py` | Typed requests, corrections, lifecycle cleanup |
| `agent/model.py` | Bounded Gemini and z.ai requests and per-model timing log |
| `agent/executor.py` | `direct` executor: one model call per step; done claims checked on screen |
| `agent/race.py` | Default executor: Jev triage races the planner; irreversible requests need approval |
| `agent/triage.py` | One Jev call: act / plan / no action, target box, irreversible flag; zoom refinement |
| `agent/jev_flash.py` | Planner: Flash semantic stages + Jev grounding; steps-aware completion check |
| `arbiter.py` | Commit gate for the pipelined voice layer (PLAN.md §14) |
| `agent/jev.py` | Image normalization, bounded HTTP transport, presence/uniqueness gates |
| `browser.py` | Isolated screenshot/input development backend |
| `agent/loop.py` | Single HID writer, verification, recovery and memory |

## Models and latency

Current default: **race** (`EXECUTOR=race`): Jev triage and the Jev + Flash planner start
together on every request; a confident Jev click or "no action" wins, otherwise the planner's result is
used. Completion is one Jev call on goal + executed steps + screen. Evidence and numbers:
[`bench/arch/README.md`](../bench/arch/README.md). `EXECUTOR=jev-flash` runs the planner alone.

Option: **direct** (`EXECUTOR=direct`): one vision-model call per step returns the actions
and their coordinates. With `ZAI_API_KEY` set it uses z.ai GLM-4.6V-FlashX (p50 2.5 s per
call over 1,299 calls; 12 s timeout, up to three tries), otherwise Gemini. After the run
has sent input, a `done` claim costs one more call that must see the goal on screen.
z.ai serves only this executor: GLM does not follow jev-flash's stage schema. Not yet
compared with race end to end against independent checks.

Planner details (the race executor's plan path):
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
button. The `wait` skill blocks until the screen changes (at most 10 s) for slow
loads; the direct executor is offered it, race not yet. Semantic dragging maps to the existing `swipe` action. A hover and its
subsequent submenu click always use different screenshots. Shared action/UART
schemas are unchanged. Jev Box remains experimental, and the agent's successful
completion claim should be measured against an independent task outcome.

## Local browser testing without the hardware adapter

The browser backend uses an isolated profile, real screenshots, and pixel clicks
and keyboard input. Models receive no DOM or accessibility tree. It works on
this Wayland laptop without controlling the user's desktop or logged-in browser.

```sh
.venv/bin/python -m pip install -e '.[browser]'
.venv/bin/python -m playwright install chromium
.venv/bin/python -m bench.experiments.local_benchmark --runs 3
.venv/bin/python -m bench.experiments.local_scenarios --case all
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
.venv/bin/python -m bench.experiments.probe --runs 3
FAST_MODEL=gemini-3.8-flash .venv/bin/python -m bench.experiments.probe --runs 3
```

Two opt-in comparisons use real API calls and independent exact-answer oracles:

```sh
# Separate task families for calibration and held-out evaluation. Preserve this order.
.venv/bin/python -m bench.experiments.routing_benchmark prepare
.venv/bin/python -m bench.experiments.routing_benchmark jev
.venv/bin/python -m bench.experiments.routing_benchmark calibration
.venv/bin/python -m bench.experiments.routing_benchmark freeze
.venv/bin/python -m bench.experiments.routing_benchmark heldout
.venv/bin/python -m bench.experiments.routing_benchmark report
# Full short workflows: successive clicks, literal URL typing, and three fields.
.venv/bin/python -m bench.experiments.sequence_benchmark --repeats 3 --output work/sequence-comparison-v2
```

The routing study freezes a hashed 36-task manifest before API calls and fits
thresholds only on calibration outcomes. Held-out calls require a frozen policy.
It compares 3.5 Flash-Lite/low, 3.8 Flash/low and 4 Argon/medium, including Jev
latency in counterfactual routing results. These screenshot reasoning probes do
not establish production reliability or price savings. The separate sequence
suite uses screenshot-only agents, real pixel/keyboard input, and post-run DOM
oracles. Its URL form is a local stand-in, not Chrome's address bar or YouTube.
Reports retain failures; use a new output directory when changing prompts.
Neither benchmark changes `.env` or enables model routing automatically.

Semantic plans may contain zero stages only to request independent Jev completion
verification against the original user goal as well as the proposed evidence.
This avoids accepting a weaker model-written completion condition or inventing
no-op actions after the goal is already done.
The prompt batches a field click with its typing and checks visible values rather
than keyboard focus, which is not reliable evidence in screenshots.

A separate optional Live feasibility probe sends a synthetic screenshot and a
typed request, then validates an inert function call. It does not control the
device, record a microphone, or play audio:

```sh
.venv/bin/python -m bench.experiments.live_probe --runs 3 --output work/live-probe-new
```

Verified on 2026-10-17: `gemini-3.8-live` returned the exact code in 3/3 function
calls in 0.51–0.73 s after sending, plus 0.38–0.55 s for a new connection. Sending
image and text together in `send_client_content` worked. The initial approach
sending a video frame separately followed immediately by a text turn failed all
three probes; do not equate that failed integration with lack of image support.
This establishes a small screenshot/tool-call path, not voice quality or reliable
multi-step execution. Google documents image input, function calling, audio
response mode and no strict structured-output mode for
[3.8 Live](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-live).
A production voice adapter would still need session resumption, interruption,
validated tool dispatch and stale-frame/action cancellation. It is not enabled.

Offline tests send no real input and make no model requests:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

## Touch display and projector view

`--display PORT` serves a local page (`brain/display/`) fed by live agent events (`brain/events.py`):

```sh
.venv/bin/python -m brain --backend browser --url https://www.google.com --headed --display 8770
# open http://127.0.0.1:8770 (on the Pi: Chromium kiosk on the Touch Display 2)
```

- **Main view:** live screen with the click marker, history (actions, Flash/Jev reasoning, model time), STOP
  (same cancel path as barge-in), saved-task buttons and a typed-goal box. **Details:** the last Jev check's
  present / unique / clear / box scores against the `.env` pass marks.
- **Taps:** "Tap = click" clicks that spot on the target through the normal controller. "Tap = this" marks a point,
  and the next goal is sent to Flash with it ("delete this").
- **For voice:** post a finished transcript to `POST /goal {"text": "..."}`; `POST /stop` is the barge-in.
- The server only queues goals and taps. The main loop stays the single HID writer. The `pi` backend streams
  capture frames at 4 fps; other backends update the screen once per step and after taps (they aren't thread-safe).
- The page is read once at startup, so restart the app after editing it.

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

### Execution architecture experiments (2026-10-17)

`bench.experiments.execution_benchmark` compares isolated screenshot-only browser workflows
with exact post-run DOM oracles. Experimental executors do not change the active
factory or `.env`. The same controller, settling and cancellation path are used.
All API inputs are synthetic screenshots and task text. Direct Live uses the
existing coordinate grid; Jev variants ground semantic targets without it.

```bash
# Prove supported routines finish without any LLM fallback:
.venv/bin/python -m bench.experiments.execution_benchmark --output work/my-no-llm-test --repeats 2 --modes routine-refined --cases two_clicks type_url three_fields double_click hover_menu drag --no-llm

# Compare direct Live, compact Live commands and Jev with narrow Live grounding:
.venv/bin/python -m bench.experiments.execution_benchmark --output work/my-executor-test --repeats 2 --modes live-direct-v2 live-compact jev-live-ground --cases two_clicks type_url three_fields hover_menu drag ambiguous
```

Every study freezes source files and a manifest. Use a fresh output directory for
changed code; `--only-pending-from <study>` runs just missing case/mode/repeat
combinations into a new study after a harness repair. Warm Live connections are
established before the task timer and reported separately; `--cold` includes
connection time. No audio input or playback is used. The URL task is an in-page
local form, not native browser navigation. `calendar` includes six fields, a popup,
and time arithmetic. `overlap-refined` tests an explicit first action while Flash
plans, retaining one HID writer and fresh-screen gates.

Literal routines currently require bounded command syntax. Ordinary paraphrases
may still need Flash. Low-confidence target boxes get one Jev control-type retry;
`jev-live-ground` may then request coordinates from Live only when presence,
uniqueness and stage checks pass. It cannot use Live to bypass ambiguous targets.
Independent result checks and action traces are authoritative for benchmark
success; a model saying done does not establish success. Shared telemetry CSV
retains its existing runtime-status semantics. See HANDOFF.md for results and
known failures before considering any default routing changes.
