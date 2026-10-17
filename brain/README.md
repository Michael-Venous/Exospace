# Brain (Pi 5 runtime)

Screen capture, the Pico HID client, the deterministic controller, the agent loop and the model executors.
It runs on a laptop during development (`--backend browser`) and on the Pi for the demo.

## Running

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[voice,browser]'
cp -n .env.example .env     # then ZAI_API_KEY or GEMINI_API_KEY (either one), TYPESAFE_API_KEY, DEEPGRAM_API_KEY
```

| Backend | Command | Target |
|---|---|---|
| `pi` (default) | `.venv/bin/python -m brain --display 8770` | capture card + Pico (`CAPTURE_DEVICE`, `PICO_SERIAL`) |
| `browser` | `.venv/bin/python -m brain --backend browser --url URL --os linux --headed --display 8770` | an isolated Chromium; models see screenshots only, never the DOM |
| `local` | `.venv/bin/python -m brain --backend local --os macos` | this machine's own desktop (`-e '.[local]'`, PyAutoGUI; not on Wayland; `LOCAL_KEYBOARD` sends keys through a Pico) |
| `qemu` | `.venv/bin/python -m brain --backend qemu` | a QEMU + OVMF UEFI virtual PC for restart and firmware tests (packages `qemu-system-x86`, `ovmf`) |
| `android` | `.venv/bin/python -m brain --backend android` | an emulator or USB-debugging phone over `adb` (screencap in, `input` taps and keys out) |

For the browser backend, run `.venv/bin/python -m playwright install chromium` once, or set
`BROWSER_EXECUTABLE` to an existing Chrome. Add the voice page with
`.venv/bin/python -m brain.voice.layer --agent http://127.0.0.1:8770` ([voice/README.md](voice/README.md)).

Typed goals work too. Ctrl+C during a task cancels and releases input. The next line corrects the
previous goal; `/new GOAL` starts a different one. Ctrl+C at the prompt exits. `--max-steps` defaults to 30.
`--os` or `TARGET_OS` skips OS detection (`TARGET_OS` still detects firmware screens).

On the Pi, `python -m brain.doctor` checks the capture card, Pico, UART and keys; the services and deploy
scripts are in [`scripts/pi/`](../scripts/pi/).

## How a request runs

`EXECUTOR=race` (default, PLAN.md §10; evidence in [bench/arch/README.md](../bench/arch/README.md)):
1. **Triage and plan start together.** One Jev call decides act now / plan / no action, the target box, the
   gesture and whether the request is irreversible. Flash 3.8 plans semantic stages from the same screenshot.
2. **A confident Jev act wins:** the box is zoom-refined and clicked in about a second, and Flash's plan is
   dropped. A remark is remembered with no input ("Noted.").
3. **Otherwise the plan runs.** Each stage has a visible precondition and expected result; Jev grounds every
   pointer target and checks the previous stage on a fresh screenshot. Mismatches replan; real ambiguity asks.
4. **Irreversible requests ask first** ("This can't be undone… Should I go ahead?"). The approval is bound to
   that request and the connected device.
5. **Completion** is one Jev call on goal + executed steps + screen.

**Language model:** z.ai and Gemini are interchangeable. `ZAI_API_KEY`, when set, serves every executor
(GLM-4.6V-FlashX), otherwise Gemini 3.8 Flash. Both have the same settings: model, thinking, a fallback model on
503/429, and the planner model (`.env.example`).

**Quick start:** goals made only of unambiguous commands ("press ctrl c", `type "hi"`), or that start by opening
an app or scrolling ("open calculator, then …"), take their first step with no model call (`agent/quick.py`);
the executor checks the result next step.

Other executors: `EXECUTOR=direct` makes one vision-model call per step (p50 2.5 s on GLM; once a run has sent
input, a `done` claim costs one more call that must see the goal on screen); `jev-flash` runs the planner
alone; `legacy-dual` is the older fast-executor + separate-planner pair.

The loop is the single HID writer. It discards a plan when the screen moved during inference (unless the screen
never settles, like a video), skips the rest of a batch when a click changed the screen, cancels after 3 failed
verifications, and stops a run that repeats the same batch from the same screen. A run survives a restart, replug
or blank screen: it waits up to 180 s for video, re-detects the device and continues on the new screen.

## Restarts and firmware screens (experimental)

The `boot_keys` skill arms setup keys (F2, DELETE, ESC…) that the controller taps every 0.2 s while the target
restarts and for 5 s after its video returns. Taps start only once the Pico sees the USB reset, so a screen that
merely blanks gets none, and a run that ends disarms them. Firmware screens get the `bios` profile: they have no
pointer, so a click presses DOWN/UP until the menu's highlight bar covers the clicked row; right-click, scroll
and swipe are rejected. A `bios` profile is detected again once its screen has changed a lot.

Limits: Mac boot keys must be held, which taps cannot do. PCs with Fast Boot ignore USB keyboards during POST,
and a restart that never resets USB gets no taps. Highlight detection needs a solid bar, so menus that only
recolor text need arrow keys. Measured: 4/21 on a QEMU firmware task.

## Modules

| Module | Responsibility |
|---|---|
| `app.py` | CLI, backends, connection lifecycle, typed goals and corrections |
| `capture.py` | Capture-card frames, no-signal detection, active-rect detection |
| `hid.py` | UART client for the Pico protocol (PLAN.md §5.1); `LocalHID` for the local backend |
| `browser.py`, `qemu.py`, `android.py` | Development targets (capture + input): isolated Chromium, a UEFI virtual PC, a phone over adb |
| `controller.py` | Runs actions (§5.3), coordinate mapping (§5.2), settle detection, typing |
| `monitor.py` | Connection generation; cancels on unplug or lost video (§5.6) |
| `device.py` | OS identification from the first frame, or `--os` / `TARGET_OS`; `bios` re-detection |
| `cancel.py` | The one cancellation token |
| `keys.py`, `imaging.py`, `verification.py` | Key names; crops and JPEG prep; screen-change checks |
| `events.py`, `display/` | Live event stream and the touch display server |
| `arbiter.py` | Commit gate for pipelined voice segments (PLAN.md §11; not wired yet) |
| `telemetry.py`, `errors.py`, `doctor.py` | Benchmark CSV (§5.7); readable errors with a private log; hardware check |
| `agent/race.py`, `agent/triage.py` | Default executor; the triage call, zoom, gesture, irreversible and approval checks |
| `agent/jev_flash.py`, `agent/jev.py` | Flash stages + Jev grounding and completion; Jev HTTP client and gates |
| `agent/executor.py`, `agent/prompts.py`, `agent/model.py` | `direct` executor, its prompts and per-OS hints, Gemini and z.ai clients |
| `agent/quick.py` | Model-free first step for unambiguous commands |
| `agent/orchestrator.py`, `agent/factory.py` | `legacy-dual`; builds the executor named by `EXECUTOR` |
| `agent/loop.py`, `agent/memory.py` | Agent loop; `session.json` memory (§5.5) |
| `skills/` | `open_app` (OS search), `hover`, `wait`, `boot_keys` |
| `voice/` | Voice layer and the older laptop voice demo |

## Touch display

`--display PORT` serves `brain/display/page.html`, fed by live agent events (`brain/events.py`). On the Pi it runs
in Chromium kiosk mode on the Touch Display 2; on the projector it is the demo view.

- **Main view:** live screen with the click marker, history (actions, Flash/Jev reasoning, model time), STOP
  (same cancel path as barge-in), saved-task buttons and a typed-goal box. **Details:** the last Jev scores
  against the `.env` pass marks.
- **Taps:** "Tap = click" clicks that spot through the normal controller. "Tap = this" marks a point for the
  next goal ("delete this").
- **API:** `POST /goal {"text": "..."}`, `POST /tap`, `POST /stop` (barge-in; also drops queued goals),
  `GET /events` (SSE). The server only queues work; the loop stays the single HID writer.
- The `pi` backend streams frames at 4 fps; other backends update once per step and after taps.
  Restart the app after editing the page.

## Logs and tests

- `bench/results.csv`: one row per step (PLAN.md §5.7). `MODEL_LOG` (`bench/model_calls.jsonl`): role, model,
  attempt, elapsed and outcome per model call, never prompts, screenshots or keys.
- Offline tests send no input and make no model calls: `.venv/bin/python -m unittest discover -s tests`.
- Benchmarks and studies with real API calls: [bench/README.md](../bench/README.md).
