# Exospace prototype handoff

## Overnight branch state — 2026-10-17 (`overnight-ultracode`)

Built unattended on `overnight-ultracode`; `main` is untouched. All 101 unit tests pass on
the laptop and on the Pi at `3d7f138` (`.venv/bin/python -m unittest discover -s tests`).
Where this section and the one below differ, this one holds for this branch.

**Defaults and latency.** With `ZAI_API_KEY` in `.env`, `EXECUTOR=direct` runs
`glm-4.6v-flashx`: one call per step returns the actions and their coordinates. Measured
over 1,299 logged calls: p50 2.5 s, p90 4.3 s, p99 14 s per call. Each call has a 12 s
timeout (`MODEL_TIMEOUT_S`) and up to three tries, so a step whose calls all hang waits
about 37 s before it fails. Some goals start with no model call at all
(`brain/agent/quick.py`): plain commands such as "press ctrl c" or `type "hi"`, and goals
that start "open calculator, then ..." or "scroll down and ...". Once a run has sent
input, a `done` claim costs one extra call (about 1.7 s) that must see the goal on screen.

**Any OS.** The device profile (§5.4) names the OS as windows, macos, linux, chromeos,
android, ios or bios, and each OS gets a short hint in `OS_HINTS` (`brain/agent/prompts.py`).
- Phones: `--backend android` drives a phone over adb with touch input.
- Firmware: `--backend qemu` runs OVMF on a scratch copy of its variable store.
  - Clicks become arrow-key presses.
  - `boot_keys` taps the setup keys through a restart, but only once the Pico reports the
    USB reset. A run that ends disarms them.
  - A run waits up to 180 s for the target to come back after a restart, then carries on
    with the same goal (§5.6).
- A `bios` profile is classified again once its screen has changed a lot since it was last
  classified, mid-run or between goals (a boot logo can look like firmware).

**End-to-end results.** These used the live model, and each oracle checked the result
outside the agent. A run counts only when its oracle could tell (one alarm run started
with the alarm already set, so it is left out).

| Goal | Target | Result |
|---|---|---|
| Open a text editor, type a line, save `~/hello.txt` | nested XFCE desktop | PASS, 36.5 s (the 3 runs before the fixes failed) |
| Work out 1234*5678 in the calculator | nested XFCE desktop | PASS, 24.2 s, with the answer stated |
| Turn on dark theme | Android emulator | 2 of 3 passed (46.2 s, 50.5 s); the failure came before the Android hint |
| Set an alarm for 6:45 AM | Android emulator | 1 of 3 passed (59.1 s). The Android hint now names the Clock app for an alarm, so this is no longer a blind test; its one run since failed at the clock-face time picker |
| Add contact Ada Lovelace, 555 123 4567 | Android emulator | 0 of 9 passed. One run first passed a loose check with the name "Ada Lovelaceyy555 123"; the exact check now fails it. The last run typed the number into the name field |
| Restart into firmware setup and set the boot time-out to 7 s | QEMU/OVMF | 4 of 21 passed |

**Limits and risks**
- **Contract notes need team sign-off.** PLAN.md now has notes on:
  - §5.3: the `wait` and `boot_keys` skills
  - §5.4: the new OS names and bios re-detection
  - §5.6: resuming after a restart
  - §3: firmware is now in scope
  - §13: `direct` is the default executor with z.ai
- **Untested targets.** iOS and ChromeOS have not been tried at all. Android has only run on
  the emulator, where the on-screen keyboard covers form fields; a real phone typing
  through the Pico keyboard is untested. The OS hints were tuned on Linux and Android only,
  and the one-batch save hint covers Linux only.
- **Firmware is unreliable.** The model often claims `done` before saving with F10 then Y.
  A click on a list that scrolls under its highlight bar is rejected, so the model must
  use the arrow keys there.
- **Thin re-detection margins.** The threshold (40) is close to the smallest
  firmware-to-desktop change measured (44.1, OVMF to Linux). One firmware-to-phone pair
  changed by only 1.9, so a phone misread as firmware can stay `bios`. Consecutive firmware
  screens changed by up to 114, so some firmware steps pay for an extra classification call.
- **Restarts the Pico cannot see.** PCs with Fast Boot ignore USB keyboards during POST,
  and a restart that never resets USB gets no key taps.
- **Batch guard.** When a tap changes the screen mid-batch (for example by opening the
  keyboard), the rest of the batch is skipped and planned again, which costs a step.
- **Quick-path misses.** The quick path decides with a word list, so "open a text editor"
  still goes to the model.
- **Model quirks.** The model sometimes leaves out a coordinate, which wastes a step, and
  has once given an empty `done` message. One call hung for the full 12 s before its retry.
- **Self-reported success.** The success column in `bench/results.csv` is the agent's own
  claim. Only the oracles above are independent checks.

## Current verified state — 2026-10-17 (supersedes historical snapshot below)

User selected **Jev 1.14 + Gemini 3.8 Flash**, deferred B200, and requested local
end-to-end testing while mic, speaker and adapter are unavailable. Both keys are
in the ignored owner-only `.env`; never print or copy them into reports. The first
Gemini key exhausted its 20-request free quota; user replaced it and subsequent
live calls worked. No Argon calls were made. User is now considering optional
`ask_argon` for difficult reasoning; this is a recommendation under discussion,
NOT enabled or implemented. Current routing choices are `execute` and `ask_flash`.

Implemented locally:
- Jev batches explicit routing Choice, target presence/uniqueness/clarity, Box
  grounding, previous-stage verification and next-stage readiness.
- Explicit single pointer commands can execute without Flash. Otherwise Flash
  produces cached semantic stages. Actual controller acknowledgement advances the
  plan; stale/cancelled work cannot advance it. Goal/device changes invalidate it.
- Uncertainty triggers bounded Flash replanning or user clarification. Choice and
  Box confidence have different semantics and are not given a shared cutoff.
- After failed Jev localization, Flash may ground ONE visible pointer action.
  Such fallback coordinates are discarded if the screenshot changes before
  execution. Future actions require another observation. No invented user intent.
- Hover uses existing skill + UART move; drag maps to existing swipe. Hover-menu
  recovery retains interaction context so Flash can move outside a menu to reveal
  a covered result. No shared action/UART/memory/benchmark schema changed.
- Added isolated Playwright browser capture/input backend and live fixture suites.
  Models get screenshots only, not DOM. DOM is used solely as a post-run oracle.
- Fixed small field changes being lost by whole-screen thumbnail averaging, and
  avoided double-counting unchanged pixels while semantic verification is pending.

Validation:
- **47 offline tests pass locally**, compileall, pip check and git diff --check pass.
- Final calendar suite **3/3 passed**, checking all six stored fields exactly:
  31.166 s with popup; 27.535 s without popup; 27.677 s with popup + moved layout.
  Each used 3–4 Flash calls over 10–11 turns. Jev medians ~0.20–0.22 s/request.
- Direct click: 1.317 s, 1 Jev call, 0 Flash calls, outcome confirmed.
- Three-toggle sequence: 24.766 s, 6 Flash / 10 Jev calls, outcome confirmed.
  False abstentions caused excess replanning; not yet an efficient showcase.
- Ambiguous Continue buttons: 4.273 s, 1 Jev / 1 Flash call, asked which workspace;
  neither button clicked.
- Hover → submenu click → reveal covered confirmation: 29.465 s, 8 Flash / 10 Jev
  calls, passed after targeted visual-fallback recovery. Earlier hover runs failed
  verification even when the click succeeded; preserve those failures in reports.
- Live negative checks: present target accepted; absent and duplicate targets
  abstained. These few examples do not calibrate confidence or establish reliability.

Evidence (all synthetic/private development artifacts in ignored work/):
`work/calendar-final/`, `work/orchestration-final/`,
`work/hover-visual-fallback/`, `work/jev-negative-checks/`.
Earlier failed iterations remain in work/. User-facing consolidated JSON and final
screenshots are in `/home/mike/Documents/Codex/2026-10-17/con/outputs/`.
No broad production success-rate or speed comparison against KaiVM is established.

Run locally:
```sh
BROWSER_EXECUTABLE=/opt/google/chrome/chrome .venv/bin/python -m brain.local_scenarios --case all
BROWSER_EXECUTABLE=/opt/google/chrome/chrome .venv/bin/python -m brain.local_benchmark --runs 3
```
Playwright Chromium is also installed, so the executable override is optional.
`python -m brain --backend browser --url URL --os linux --headed` provides typed
interactive goals in an isolated browser profile. Desktop-wide Wayland input,
physical HDMI/Pico, phone adaptation, audio and automatic cross-device resume
remain unverified/missing. Latest changes are local only; Pi has the earlier
baseline. All project work remains uncommitted; no push/tag/service deployment.

Next priorities: calibrate Jev routing/presence gates on held-out tasks to reduce
false abstentions, benchmark cached plans and one-step visual fallback on more
apps, and evaluate Argon only on tasks Flash demonstrably struggles with. Keep
B200 deferred. Hardware/firmware owner coordination and earlier contract concerns
remain applicable when physical integration resumes.

## Historical snapshot (superseded where the current state above differs)

Last verified: **2026-10-17 America/New_York**. Read PLAN.md and CLAUDE.md first.
The prototype files were already untracked when this continuation began. All
work remains uncommitted; no tag or remote push was made.

## Goal and current decision

The user wants immediate response, platform-independent operation and intelligent
long-horizon work. They requested Gemini 4 Argon, then asked about Jev (TypeSafe's
decision model, not JEPA) plus an accessibility translation layer. They report
access to a B200 at **UF HiPerGator**; a friend is setting up API access. There is
no endpoint available to this session yet.

**Implemented baseline:** Gemini 3.5 Flash-Lite at low thinking for fast responses
and visual execution; Gemini 4 Argon at medium thinking for planning/recovery.
FAST_MODEL and PLANNER_MODEL are configurable. Compare 3.8 Flash for harder visual
execution before choosing a winner. No live latency or task-success claim yet.

**Recommended experiment, not implemented:** screenshot parsing → normalized
candidate controls → Jev-style selection → existing HID controller. Argon supplies
plans, generated content and recovery. Native accessibility can populate the same
control representation, but needs target-side software/permissions and must remain
optional for the no-install demo. Use pretrained B200 inference first; train only
for measured failures with held-out evaluation. Research and primary links are in
brain/README.md. Jev image/Box support is experimental; Box returns a location even
when the object is absent. AnyJev is an open decision-model candidate, not TypeSafe
Jev weights, and its vision backend is on its roadmap.

## Completed

- `brain/skills/__init__.py`: OS app-search shortcuts, validation, cancellation.
- `brain/device.py`: first usable frame → active rect → OS classification or override.
- `brain/agent/model.py`: shared Gemini transport, explicit request timeouts, bounded
  transient/malformed-response retries, separate private per-model timing log.
- `brain/agent/executor.py`: fast execution + delegation metadata; rejects malformed
  batches without executing dependent actions. Public HID/action contracts unchanged.
- `brain/agent/orchestrator.py`: fast model responds before delegating; Argon returns
  guidance and subtasks, never HID commands. Fresh capture after planning; plan refresh
  after eight steps, recovery, task/device changes; repeated delegation bounded.
- `brain/agent/loop.py`: one HID writer, settle checks, memory/transcript, telemetry,
  cancellation and release, recovery after two failed verifications, cancel after
  three, step limit. Drops actions if the screen changes during inference. A `done`
  returned after input in the same batch is deferred until a fresh-screen verification.
- `brain/app.py`, `brain/__main__.py`: typed-goal REPL, local/Pi backends, OS override,
  Ctrl+C interruption, correction continuation, `/new GOAL`, target re-detection.
- `brain/probe.py`: opt-in synthetic-image model-access/complete-response latency
  probe. No desktop capture or HID. No live probe has been run.
- Fixed `CancelToken.run` swallowing worker TimeoutError indefinitely. Queued
  abandoned calls are canceled; already-running requests may finish remotely.
- Memory creates parent directories and preserves subtasks/done fields.
- Monitor can close cleanly and requires a real initial video signal.
- Updated config example, ignore rules, runtime documentation and PLAN §13.

## Verified software environment

Local source: `/home/mike/Documents/code/Exospace`.
Local `.venv`: Python 3.14, google-genai 2.31.0, OpenCV 5.0.0.93, NumPy 2.5.4.

Pi: `ssh mike@100.76.219.77`, observed host **nicopi**, Python 3.11.2.
Deployed copy: `/home/mike/Exospace`; venv `/home/mike/Exospace/.venv`.
Installed editable package with google-genai 2.31.0, OpenCV 5.0.0.93, NumPy 2.4.6.
No service installed, boot files changed, hardware input sent, or GPU jobs started.

**Verified on both machines:** all 21 offline tests pass with
`.venv/bin/python -m unittest discover -s tests -v`.
Tests cover platform shortcuts, malformed batches, direct/delegated execution,
planner cancellation, stale-screen rejection, fresh completion verification,
recovery/cancel limits, plan reset, model transport configuration, bounded JSON
retry, timeout propagation and privacy of timing logs. CLI help works. Compile
checks passed; local `pip check` and `git diff --check` passed.
The real SDK client constructors were also checked on Pi with a dummy key and
no request. Tests use synthetic video/recording HID/mocked model calls, not hardware.

## Blockers and next actions

1. **Credentials:** no GEMINI_API_KEY in the local environment or project .env,
   and none in the inspected Pi login environment. Asked user for the path to an
   existing key file, not the secret itself; no response yet. Configure privately,
   then run `.venv/bin/python -m brain.probe --runs 3` on Pi. Compare with
   `FAST_MODEL=gemini-3.8-flash` and representative GUI tasks; synthetic probes
   measure request latency only.
2. **Hardware:** no capture card/Pico observed. Pi `/dev/video19`–`35` names are
   rpi-hevc-dec/pispbe internal devices. `/dev/ttyAMA10` and `/dev/serial0` exist;
   `/dev/ttyAMA0` does not. Do not use debug UART as header UART. Previous handoff
   says enable `dtparam=uart0=on`, disable serial console, reboot; not done here.
3. **Firmware:** still missing. Previous design proposes CircuitPython keyboard,
   absolute mouse and touchscreen, UART on Pico GP0/GP1 at 115200, GP22 dev mode.
   Keep wiring/firmware work coordinated with hardware owner.
4. **Contract review remains:** previous untracked code assumes ping `mounted`,
   positive-down scroll, remember/done-summary actions, Linux/unknown OS values,
   and a status field in memory. Existing monitor blank grace is 1 s while PLAN
   says 300 ms; loop refuses blank input immediately. Prior firmware proposal
   mentions multitouch while PLAN says single-touch. These differences predate
   the continuation; shared-contract updates still require human/team agreement.
5. **B200 deferred by user (2026-10-17):** skip GPU serving, training and endpoint
   setup for now. It is not a blocker. Use hosted model APIs for this prototype.
6. **Not yet implemented:** voice, automatic task resume on reconnect, accessibility
   adapters, Jev integration, B200 executor, real-task latency benchmarks, firmware.
   The current REPL re-detects a target before the next request/correction.

## Running once credentials and devices are ready

```sh
cd ~/Exospace
.venv/bin/python -m brain --backend pi
```

Local development needs `pip install -e '.[local]'`, desktop permissions and a
supported input environment. PyAutoGUI is not a general Wayland input solution.
`MODEL_LOG` records per-role/model timings separately from the shared CSV. No
prompts, screenshots or keys are stored in that timing log; session.json contains
conversation/facts and is ignored by Git. Keep all private runtime files out of
commits and deployment archives.

## Focused Jev vision research — 2026-10-17

Current scope update: the user explicitly deferred B200 work. Recommended next
comparison is direct Jev + smart planner against the existing fast-LLM + planner
baseline. Jev is not integrated yet; retain the baseline until real measurements
justify replacement. A third model is optional, not a required serial stage.

User asked for clearer image-capability evidence. Revised recommendation: test
**direct screenshot → Jev 1.14 → HID**, with Argon supplying narrow objectives and
text, before assuming a parser/accessibility layer is necessary. Box accepts a
natural-language target description; no candidate boxes are required. Batch
presence, uniqueness, location and next-state checks against one screenshot.
Questions are independent, so a Box cannot consume another question's newly
selected target within that same request: provide a known target, speculate over
known targets, or use a second request.

Native support is confirmed in official docs: images on 1.13/1.14, Box on 1.14;
pin `jev-1.14.0` and use the image opt-in header. Current limits are PNG/JPEG,
280–1568 pixels on each side; images outside limits are rejected, not auto-resized.
Use multipart and one shared image in state; map crop/padding coordinates back
before HID. Preview remains explicitly experimental. Box has no absent-target
sentinel. No live API requests or benchmark runs were made in this research.

New evidence: Benchmark Heaven's ImageJevBench Box preview (14–15 Oct) measures
300 public images including 50 synthetic UI examples. Jev 1.14: median request
latency 0.33 s, mean IoU 0.628 overall, 70% IoU >= .5, UI mean IoU .578; reported
cost $0.057/1,000 boxes. These are independent author measurements, not our Pi
results, and neither IoU nor their Capability score is click/task success. UI-only
click-center accuracy and Pi p95 latency remain unknown. The method file retains
older 256-pixel/600-sealed statements; current official API docs say 280 pixels,
and the current board says 546 sealed after quality control. Use current API
limits; do not merge those versions silently.

Useful B200 comparison: SeanLiu/Jev-Vision is a separate open Qwen3-VL-8B adapter
and decision-head model (not TypeSafe Jev). Author reports 150–190 ms screenshot
inference on H100 batch 1; expects marked candidates, not arbitrary Box outputs.
Model card reports weaker desktop grounding, overconfidence on long text, and no
mobile/logged-in/form-heavy-flow evaluation. Treat as a candidate, not a proven
cross-device substitute.

Next experiment: identical desktop + phone screenshots for Jev, Gemini baseline,
and later B200 candidate; include absent targets, duplicate labels, small icons,
modals and rotations. Measure click-center hits, false clicks on absent targets,
abstention coverage, p50/p95 complete Pi request latency, and full task success.
Keep Argon fallback; no speculative typing/clicking while it replans.

Sources checked:
- https://docs.typesafe.ai/images
- https://docs.typesafe.ai/primitives/box
- https://docs.typesafe.ai/models
- https://docs.typesafe.ai/introduction
- https://benchmarkheaven.com/image-jev-bench#box-track
- https://github.com/fstandhartinger/jevbench/blob/main/datasets/imagejev-box-v0.1/METHOD.md
- https://huggingface.co/SeanLiu/Jev-Vision
