# Exospace prototype handoff

## Execution architecture benchmark in progress — 2026-10-17

User requested benchmarking all practical avenues after the architecture reassessment.
Experimental code only; active `.env` and executor factory are unchanged. Added a
literal recipe compiler, stagewise Jev routine executor, persistent Gemini Live
function-call adapter, direct Live executor with hover/drag, compact semantic Live
commands, and explicit-prefix/background-Flash prototype. Shared contracts unchanged.
`brain/execution_benchmark.py` runs isolated screenshot-only browser fixtures with
independent DOM end-state checks; `brain/execution_report.py` aggregates successes,
failures, timing and instruction-compliance adjudication. No microphone/audio,
physical target/HID, native browser address bar, B200, or deployment used.

Initial smoke evidence: `work/execution-smoke-v1`, `work/recipe-diagnostic-v1`,
`work/execution-smoke-v2`. Retained failures exposed vague field targets and a
leftover Live turn-complete event; adapter now waits for the matching request ID.
Main v1 study: `work/execution-comparison-v1`, 7 tasks × 4 modes × 2 repeats.
Frozen original sources are saved under its `frozen_source/` before subsequent
experimental changes. v1 Live-direct lacked explicit hover support and clicked
Tools instead: end-state-only passes must be adjudicated as instruction failures.
Later fixtures explicitly detect that error. Broad semantic Live plans also
sometimes omitted field-focusing clicks; compact commands make those mechanical
steps deterministic. Jev whole-goal checks often rejected correctly completed
screens because earlier actions were no longer visible; stagewise routines retain
verified transitions and require final-effect evidence instead of re-inferring
history from the last image. No probability threshold was lowered.

90 offline tests currently pass. A preliminary authored language-coverage probe
compiled only 5/12 phrasings (`work/execution-language-coverage.json`); this narrow
grammar is not evidence for 90% real-world coverage. v1 is complete; follow-up study `work/execution-comparison-v2` is running.
Remaining queued studies: `execution-no-llm-v1`, `execution-heldout-v1`,
`execution-overlap-v1`, `execution-cold-v1`, `execution-grounding-v1`, and
`execution-calendar-v1` under work/. The last study reuses the six-field calendar
with popup and time arithmetic. Grounding recovery uses Live only for missing
coordinates after Jev confirms presence, uniqueness and stage readiness; ambiguous
targets cannot use it. A bounded Jev control-type refinement is tested first.
All test versions freeze source and manifests. Report failures and bounded
recommendations after these finish. Do not switch defaults on these exploratory results.


## Execution architecture reassessment — 2026-10-17

User asked to pause and reassess the reliance on ~5-second LLM calls for ordinary
multi-step commands. This turn is a design review: no runtime, model configuration,
threshold, or shared §5 contract changes. Existing model comparisons remain evidence,
but optimizing model routing is no longer the next priority.

Code inspection confirms the architectural bottleneck: `direct_command` only
accepts a single pointer command (including double-click). Unsupported requests
first make an empty-proposal Jev assessment and then unconditionally call Flash.
The controller already supports typing, hotkeys, double-click, hover and drag;
the missing layer is a bounded way to compose them from explicit user instructions.
In `work/sequence-comparison-v2/type_url-flash-1/model_calls.jsonl`, Flash consumed
4.060 s and individual Jev calls 0.151–0.191 s. This was a synthetic local URL form,
not native Chrome navigation. Removing that planner call has not yet been tested.

Recommended next architecture: Jev selects and verifies small executable recipes;
Flash supplies missing reasoning/content or recovery plans. Action count alone
must not cause escalation. Candidate recipes: explicit click sequences, literal
field fills, browser URL navigation, open-app, double-click, hover then submenu
click, and named-source/named-destination drag. Recipes bind arguments to exact
user-provided spans (quoted strings, URLs, explicit targets). For broader language,
bounded span candidates can be offered to Jev with an abstain option; do not assume
it generates arbitrary strings or free-form plans. Reject unsupported grammar,
ambiguous argument binding, implicit reasoning, and conflicting instructions.
Do not split `and` inside literal text or silently omit clauses. Existing HID
ASCII folding means exact non-ASCII typing is not yet supported and must not be
claimed as verified literal entry.

Each recipe is a small state machine, not a blind macro. On a fresh frame, Jev
checks prior outcome and selects/grounds the next allowed action. Dependent menu
items are grounded only after the menu opens. Batch checks and candidates that
share the same frame, but never treat independently answered questions as a
sequential plan. A field click plus exact typing can be one bounded stage; check
the visible value, not an invisible caret. Preserve original-goal completion,
single HID writer, execution acknowledgement, stale-frame checks, cancellation,
and existing evidence thresholds. Platform-specific shortcuts belong in adapters;
screen-based target descriptions and recipe semantics remain portable. Neither
the 90% coverage estimate nor production reliability has been established.

For longer tasks, explore overlap only after the direct recipe path is validated:
execute an explicit, independently justified prefix while Flash plans the remainder.
Example: open Chrome for an explicit request to open Chrome and then compare items.
Do not let Jev and Flash separately write HID. Track goal/device/session and action
versions internally; supply completed actions and current observations to the
planner. Accept only still-applicable remainder steps, reground on the current
frame, discard obsolete proposals, and do not repeat completed actions. Stop at
the first choice requiring missing reasoning. This improves time to first useful
action; it does not guarantee a shorter full task or justify speculative actions.

Next implementation/evaluation order: (1) bounded literal recipe compiler and Jev
selection, (2) withheld paraphrases, literal-text edge cases, ambiguous/conditional
requests, negative cases, and true end-state oracles, (3) execution tests with Flash
disabled to prove zero-call coverage, (4) measure first useful input and full-task
p50/p95, fallback rate, wrong actions and false completion, (5) background planning
with cancellation, screen-change, duplicate-action and stale-result tests. Compare
the existing two-click/typing/field fixtures, then native browser/device paths when
available. Do not infer a new latency result by subtracting Flash time from an old run.

Primary reference rechecked: https://docs.typesafe.ai/primitives and
https://docs.typesafe.ai/primitives/choice . These document fixed-choice judgments,
independent batched questions and composition in application code. Recipe control
and argument binding above are our proposed architecture, not demonstrated vendor
capabilities or implemented runtime behavior.

## Current routing and Live evaluation — 2026-10-17

**Decision:** keep **Jev 1.14 + Gemini 3.8 Flash** as the active executor. No
automatic Lite/Flash/Argon model router is enabled; `.env` was not changed. User
prioritizes reliability over a small speed gain, permits Lite only for extremely
basic literal sequences, and wants it removed from automatic routing if Jev
admits complex tasks. No complex false admission was observed in this pilot,
but Lite also produced a false completion on a basic task. Keep it experimental.
B200 remains deferred. Latest model/verification changes are local; no deployment
or push was performed by this agent. Baseline commit `f45eeb2` appeared during
this session; preserve it and other contributors' work.

### Measured model/routing comparison

`brain/routing_benchmark.py` freezes a hashed 36-case manifest, with different
calibration and held-out task families. Calibration used one call/model/case;
held-out used two repeats. Models saw screenshots and requests only. Exact
oracles were independently checked with dynamic programming, Decimal, and datetime.
Configs: 3.5 Flash-Lite/low, 3.8 Flash/low, 4 Argon/medium. All endpoints worked.

- Jev: 32/32 structural labels correct on first pass, repeated choices stable;
  median request ~0.16 s. It falsely marked four clear menu tasks ambiguous.
  This is a synthetic pilot, not broad reliability or calibrated correctness.
- Calibration: Lite 10/18; Flash and Argon 18/18 each.
- Held-out: Lite **20/36**, Flash **36/36**, Argon **36/36**; these are 18 distinct
  tasks repeated twice. Mean full response time: **1.46 / 7.11 / 12.35 s**.
- Frozen candidate: complexity score <1.5 → Lite; <2 → Flash; otherwise Argon;
  ambiguity/API failure → Flash. This improved calibration mean to 5.45 s, but
  held-out mean was **7.94 s vs Flash 7.11 s**, about 12% slower, same accuracy.
  Task-cluster bootstrap 95% interval for router-minus-Flash latency is
  [-0.54, 2.51] s: no established latency benefit. Do not enable those thresholds.
- Challenge set: 22 further tasks, two repeats. The original frozen Lite boundary
  admitted 12/12 literal trials and **0/32 nonliteral/ambiguous trials**. A stricter
  confidence overlay also admitted zero unsafe tasks but rejected all literal
  tasks, so it offered no useful coverage. Confidence is not a correctness guarantee.

Evidence: `work/routing-benchmark-v1/`, `work/lite-gate-v1/`. The gate challenge
uses one request phrasing with the actual requirements on screen, including hidden
arithmetic, translation, summary, constraints, missing preferences and quoted text.
No confidence/localization threshold or shared §5 contract was relaxed.

### Full short workflows and fixes

`brain/sequence_benchmark.py` runs real Jev + Gemini plans through screenshot-only
browser input. Post-run DOM checks are independent. The URL fixture is a local
form; it is NOT Chrome's native address bar or the actual YouTube site.

- Initial baseline (`work/sequence-comparison-v1/`): Lite 4/6, Flash 6/6.
- Diagnostic run proved invisible focus checks were causing unnecessary replans.
- After the plan-contract correction (`work/sequence-comparison-v2/`, 3 runs/case):
  Lite 8/9, Flash 9/9. Median end-to-end times (including failed runs):
  two clicks **6.72 / 9.58 s** (Lite 2/3, Flash 3/3); URL typing **4.93 / 6.42 s**
  (both 3/3); three fields **6.54 / 14.68 s** (both 3/3).
- One Lite failure falsely declared done after opening the catalog but BEFORE
  clicking its first result. Jev accepted Lite's weaker completion wording (0.96).
  This is more serious than a timeout and remains preserved in the reports.

Corrections applied in `brain/agent/jev_flash.py`: batch field click + literal
input; forbid keyboard focus/caret as visual conditions; permit empty semantic
plans only for independent completion verification; and judge completion against
**the original user goal plus the model's proposed evidence**, so a weaker plan
cannot silently redefine success. Presence/uniqueness/coordinate gates unchanged.

Latest validation:
- **59 offline tests pass**, compileall and git diff --check pass.
- Exact false-positive screenshot replay: false done prevented **3/3**; new Jev
  completion scores 0–0.01 (`work/goal-verification-replay/`). Fixed weak completion
  proposal was replayed, not a real Gemini generation; no input was executed.
- Following the goal-verification fix: two clicks **2/2 per model**, Lite median
  6.04 s, Flash 9.11 s (`work/goal-verification-regression/`).
- Ambiguous Continue: asked the user, no click, **3.09 s**.
- Calendar with popup, six exact saved fields and computed end time: **passed in
  31.37 s**, 5 Flash calls / 14 Jev calls. See `work/goal-verification-calendar/`.
- These small regression checks do not establish production reliability.

### Gemini Live feasibility

User asked whether Live supports images and whether its protocol is practical.
Official docs: https://ai.google.dev/gemini-api/docs/models/gemini-3.8-live and
https://ai.google.dev/gemini-api/docs/live-api/capabilities . It accepts images,
text/audio/video and supports function calls; strict structured outputs are not
supported. Response modality is audio; text is available via transcription.

`brain/live_probe.py` now tests an inert screenshot-reading function, without a
microphone, audio playback or HID. Initial separate video-frame + immediate text
turn approach failed 3/3 (`work/live-probe-v1/`); do not attribute that alone to
model vision capability. Sending image + text in ONE client-content turn passed
**3/3** (`work/live-probe-v2/`), with correct function calls **0.51–0.73 s** after
sending; fresh connection setup added **0.38–0.55 s**. This establishes small API
feasibility, not voice quality, warm-session performance or multi-step planning.
No Live runtime adapter is enabled. Potential next step: evaluate Live as voice,
acknowledgement and interruption interface that dispatches validated high-level
tasks to the existing executor. Preserve single-writer HID, cancellation, freshness
checks, session resumption and deduplication before any real integration.

User-facing combined evidence is exported under
`/home/mike/Documents/Codex/2026-10-17/con/outputs/`. All benchmark requests were
synthetic; secrets remain only in the ignored `.env`. Hardware/voice/Pi deployment
and cross-device limitations from the earlier checkpoint still apply.

## Earlier prototype verification — 2026-10-17 (superseded by study above)

User selected **Jev 1.14 + Gemini 3.8 Flash**, deferred B200, and requested local
end-to-end testing while mic, speaker and adapter are unavailable. Both keys are
in the ignored owner-only `.env`; never print or copy them into reports. The first
Gemini key exhausted its 20-request free quota; user replaced it and subsequent
live calls worked. At this earlier checkpoint no Argon calls had been made. The user was considering optional
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

## Claude review (2026-10-17): open findings, highest priority first
All 77 tests pass (`.venv/bin/python -m unittest discover -s tests`). No secrets are tracked, and `.env` is ignored.

1. **The Pico firmware doesn't exist yet.** `firmware/pico/` contains only a README. The whole hardware path (`--backend pi`) is blocked on it. The spec is in the earlier "To do next" item 5.
2. ✅ **Fixed.** **Animated screens can stall the agent** (`brain/agent/loop.py:83`). If the screen changes while the model is thinking, the actions are discarded and the step restarts. On a screen that never stops changing (a playing video, a spinner, an animated ad), every step is discarded until `max_steps` runs out, at roughly 2.5 s of settle timeout plus a model call each time. Fix: after 1–2 discards in a row, act anyway, or compare only the region around the target.
3. ✅ **Fixed.** **A typed goal can be silently dropped** (`brain/app.py:88-94`). If the target disconnects during `detect_profile`, the code calls `continue`, which discards the goal and goes back to the prompt. Fix: keep the goal and retry once the target reconnects.
4. ✅ **Fixed.** **Errors are unreadable when debugging.** `loop.py:154` and `app.py:109` replace every exception with a generic message, and no traceback is logged anywhere. Hardware errors (`HIDError`, a capture card that won't open, a bad serial port) contain no secrets but are hidden too. Fix: print `str(exc)` for local errors (HID, capture, serial, `OSError`), and write tracebacks to a log file under `work/`.
5. **Experiment code is mixed into the runtime package.** `brain/agent/*_experiment.py` and `brain/*_benchmark.py` sit in the shipped package. CLAUDE.md says code quality is judged and to avoid dead code. Move experiments to `bench/` or delete them before submission, so `brain/` holds only the runtime.
6. **`jev_flash.py` style doesn't match the rest of the codebase.** It uses dense one-liners, no spaces after commas, missing type hints, and imports inside functions. The AI-track judges read the code, so run `ruff format` and add types.
7. **Most work is uncommitted.** The only commit is `f45eeb2`; many modified and untracked files exist only on this laptop. Commit to a branch (e.g. `agent/prototype`) so work isn't lost and teammates can pull it.
8. **Low priority:** `jev_flash.py:249` checks whether the fallback screenshot changed by comparing JPEG bytes for exact equality. Real capture-card frames are never byte-identical, so the check always reports "changed." That currently only triggers after a discarded step, where replanning is correct anyway, but compare thumbnails if it's ever used elsewhere.
