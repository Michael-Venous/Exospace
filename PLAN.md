# Exospace — Team Plan (shared context for all agents)

> Read this fully before writing code. If your task conflicts with this file, stop and ask your human.
> Contracts in §5 are shared across teammates: **do not change them without telling the team.**

## 1. What we're building
A portable box that plugs into a computer or phone over one USB-C cable, **sees the screen** (HDMI capture) and **controls it** (pretends to be a USB keyboard/mouse/touchscreen). The user presses a button and speaks a goal; the box completes it. **Nothing is installed on the target.** It keeps memory, so you can unplug, move to another device, and continue the same task.

**Pitch:** "Plug it into any computer or phone, say what you want, it does it — and remembers you across devices."

**Event:** SwampHacks XII, hacking Fri Oct 16 8:30 PM → ~36 h. Team of 4. Tracks: Hardware (primary), AI, Overall, MLH Gemini.

## 2. How we're different (keep every decision serving these)
1. **Speed** — measured per step and per task, shown live with an on-screen timer. Prior art (KaiVM) works but is slow.
2. **Cross-device continuity** — hot-swap computer A → computer B → Samsung phone with automatic adaptation and retained context.
3. **Voice you can interrupt** — barge in, correct, continue.
4. **Recovery** — moved window / unexpected dialog doesn't derail it.

Positioning vs accessibility devices: we *complement* head mice/switches (they replace the mouse; we replace the steps). Say "designed for limited hand mobility," never "validated."

## 3. Non-goals (do NOT build)
BIOS repair (changing settings in the firmware setup screen is in scope, by user request 2026-10-17), scheduling, a large dashboard, fine-tuning (unless the benchmark shows one specific weakness), fully-offline claims. These overlap with prior art or eat time.

## 4. Architecture
```
            ┌──────────── Pi 5 ("brain") — own power, never reboots ────────────┐
 user ─────►│ button (GPIO) + USB mic + speaker   memory (session.json)         │◄──► B200 server
 target ───►│ capture card (/dev/video*)          controller + agent loop       │     (models, HTTP)
 screen     └───────────────┬───────────────────────────────────────────────────┘
                            │ UART JSON lines (§5.1)
                            ▼
                 Pico ("hands") ── USB via hub ──► target sees keyboard + abs mouse + touchscreen

 target ◄═ one USB-C ═► USB-C hub ─HDMI─► capture card ─USB─► Pi 5
                                └─USB-A─► Pico
```

**Layers (top → bottom):**
- **Planner** (frontier model, Gemini first; Claude as alternate): goal → subtasks; called on new request, device change, or repeated failure. Owns context.
- **Executor**: screenshot + subtask → next action. Baseline = same frontier model. Candidate = UI-TARS-1.5-7B on B200 (vLLM, OpenAI-compatible API). Adopt only if benchmark says faster at equal success.
- **Controller** (deterministic Python, no AI): runs actions, per-OS shortcut skills, typing, settle detection, cancellation, coordinate mapping.
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
| `scroll` | `dy` (±) | wheel |
| `tap` | `x,y` (0–32767) | touchscreen (Android) |
| `swipe` | `x1,y1,x2,y2,ms` | touchscreen |
| `release_all` | — | **must** release every key/button/touch |

- Pico → Pi unsolicited events: `{"ev":"mounted"}`, `{"ev":"unmounted"}` (USB host attach/detach). On `unmounted`, Pico auto-runs `release_all`.
- HID descriptors: boot-protocol keyboard + absolute mouse (0–32767) + single-touch digitizer. Firmware: CircuitPython (`boot.py` custom descriptors) unless it blocks us.

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
{"type":"skill","name":"wait"}                 // until the screen changes, at most 10 s
{"type":"skill","name":"boot_keys","args":{"keys":["F2"]}}  // tap firmware keys through the next restart
{"type":"scroll","dy":-3}
{"type":"done"} | {"type":"ask_user","question":"..."} | {"type":"fail","reason":"..."}
```
*(2026-10-17: `wait` and `boot_keys` added by user request for restarts and firmware setup. On a
`bios` profile a click or double_click moves the menu highlight with arrow keys (double_click
then presses ENTER); right_click, scroll and swipe are rejected.)*
*(2026-10-17, approved: `wait` added for slow loads and installs; it sends no input, so a still screen after it
is not a failed verification. Offered to the direct/legacy executors; race gets it once measured end to end.)*

### 5.4 Device profile (set on every connect)
```json
{"os":"windows|macos|linux|chromeos|android|ios|bios","rect":[x0,y0,x1,y1],"input":"mouse|touch","connected_at":ts}
```
OS is detected **from the first frames** (USB can't reliably tell us). `bios` means a firmware
setup or boot screen. Every executor step gets the OS with its keyboard conventions.
*(2026-10-17: linux/chromeos/ios/bios added by user request to run on any OS and the boot screen.
A restart's boot logo can read as `bios`, so a `bios` profile is re-detected once its screen
has changed a lot since it was classified, mid-run or between goals; `connected_at` stays.)*

### 5.5 Memory (`session.json`, persists across swaps)
```json
{"transcript":[...], "facts":{"event":{"title":"","date":"","time":"","location":""}},
 "task":{"goal":"","subtasks":[],"done":[]}, "prefs":{}}
```
Planner gets memory + device profile on every call.

### 5.6 Cancellation — ONE path for everything
Triggers: Pico `unmounted` · capture no-signal/black > 300 ms · button press during execution (barge-in) · 3 failed verifications.
Effect: Pico `release_all` → drop action queue → (if disconnect) wait for new device → build profile → planner resumes from memory.
*(2026-10-17: resume implemented. After a disconnect or lost video the run waits up to 180 s for
live video, tapping armed `boot_keys` once the Pico has unmounted (the restart), re-detects the
profile if the connection changed and continues the same goal on the new screen.)*
*(2026-10-17, user decision: with `EXECUTOR=jev` a run not done within 20 s (raised from 5 s) is also cancelled
this way (`DEADLINE_S`, brain/agent/jev_only.py): a request Jev alone can't finish that fast
isn't reliable enough. A deadline that passes while waiting for a reconnect still ends the run.)*

### 5.7 Benchmark log (`bench/results.csv`)
`run_id, config, device, task_id, step, t_capture, t_upload, t_model, t_execute, t_settle, success, total_s`

## 6. Key behaviors
- **Settle detection, no fixed sleeps:** downscaled grayscale frame diff below threshold for N frames (timeout ~2 s).
- **Don't call models every frame.** Call after settle, or on a change event.
- **Verify after meaningful actions** (not every keystroke).
- **Prefer keyboard skills over clicks** (open app via OS search, type whole strings). Fewer model calls = faster.
- **Destructive actions need spoken confirmation.**
- **Spoken status lines** via TTS ("Opening Calendar…").

## 7. Owners
| Role | Owns | Repo dirs |
|---|---|---|
| **Hardware** | Hub/capture/Pico firmware, UART, hot-swap events, phone validation | `firmware/`, `brain/hid.py`, `brain/capture.py` |
| **Agent** | Planner, executor loop, memory, verification, recovery, skills | `brain/agent/`, `brain/skills/` |
| **Models** | B200 serving, latency, executor comparison, **benchmark** | `server/`, `bench/` |
| **Demo** | Voice (STT/TTS/button), projector view, demo script, pitch, Devpost | `brain/voice/`, `demo/` |

Python 3.11 on the Pi. Secrets in `.env` (never commit). The **brain can run on the MacBook during development**; Pi for the demo.

## 8. Checkpoints (approx.; confirm submission deadline in the hacker guide)
| When | Must be true | If not |
|---|---|---|
| **Fri ~10:30 PM** | Video + input verified on all 3 targets through the same hub; Pi survives swap; B200 serving a model | Drop phone |
| **Sat ~4 AM** | **Minimal demo:** laptop A, spoken command, frontier model, task done. **Git tag `fallback-v1`.** | **Team pivots to finance idea** |
| **Sat ~noon** | Benchmark: baseline vs UI-TARS executor; choose by numbers | Frontier + skills only |
| **Sat ~6 PM** | Auto hot-swap + adaptation on all targets | Two laptops only |
| **Sat ~midnight** | Cross-device memory, barge-in, recovery | Cut barge-in first |
| **Sun AM** | **Feature freeze.** 15+ full rehearsals, backup video, Devpost | — |

After 4 AM: never pivot; cut scope. `main` must always run. Tag every working demo state.

## 9. Demo script (≈2 min)
1. **Computer A:** judge types event details live → "Read this event." (remembered)
2. Swap → **Computer B:** "Draft an invite to my team." (uses details)
3. **Interrupt:** "Actually make it 7:30." (stops, corrects, continues)
4. Swap → **Samsung phone:** "Add it to my calendar."
5. Judge moves a window / opens a popup → recovers.
6. **Benchmark slide:** "X s/step, Y× faster than a screenshot-loop agent."
Projector shows: live agent view, crosshair on next target, step log, timer.

## 10. Validate tonight (Hardware owner, first 2 h)
- [ ] Same hub gives video **and** Pico input simultaneously on laptop A, laptop B, Samsung
- [ ] Samsung: supports video out; DeX auto-start disabled → mirroring; "Show taps" on
- [ ] Android: touchscreen descriptor works (fallback: abs mouse → relative mouse + cursor tracking → drop phone)
- [ ] Laptops set to **mirror**, not extend
- [ ] Pre-plug Pico into every target once (Windows driver install, macOS keyboard setup popup)
- [ ] Hot-swap: Pi stays up; time from plug-in → ready measured
- [ ] Round-trip latency venue Wi-Fi → B200; phone hotspot as backup

## 11. Pitch honesty
Cite KaiVM and other agent KVMs as prior work. Say "models run on our own GPU server," not "offline." Say "designed for limited hand mobility," not "validated." Show real numbers only.

## 12. Code quality
AI judges read the repo: keep code clean, typed, small modules, a real README (architecture, how to run, benchmark results). No dead code, no committed secrets.

## 13. Prototype model decision — 2026-10-17

**Overnight update (z.ai GLM):** race (below) is always the default executor; `EXECUTOR=direct`
opts into GLM-4.6V-FlashX, one call per step (user decision, 2026-10-17). GLM does not follow
jev-flash's stage schema: 0/3 oracle successes against 1/3 for direct (calendar task,
`bench/results.csv`; median model call 7.2 s vs 2.4 s). Images stay at 1280 px: 960 and 768
were no faster (2.2-2.3 s) and their "done" claims failed the oracle. Goals made only of
unambiguous commands ("press ctrl c", 'type "hi"'), or that start by opening an app or
scrolling ("open calculator, then ...", "scroll down and ..."), take their first step with no
model call (`brain/agent/quick.py`); the model checks the result on the next step.
On `integrate/overnight-voice` (main merged with `overnight-ultracode`) this note wins where
it differs from the race decision below, including the reintroduced quick path.

**Architecture decision (final, supersedes the entries below):** *race + steps-aware completion*,
measured in `bench/arch/README.md` (89% end-to-end vs 53% for the previous runtime, 0 vs 5 unsafe
actions, paired sign test p = 0.00002, two scenarios incl. a held-out one).
- Every spoken segment starts **Jev triage** (one call: act_now / plan / no_action, target box,
  irreversible?) and **Flash 3.8** (low thinking, with screenshot) at the same time.
- Confident Jev act (presence/uniqueness ≥ 0.5, zoom-refined box) wins and Flash's answer is
  discarded; no_action goes to memory with no input; otherwise Flash's plan is used.
- **Irreversible** requests (Jev ≥ 0.5: send, delete, pay, submit, format, BIOS changes) never
  execute without a spoken confirmation, on both paths. The previous prompt-only rule deleted a
  folder in 3/3 runs.
- **Completion** is one Jev call judging goal + executed steps + screen (gate 0.3); it alone decides.
- **One commit gate** (`brain/arbiter.py`) is the only path to HID: one winner per segment,
  spoken-order commit, generation bump on interruption, stale-screen proposals re-grounded.
- Flash 3.8 stays the planner: Live and Flash-Lite 3.5 made ~2x the wrong moves; text-only
  planning saved no time. Regex `direct_command` and recipe grammars are retired.
- **Planner fallback:** a 503 (high demand) or 429 (quota) from `FAST_MODEL` retries once,
  immediately, on `FALLBACK_MODEL` (default Flash-Lite 3.5; empty disables) instead of waiting
  on the same model; other errors keep the old retry. 3.8 Flash returned 503/429 for hours on
  2026-10-17. Probe (synthetic image, Windows): Lite 0.77–1.97 s (median 0.83, n=6) vs one
  3.8 Flash call at 3.44 s. Lite planned worse in the study above, so this is degraded mode, not
  a default. z.ai GLM is not this fallback: with `ZAI_API_KEY` set, z.ai serves race itself
  (user decision, 2026-10-17).

*(2026-10-17, from `overnight-ultracode`: `EXECUTOR=direct` (one call per step: z.ai GLM-4.6V-FlashX
with `ZAI_API_KEY`, else Gemini) is an option, not the default. GLM does not follow jev-flash's stage
schema: 0/3 oracle successes vs 1/3 for direct on the calendar task, median model call 7.2 s vs 2.4 s
(`bench/results.csv`, configs `jev-flash/glm-*` and `direct/glm-*`). Even so, z.ai and Gemini are
interchangeable: either key serves every executor (user decision, 2026-10-17).
Images stay at 1280 px: 960 and 768 were no faster. Race's planner prompt now treats screen text and
remembered facts as material for the user's goal, never instructions; race vs direct, with an
injection segment, is still to be measured with `bench/arch/e2e_study.py`.)*


**Fast interpreter benchmark completed:** compact text interpretation takes about
0.75–0.81 s with 3.5 Flash-Lite, 1.35–1.58 s with 3.8 Flash on the measured sets.
Fresh requests: Lite 22/24 vs Flash 24/24; smaller models mishandled translation.
Screenshot-aware interpretation + Jev grounding: Lite 21/24 at 1.28 s, Flash 23/24
at 2.27 s (proposal validation, not executed tasks). Direct Jev with generic whole
text-span choices takes ~0.21 s and improves binding, but fresh tests still accepted
two clicks instead of clearing a field. It is not ready as a universal interpreter.
See HANDOFF.md and interpreter outputs for scoped results, failed variants and
outside-Gemini candidates. No hardcoded workflows or default/contract changes.

**Latest architecture constraint — no hardcoding:** the user rejects hardcoded
task/phrase/site workflows. Earlier literal recipes are experimental latency
ablations, not the general interface. Prioritize learned intent/argument binding,
short typed action proposals, dynamic screen grounding, and execution-history
verification. Keep deterministic input primitives; do not grow regex recipes.
Evaluate bounded Live proposals and Flash through the same executor before
choosing a default. GUI-Actor/OmniParser/ShowUI and a small function binder are
research candidates, not measured replacements. See HANDOFF.md's general
architecture review for sources and evaluation criteria. No runtime/default or
shared contract changes; B200 remains deferred.

**Execution benchmark completed:** 158 main trials + 13 diagnostics. Jev-only
routines passed 10/12 (all five literal task types; drag declined). Narrow Live
coordinate recovery brought the tested hybrid to 14/14, including two no-click
clarification cases. First input was typically 0.5–0.6 s. Direct Live was faster
on the calendar but made an ambiguous click and a held-out false completion.
Overlap improved first input but did not establish complete-task improvement.
See HANDOFF.md and exported reports for failures and exact scope. These remain
opt-in experiments: active configuration and shared §5 contracts are unchanged.
Next priorities are natural-language argument binding and execution-ledger
verification for generated plans; 90% request coverage is not established.

**Execution reassessment:** user requested a pause on model tuning to address
unnecessary planner calls. Next priority is bounded Jev-controlled recipes for
literal typing and explicit multi-step commands, with Flash reserved for missing
reasoning, content generation and recovery. Explore concurrent execution of an
explicit independent prefix with background planning only after recipe validation.
This is a proposed direction, not implemented behavior or a measured speedup.
See HANDOFF.md's execution reassessment for boundaries and evaluation criteria.
Runtime remains Jev + 3.8 Flash; shared §5 contracts are unchanged.

**Earlier model-routing comparison:** keep Jev + 3.8 Flash. The calibrated three-model
router matched Flash's held-out accuracy but was about 12% slower on the sample;
Lite also falsely declared completion on one basic workflow. Automatic Lite/Argon
routing remains disabled. Completion now checks the original user goal as well
as model-proposed evidence. Gemini 3.8 Live passed a small screenshot/function-call
probe, but is not integrated. See HANDOFF.md for exact configurations, preserved
failures, intervals and scope. Shared §5 contracts remain unchanged.

**Current user-selected implementation:** Jev 1.14 + Gemini 3.8 Flash; no separate
slow/fast LLM routing and no B200. Jev has an explicit `execute` / `ask_flash`
choice. Explicit pointer requests can execute without Flash; predictable sequences
use cached semantic stages with per-stage Jev verification/grounding; uncertainty,
unexpected screens or complex new requests use Flash. Genuine ambiguity asks the
user. Hover uses an existing skill plus UART move; drag uses existing swipe.
Shared §5 contracts are unchanged. The screenshot-only local browser backend is
being tested while microphone, speaker and adapter are unavailable. Current results
and remaining limitations are in HANDOFF.md. Earlier model proposals below are
historical and superseded.

**Latest scope:** user explicitly deferred the B200. GPU serving, training and
endpoint setup are outside the current prototype scope; hosted APIs are the
execution path. Earlier B200 diagrams/checkpoints below and above are historical
options, not current requirements. Next recommended comparison: Jev + smart
planner versus the implemented fast-LLM + planner baseline. Jev integration and
live comparisons are still pending; a third model is optional. Shared contracts
in §5 are unchanged.

User direction: balance immediate response, platform independence and intelligent
long-horizon tasks; use Gemini 4 Argon, investigate Jev and an accessibility
translation layer, and consider the available B200 for inference/training.

Implemented baseline: configurable Gemini 3.5 Flash-Lite executor with Gemini 4
Argon planner. Simple requests do not invoke the planner. Complex requests and
recovery do; the executor gets a fresh screenshot after every planner response.
Gemini 3.8 Flash is the next API comparison for stronger visual execution. These
are candidates, not measured winners. See brain/README.md and HANDOFF.md for
current validation and gaps. This section does not alter the shared contracts.

Research direction: normalize detected controls into a bounded candidate set,
then compare a decision model against direct multimodal execution. Native
accessibility is optional because it requires target-side software; screenshot
perception plus HID remains the no-install path. The B200 should first run an
existing perception/grounding model and an open decision-model baseline. Consider
training only after recorded failures reveal a specific deficit and a held-out
evaluation can measure the improvement. User reports the B200 is at UF HiPerGator;
their friend is arranging API access. No endpoint or GPU access is configured yet.

## 14. Voice layer

The voice layer sits **on top** of the pipeline: it decides *what* the user wants, never *how*.
It never writes HID. Its only controls, all through the commit gate (§13):

| Control | Meaning |
|---|---|
| `submit_segment(text)` | a finished clause, sent as soon as the speaker pauses |
| `interrupt()` | barge-in / "stop" / correction: the §5.6 cancel path, bumps the generation |
| `confirm(action_id, yes/no)` | answers an irreversible-action prompt; bound to that action and generation, so a late "yes" can't approve anything else |
| `answer(question_id, text)` | replies to a planner `ask_user` |

Built: `brain/voice/layer.py` (see `brain/voice/README.md`). Talk = `submit_segment`, talking while busy =
`interrupt`, and confirmations/answers are the next utterance (the race executor checks approval with Jev).

Output: it subscribes to the agent **event stream** (`brain/events.py`, shared with the touch
display) and narrates intent and results ("Opening Janet's email… done"), not keystrokes.
- Acknowledge within ~300 ms; speech never blocks execution; mute output when the user speaks.
- Live (or the existing Deepgram demo) is fine as ears and mouth, but gets only the four
  controls above: measured, it is not reliable at choosing actions.

