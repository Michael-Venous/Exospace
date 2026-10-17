# Exospace prototype handoff

## Overnight branch state — 2026-10-17 (`overnight-ultracode`)

Built unattended on `overnight-ultracode`. `integrate/overnight-voice` is `main` merged with it;
where the two conflicted, overnight's behavior won, and where this section and the ones below
differ, this one holds there. After the merge, 197 unit tests pass on the laptop
(`.venv/bin/python -m unittest discover -s tests`).

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

## Status 2026-10-17 (Claude): read this first
**`main` is the source of truth.** All branches are merged into it (PR #9); `agent/prototype` is
superseded, so switch to `main` before continuing.

Done and on `main`:
- **Runtime:** race architecture is the default (`EXECUTOR=race`): Jev triage (act / plan / no action,
  target, gesture, irreversible flag) races the Jev + Flash planner; steps-aware completion; experiments
  moved to `bench/experiments/`. Evidence: `bench/arch/README.md` (89% vs 53%, 0 vs 5 unsafe).
- **Voice layer:** `brain/voice/layer.py` (PLAN.md §14), smoke-tested through the real agent.
- **Pi:** `scripts/pi/deploy.sh` deployed it; 151 tests pass on the Pi. `python -m brain.doctor` lists
  what's left there.
- **Protocol:** `tests/test_pico_loopback.py` runs the real `PicoHID` + `Controller` against the real
  firmware command handler over a pty: gestures, taps, errors and unplug events all agree.
- **Docs:** `docs/demo.md`, `docs/devpost.md`, `docs/hardware-bringup.md`.

Waiting on hardware / sudo (in order):
1. On the Pi with sudo: enable uart0, add `mike` to `video` and `dialout`, `loginctl enable-linger`
   (exact commands printed by `scripts/pi/setup.sh`), reboot, copy `.env` (chmod 600).
2. Plug in the capture card and Pico; flash `firmware/pico/` (see its README); run `python -m brain.doctor`.
3. `systemctl --user enable --now exospace-agent exospace-voice`; run `docs/demo.md` on real targets
   (Windows laptop, Mac, Samsung phone) and re-measure latency with the capture card.


## Fast interpreter and direct messy-request Jev benchmark — 2026-10-17

**Completed:** model-driven interpretation, compact output, raw-screen Jev,
whole-span Jev binding, screenshot-aware proposals, and fresh wording/labels.
No task/phrase/site regex parser was added. No runtime defaults, shared contracts,
physical HID, mic/speaker, Pi deployment, or B200 work changed.

- Main language-only comparison: 24 requests x 2 repeats/model with the SAME
  oracle element inventory. Lite minimal **44/48, median 0.807 s**; Flash low
  **48/48, 1.353 s**; Flash 3.6 minimal **46/48, 1.044 s**; warm Live **42/48,
  0.728 s**; factorized Jev **38/48, 0.171 s**. Inventory perception is excluded.
  Lite's two other strict failures were safe clarification instead of planning;
  its two wrong actions treated generated writing as literal text.
- Short output development ablation: Lite **23/24, 0.751 s**. Fresh 12 requests
  with renamed controls x2: Lite **22/24, 0.783 s**, Flash **24/24, 1.577 s**,
  Flash 3.6 **22/24, 1.013 s**, status-gated Jev **18/24, 0.204 s**. Both smaller
  Gemini models mishandled translation; this is not a proven reliable routing gate.
- Raw request -> screenshot-aware interpreter -> Jev grounding, no supplied
  inventory: Lite **21/24, 1.275 s** median total; Flash **23/24, 2.272 s**.
  Interpretation alone **1.059 / 2.024 s**. Both declined a hover; Lite also
  attempted generation/reasoning. Frozen-frame proposal checks, NOT execution.
- Direct screenshot + messy request Jev: whole contiguous-span choices improve
  literal text binding over independent start/end token indices. At **0.206 s**,
  **31/48** strict decisions passed, including **24/36** action requests accepted
  correctly with no wrong accepted actions in the development sample. Same-fixture
  independent indices: **22/48**, seven wrong accepted actions.
- That promising Jev result did NOT fully generalize: fresh requests **11/24**
  strict at **0.211 s**; nine accepted actions, seven correct and two field clicks
  substituted for clearing. Quoted text declined. Keep direct Jev experimental.
- Lite first token median **0.741 s**, median tail after first token **0.087 s**.
  Cutting output tokens alone has limited headroom; external inference comparisons
  must measure first-token/full-valid-action latency, cold starts and p95.

Evidence: `bench/interpreter_{study,optimized,screen,freeform,heldout,screen_heldout,report}.py`;
`work/interpreter-*/` frozen sources/manifests/rows; `bench/results.csv` explicitly
labels these as laptop microbenchmarks. Early screen fixtures rendered some roles
as buttons; repaired fixture comparisons preserve both versions. Raw no-inventory
text interpretation sometimes invented Notes-app navigation. Live uses a warm
persistent session; initial setup 0.370 s excluded from request timings. Whole-span
candidate budget is bounded; candidate enumeration is generic, not phrase matching.
95 existing unit tests passed, new scripts compile, parser boundary checks passed.

Report: `/home/mike/Documents/Codex/2026-10-17/con/outputs/exospace-interpreter-summary.txt`
with JSON and CSV alongside. Outside-Gemini candidates researched but NOT measured:
Cerebras GPT-OSS-120B/Qwen 3.8 27B, Groq GPT-OSS-20B, GPT-6 Luna none/low reasoning.
Only Gemini/Jev credentials configured. Official sources are in the report.

**Next action:** retain current defaults; evaluate a general typed interpreter
through the actual executor/ledger, with screenshot context or measured screen
inventory construction. Preserve Jev whole-span as an opt-in fast-path candidate;
test unknown/ambiguous/empty/quoted literals and sequential progress before enabling.
External API comparison can use the same frozen cases once credentials exist.

## General architecture review: no hardcoded workflows — 2026-10-17

**User constraint:** no hardcoding. The earlier regex/literal recipe experiments
remain benchmark ablations, not the proposed general command interface. Their
0.5-second first-input result excludes the unsolved general language/argument
binding problem; only 5/12 authored phrasings compiled. Do not expand phrase,
website, or task-specific recipes as the product architecture.

**Review recommendation (planned, not implemented):** retain capture + HID,
device continuity, interruptibility, and deterministic input primitives. Use a
learned semantic proposer to produce short, typed, screen-conditioned actions;
ground and verify with Jev or a specialist; execute until new observation or
missing reasoning requires another proposal. Generic fill/click/hover/drag
semantics are controller behavior, not hardcoded task solutions. Maintain verified
transition history so completion does not infer every past step from the last
frame. Distinguish input acknowledgement from observed effect. Resolve hidden
targets on fresh frames, and revalidate proposals after intervening actions.

Jev selects supplied candidates; it does not supply arbitrary plans or free text
through the documented Choice/Score/Noul interface. Dynamic screen elements plus
learned text-span/argument binding may permit a general Jev next-action loop, but
candidate generation/recall and binding cost must be included in benchmarks.
Do not hide that cost in preprocessing or assume independent Jev answers form a
coherent joint action. Keep explicit abstain/clarify/planner options. Escalate on
missing capability/evidence, not number of steps or a single complexity score.

Near-term comparison: Flash-generated typed actions with repaired verification;
bounded Gemini 3.8 Live semantic proposals with the same executor; dynamic
screen-candidate Jev control if a measured general candidate source is available.
Live should emit typed commands directly, not serialize them to regex-parsed
English. Compare with/without redundant Jev guards to measure their net value.
Use unseen apps/layouts/ordinary paraphrases, arbitrary text, ambiguity, hidden
menus, task corrections and interruptions. Report first *correct* action, full
verified success, false completion/wrong actions, fallback rate, p50/p95 latency,
and all parsing/grounding/verification/connection costs. No new inference trial
was performed in this review; prior results retain their original scope.

Primary-source candidate review (read 2026-10-17; no performance measured here):
- GUI-Actor 2B/3B: action-attention grounding specialist and grounding verifier;
  promising Jev Box comparison, not a language planner:
  https://github.com/microsoft/GUI-Actor
- OmniParser V2: screenshot-to-elements candidate source; preserve full-screen
  fallback for omitted controls and measure detector/OCR/captioning overhead:
  https://github.com/microsoft/OmniParser
- ShowUI-2B: learned GUI action/grounding baseline;
  https://github.com/showlab/ShowUI . ShowUI-pi is a separate flow-based drag
  specialist, useful later for continuous paths:
  https://github.com/showlab/showui-pi
- FunctionGemma 270M: text-only function/argument binding research candidate;
  Google explicitly positions it for task-specific fine-tuning, not a ready
  general GUI controller: https://huggingface.co/google/functiongemma-270m-it
- GPT-6 Luna with none/low reasoning: hosted structured-action comparison,
  documented image input/function calling/structured outputs, untested latency
  and account access: https://developers.openai.com/api/docs/models/gpt-6-luna
- UI-TARS repo announces UI-TARS-2, while the reviewed deployment path explicitly
  supplies 1.5-7B; do not confuse publication with available weights/endpoints:
  https://github.com/bytedance/UI-TARS
- Jev primitive semantics: https://docs.typesafe.ai/primitives ; Live capabilities:
  https://ai.google.dev/gemini-api/docs/models/gemini-3.8-live

Keep B200/training deferred and native accessibility optional: mandatory target
software would compromise the no-install target design. Small parameter count
does not establish fast laptop/Pi screenshot inference. Active model configuration,
runtime, shared contracts, and other ongoing bench/arch work were not changed.

## Execution architecture benchmark completed — 2026-10-17

**Outcome:** keep the active factory and `.env` unchanged. The best tested candidate
is Jev + deterministic verified routines, with a narrow Live coordinate fallback
when Jev confirms presence/uniqueness/readiness but cannot ground a box. Flash
remains available for unsupported requests and reasoning. Do not enable unrestricted
Live control or claim broad natural-language coverage from this sample.

Completed **158 main trials plus 13 smoke/diagnostic attempts**, over 13 synthetic
task templates. Screenshot-only model inputs; real local browser mouse/keyboard
through the existing controller; independent DOM result oracle after execution.
These are repeated/development trials, not 171 independent user tasks. No mic,
audio playback, physical HID, native address bar, device swap, B200 or deployment.

Measured results:
- Jev-only, LLM fallback disabled: **10/12**. Both runs of two clicks, URL form,
  three-field typing/save, double-click, and hover/menu passed without any LLM.
  Median totals: **2.05 / 3.09 / 5.01 / 1.39 / 2.08 s** respectively. Median first
  input **0.51 s**. Both drag trials declined before input due to uncertain destination.
- Narrow hybrid (`jev-live-ground`): **14/14**. Ten routine runs used only Jev,
  two drags used one Live coordinate call each (**2.61 / 2.86 s** total), and two
  ambiguous requests used Flash to ask without clicking. Median first input **0.57 s**.
- Corrected direct Live: **15/16** on v2; it clicked an ambiguous Continue request
  in one trial. On four held-out tasks it passed **3/4**: an ordinary URL paraphrase
  omitted the field-focus click, submitted an empty destination and falsely said done.
  The five literal task types alone passed 10/10 at median **4.13 s**, first input
  **1.69 s**. Direct Live v1 additionally clicked instead of hovering in two trials;
  report adjudicates those nominal end-state successes as failures.
- Calendar (six fields, popup, time arithmetic): Flash **2/2, 22.05 s** median;
  direct Live **2/2, 16.29 s** (about 26% lower elapsed time on this tiny sample).
- Full Live semantic plans: **9/14** in v1. Compact Live commands: **8/16** in v2,
  including one connection-setup timeout. Missing steps, conservative coverage gates,
  and grounding recovery kept invoking Flash. No broad fast-LLM default is justified.
- Cold direct Live: one **25 s timeout**, one **4.58 s** success; separate compact
  Live cold runs both passed but required Flash recovery. Long-session resilience
  and voice performance remain untested.
- Explicit-prefix overlap after target refinement: repaired runs issued the first
  useful click at **0.89–0.98 s** while Flash planned, vs **4.01–4.24 s** sequential.
  Both reached the right answer but failed final completion verification; median
  total **23.80 s**, sequential **22.64 s**. Responsiveness improved, complete-task
  speed/reliability did not. Do not promote overlap as a demonstrated overall win.

Important interpretation: the v1 Flash baseline reached the correct independent
end state in all 12 non-ambiguous trials; six failed because completion verification
tried to infer the entire earlier sequence from the final image and repeatedly
replanned completed work. This is an orchestration/verification defect. Stepwise
routines retain acknowledged actions and verified transitions, then check the final
visible effect. They do not accept a model's partial rewording as the whole goal.
No shared evidence threshold or §5 contract was lowered or changed.

Coverage remains narrow: only **5/12 authored phrasings** compiled directly. This
is not a real-world coverage estimate. Two held-out literal tasks passed with Jev
only (new labels/layout and command-like text inside quotes); two ordinary paraphrases
fell back to Flash. Next work: intent/argument binding, execution-ledger verification
for generated plans, then native/device validation. The 90% request-coverage goal
has not been established.

Implementation/evidence:
- `bench/experiments/execution_benchmark.py`: opt-in variants; sources/manifests frozen per study;
  `--only-pending-from` preserves completed rows after harness repair.
- `bench/experiments/*_experiment.py`: literal compiler, routine/target refinement,
  Live session transport, direct/compact Live, narrow coordinate recovery, prefix
  overlap. `brain/agent/factory.py` and `.env` were not changed.
- `bench/experiments/execution_report.py`: independent-oracle summaries and hover-compliance
  adjudication. `bench/results.csv` has shared-schema timing rows; its success
  column retains runtime semantics, while exported JSON/CSV use the independent oracle.
- `work/execution-comparison-v1/`, `work/execution-comparison-v2/` plus
  `work/execution-comparison-v2-completion/` (41 + 7 = 48 rows),
  `work/execution-no-llm-v1/`, `work/execution-heldout-v1/`,
  `work/execution-grounding-v1/`, `work/execution-calendar-v1/`,
  `work/execution-cold-v1/`, and the three `work/execution-overlap*-v1/` studies.
  All smoke failures and diagnostics remain preserved too.
- A Live session-close timeout interrupted v2 at 41/48 rows. Bounded cleanup now
  cancels pending transport tasks and cannot abort the study; seven missing trials
  were run separately. A malformed empty-action background plan now triggers a
  fresh-screen replan after the already-executed prefix instead of a runtime abort.
- **92 tests passed at benchmark completion; 95 pass in the current checkout**,
  with compileall and git diff --check passing. Tests cover quoted
  delimiters, unsupported grammar/Unicode, stage acknowledgements, failed evidence,
  stale Live IDs, multiple proposals, cleanup, hover/drag constraints, missing/invalid
  coordinates, ambiguity gates, stale background goals and unexecuted prefixes.

Concurrent commit `61ab5ae` appeared after the final benchmark (reports finished
06:06:33 UTC; loop.py modified at 06:07:42 UTC). It includes these experiments and
other runtime changes, including stale-frame handling. This agent did not create
that commit. Preserve it; do not attribute the newer runtime behavior to these
measurements. The measured loop is recoverable as `f45eeb2:brain/agent/loop.py`.
Current 95-test suite includes three added concurrent runtime tests.

User-facing report: `/home/mike/Documents/Codex/2026-10-17/con/outputs/exospace-execution-summary.txt`.
Full data and independent-oracle CSV are alongside it as `exospace-execution-results.json`
and `exospace-execution-results.csv`. JSON includes merged v2 comparisons, all trial
rows, source manifests, interpretation notes and the language-coverage probe.

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

`bench/experiments/routing_benchmark.py` freezes a hashed 36-case manifest, with different
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

`bench/experiments/sequence_benchmark.py` runs real Jev + Gemini plans through screenshot-only
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

`bench/experiments/live_probe.py` now tests an inert screenshot-reading function, without a
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
BROWSER_EXECUTABLE=/opt/google/chrome/chrome .venv/bin/python -m bench.experiments.local_scenarios --case all
BROWSER_EXECUTABLE=/opt/google/chrome/chrome .venv/bin/python -m bench.experiments.local_benchmark --runs 3
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
- `bench/experiments/probe.py`: opt-in synthetic-image model-access/complete-response latency
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
   then run `.venv/bin/python -m bench.experiments.probe --runs 3` on Pi. Compare with
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
5. **Experiment code is mixed into the runtime package.** `bench/experiments/*_experiment.py` and `brain/*_benchmark.py` sit in the shipped package. CLAUDE.md says code quality is judged and to avoid dead code. Move experiments to `bench/` or delete them before submission, so `brain/` holds only the runtime.
6. **`jev_flash.py` style doesn't match the rest of the codebase.** It uses dense one-liners, no spaces after commas, missing type hints, and imports inside functions. The AI-track judges read the code, so run `ruff format` and add types.
7. **Most work is uncommitted.** The only commit is `f45eeb2`; many modified and untracked files exist only on this laptop. Commit to a branch (e.g. `agent/prototype`) so work isn't lost and teammates can pull it.
8. **Low priority:** `jev_flash.py:249` checks whether the fallback screenshot changed by comparing JPEG bytes for exact equality. Real capture-card frames are never byte-identical, so the check always reports "changed." That currently only triggers after a discarded step, where replanning is correct anyway, but compare thumbnails if it's ever used elsewhere.

## Claude architecture study (2026-10-17)
Full results and recommendation: [bench/arch/README.md](bench/arch/README.md). Summary (updated after review + completion study): **race + steps-aware completion check** (Jev
triage in parallel with Flash 3.8; a confident Jev act/no-action wins; completion = one Jev call judging
goal + executed steps + screen). Steps-aware check: 12/12 finished tasks accepted, 0/18 traps (concrete-only
accepted 12/18 traps). End-to-end it matched or beat the concrete variant head-to-head (17/20 vs 16/20,
0 unsafe). The earlier race + concrete run scored 18/20 with 0 unsafe, 94 s total and 21 LLM calls.
Held-out validation (second scenario, 3 repeats) and pooled analysis: race + steps-aware 47/53 = 89%
(95% CI 77–95%), 0 unsafe; current runtime 28/53 = 53%, 5 unsafe, including **deleting a folder in 3/3
runs** despite its prompt rule. Paired sign test vs current: 20–1, p = 0.00002. Add the irreversible-action
Jev question (27/27 destructive caught, 0/33 false flags at gate 0.5) before shipping the fast path. The current runtime scored 8/20 with 2 unsafe actions, 322 s and 90 calls.
Direct commands: first input 0.73 s, no LLM call. Keep the screenshot for Flash; drop Lite, Live, text-only,
high thinking and relaxed planner gates. Port steps are listed in that README.
