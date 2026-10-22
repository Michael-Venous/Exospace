# Exospace status and handoff (2026-10-18)

**Work here:** `/home/mike/Documents/code/Exospace-integrate`, branch `laptop-rig`. This laptop's build is the
main one. The last pushed commit is `7a5a571`; everything after it is uncommitted local work (below).

**No git.** No commits, pushes or pulls unless the user asks. Push to `origin` is disabled on purpose
(`git remote -v` shows `PUSH_DISABLED_by_user_request`): leave it that way.

## The rig
- **Pi 5:** `nicopi@100.76.219.77` (Tailscale), code in `~/Exospace`, `.env` already there (never copy one over).
  Reachable during the latest checks, but SSH and API uploads remain intermittently unreliable.
- **Services** (systemd user units): `exospace-agent` (display page `:8770`) and `exospace-voice` (voice page
  `:8780`). Logs: `ssh nicopi@100.76.219.77 'journalctl --user -u exospace-agent -n 200 --no-pager'`.
  To view a page from the laptop: `ssh -L 8770:127.0.0.1:8770 nicopi@100.76.219.77`.
- **Deep mode is off** on the Pi: `systemctl --user set-environment DEEP_MODEL=` (Argon was 4-24 s a step).
- **Before restarting:** check the agent is idle (the user may be mid-run); a restart kills their request.
- **Full deploy:** `PI=nicopi@100.76.219.77 OS=<android|linux|...> scripts/pi/deploy.sh` (the script's default
  `PI` is wrong for this rig). `OS=` pins the target OS. For one or two files, `rsync` them and restart.
- **Button:** GPIO17 (pin 11) to GND (pin 9), read with `gpiomon` (`brain/voice/button.py`). Hold to talk.
- **Kill processes by PID**, not `pkill -f` (it matches your own shell).

## Current deployment and verification — 2026-10-18 09:14 EDT

The latest local suite ran 348 tests: 347 passed and one optional `mss` test skipped.
The focused offline demo gate passed all 21 tests (`python -m bench.demo_smoke`).
These tests do not prove live speech or hardware latency.

Deployed on the Pi: concise step/launch timing logs, combined app/goal launch verification,
stale-launch recovery, nested-call and request-epoch cancellation fixes, idle target preparation,
60 fps capture configuration and measured-fps doctor, and the reviewed voice HTML queue/drain fixes.
The service is using race + Flash plans via `JEV_FIRST=0` in
`~/.config/systemd/user/exospace-agent.service.d/90-demo-planner.conf`; repository default remains 1.
To undo that override, remove only this drop-in, reload systemd and restart **after checking idle**.

**Do not overwrite Pi `brain/voice/layer.py` with the local version.** The remote file has newer
Speaker/audio/narration work. Local handler/progress changes still need reconciliation. The remote
HTML's audio-unlock and `m.voiced` behavior was preserved. Reload an existing voice tab to use it.

Latest real-device results (transcribed submission to result, excluding microphone/ASR):
- Open calculator and calculate 18×7:24.604 s; screenshot confirms 126. Agent started in 0.010 s
  after idle preparation;20.918 s were model steps. Run `3c5b1952`.
- Open system settings and select Appearance:failed after 34.985 s; no success claim. Run `82f44d13`.
- Earlier Firefox launch succeeded in22.417 s; Wikipedia navigation with Flash plans succeeded
  in47.045 s including 10.408 s initialization. These are different tasks, not a controlled comparison.

**Near-real-time remains unmet.** Pi upload timeouts and slow model calls dominate current runs.
Fresh-screen and approval checks remain enabled. Target OS identification has repeatedly returned
Windows for the Linux laptop; investigate this as a separate correctness issue.

Voice actions currently start on finalized speech segments, which can arrive before the user stops
speaking. There is no safe reuse of plans from partial transcripts yet. Triage and planner already
race after routing; pending-answer classification remains sequential. A deferred-planner comparison
failed on upload timeout, so the race default was not changed. UART work remains deferred.

Next: stabilize/measure Pi uploads, fix OS identification, then validate settings navigation. For
partial-transcript preparation, require matching final text, fresh screen and invalidation on correction,
interruption or device change; never execute tentative partial-text actions.

## Earlier verified capabilities (not a current latency guarantee)
- **Laptop (Linux, through the Pi):** opening apps, Firefox search, YouTube search, Blender maximize (up arrow,
  top right). Pointer calibration (`brain/calibrate.py`) finds the screen inside the frame, so black bars
  don't shift clicks. Clicks use a two-step zoom (`ZOOM_WINDOW` 0.30, then `FINE_WINDOW` 0.12).
- **Phone (Samsung, keyboard only):** apps open by Android shortcut keys (`APP_KEYS` in
  `brain/skills/__init__.py`): Win+I settings (~2-3 s), Win+S messages (~3 s), Win+B browser (~7 s), Win+C
  contacts, Win+L calendar, Win+E email. Other apps: Win (launcher), then Ctrl+F search. Brave, then a
  YouTube search, is verified.
- **Voice:** live actioning (each pause sends a segment; a segment with `continues` joins the running request),
  "On it" / "Still working" pacing, Describe mode (`gemini-3.8-live`).

## Open problems (in priority order)
1. **Touch and mouse don't register on the phone.** The Pico's touch descriptor looks correct by the research.
   Diagnosing it needs adb on the phone (`adb shell getevent -p`, `adb shell dumpsys input`), which the user
   doesn't have set up yet (wireless debugging). Until then, anything that needs a tap fails (Camera, Home).
2. **Phone Home and Back by keyboard:** untested. Try Meta+Enter (Home) and Meta+Backspace or Escape (Back)
   with the agent stopped; if they work, they replace taps for navigation.
3. **Gmail on the laptop was slow** (35-50 s): re-measure with the website hint above.
4. From earlier runs, still open: a false "done" when a link is only visible or a click lands on a login page,
   and "go back" being hard to confirm.

## Rules
Read [CLAUDE.md](CLAUDE.md) and [PLAN.md](PLAN.md): the §5 contracts don't change without the user's approval,
measure before claiming a speedup (`bench/results.csv`), no dead code, never commit secrets. Run the tests
with `../Exospace/.venv/bin/python -m unittest discover -s tests` (no API calls; this checkout has no
venv of its own).

## Review checkpoint — 2026-10-18, 08:30 EDT

Requested scope: review the latest exo branch and fix what is practical within ten minutes. Verified
`laptop-rig` in this checkout is the newest local branch and matches this handoff. Existing local changes
were preserved. No commits, pulls, pushes, deployment or service restarts were performed.

**Completed locally (additional review fixes):**
- `brain/voice/layer.html`: serialize `/talk` and `/segment` requests so the threaded server cannot process
  a continuation before its initial goal or a delayed stop after that goal. Requests have a 10 s timeout.
- The next tap is retained while the old speech stream drains. Closing waits at most 3 s for final words;
  a stalled service no longer locks out future sessions. Old idle timers cannot cancel the queued press.
  Switching microphones releases the current session and waits for a new press.
- `brain/voice/layer.py`: periodic progress narration can repeat during long requests, while duplicate
  result/stage narration remains suppressed. Invalid JSON, non-object bodies and invalid/oversized body
  lengths receive HTTP 400 instead of an uncaught handler error.

**Verified:** `/home/mike/Documents/code/Exospace/.venv/bin/python -m unittest discover -s tests` ran
332 tests successfully with one skip (optional `mss` local-capture dependency absent). `git diff --check`
passed. Output: `work/review-tests-final.log`. Seven new regression tests include five actual Chromium
checks with fake microphone audio and a controlled WebSocket: rapid retap, bounded drain, mic switching,
one retry without another barge-in, and ordered commands; two Python tests cover recurring narration and
invalid bodies. These are not live Deepgram or hardware validation.

**Live status / next action:** user reports the target is connected to the laptop. SSH to
`nicopi@100.76.219.77` still timed out during this review; no listeners on local ports 8770/8780/8781.
Once the rig is reachable, deploy the two reviewed voice files along with the earlier pending files above,
check the agent is idle before restarting, and test tap → speak → stop → immediately tap again with the
real speech service. The 3 s drain cutoff trades very late final transcripts for recovery from a stalled
service; verify it on the rig. No hardware performance claim was made.

**Remaining review finding:** `/narration` consumes a single shared queue, so multiple voice pages compete
for narration and GPIO button events instead of all receiving them. Use one voice page for the current
rig; a future multi-client implementation needs explicit event broadcast and microphone ownership.

### Concise latency logging — 2026-10-18

User reported slow Firefox opening and requested concise logs. Pi became reachable again. Observed its
latest laptop run `a010d1fe`: model calls 11.520 + 5.256 + 25.926 = 42.702 s, execution 2.206 s,
total 46.268 s, failed at the stall limit after repeated `Jev launch done` steps. This identifies model /
completion work as the dominant measured delay; no speed fix is claimed.

Added console timing per step (run ID, model, execution, settle, capture/preparation), preserving the CSV
contract. `open_app` emits one breakdown of launcher, search, and launch time, or shortcut duration.
Verified eight skill tests and 41 runtime tests; `git diff --check` passes. The prior pending Android
shortcut wait remains part of the local skills file. Deployment status follows below.

**Deployed:** `brain/telemetry.py` and `brain/skills/__init__.py` to the Pi, after checking their original
SHA256 hashes and compiling the staged sources. Confirmed the latest agent journal entry was a terminal
`fail:` result before restarting `exospace-agent`; systemd reported `active`. This also deploys the earlier
pending Android shortcut wait in the skills file. Voice files from the review are still local only.
Next: repeat "open Firefox" and inspect the new journal timing lines; no task was submitted during deployment.

### Launch verification improvement and streaming check — 2026-10-18

User requested faster operation without sacrificing reliability and asked whether execution starts before
speech ends. `brain/agent/race.py` now asks the unchanged app-identity and whole-goal questions in one Jev
call on the same frame, requiring both original gates (0.5 and 0.55). Wrong-app and incomplete-goal outcomes
retain planner fallback and approval paths. Fresh-screen rejection and cancellation are unchanged.

Verified: full suite 335 tests, one optional `mss` skip (`work/launch-combined-tests.log`); final race suite
48 tests after removing obsolete mock setup (`work/launch-race-final.log`); diff whitespace check clean.
Controlled hosted-model study completed before the user deferred end-to-end testing: six synthetic screens,
two repetitions per method, 12/12 correct for both methods, four trap types rejected. See
`bench/arch/README.md`, `bench/arch/launch_check_study.py`, `work/launch-check-study/results.json` and
`bench/results.csv`. This is limited regression evidence, not a guarantee of identical model behavior.

Deployed only the changed `race.py` after verifying the remote original SHA256. Confirmed latest journal
entry was terminal before restarting the agent; systemd reported active. Local and remote new hash:
`22477fd100ee1ed9c3e9061d31cf2b0ba40455fdcb8b18eb3802ad7e9c3d7862`.

Streaming behavior verified in code and a new browser test: speech service has 200 ms endpointing;
`speech_final` submits a finalized clause while recording continues. Subsequent clauses use
`continues=true` and join the agent at the next step. HID is sequential; this is speech/execution overlap,
not full speculative action pipelining. `brain/arbiter.py` remains unwired (per `brain/README.md`).

The user explicitly deferred UART work and the requested live end-to-end latency test. No end-to-end test
request was submitted, and no UART setting was changed. Earlier user run `c991b4e3` ultimately failed:
a later completion/recovery call took 81.81 s after a screen-change rejection. Batching does not remove
that guard or establish a new real-device latency; future authorized live validation should check this.
The earlier voice-page review fixes remain local only. No commits or pushes.

### Slow-run investigation — in progress, 2026-10-18

User renewed permission to run live latency tests after continued slowness. Another chat was restarting /
stopping the Pi agent; user confirmed they would pause that work. First scripted baseline was interrupted
by that stop and must not be counted as a latency result. Earlier actual run `c991b4e3` ended in failure
at 97.646 s, with 81.811 s in its last model step. `1341cd74` misclassified the Linux laptop as Windows,
failed its launch, and entered planner recovery. The later restarted service detected Linux correctly.

Implemented/deployed (original remote file hashes checked; agent was stopped before deployment):
- `brain/cancel.py`: nested `CancelToken.run` for the same token runs inside its existing worker instead of
  queueing a worker to wait on another worker. This prevents bounded-pool starvation; the outer wait still
  returns immediately on cancellation and discards late results. Different tokens retain independent waits.
- `brain/agent/executor.py`, `loop.py`, `jev_first.py`: a shared stale-screen note identifies a discarded
  proposal. Retry on the fresh screen with Jev; do not force Flash merely because the frame changed. Real
  failures and unsuccessful Jev steps still escalate. The stale-screen rejection itself is unchanged.

Regression suite: 339 tests, one optional `mss` skip, all others passed (`work/slow-fix-tests-final.log`).
New tests cover saturated nested worker calls, prompt cancellation, fresh-screen retry without Flash, and
fallback when that retry fails. Deployed service reported active. Live confirmation is still pending;
script `work/live_latency_check.py` measures transcribed voice submission to agent result with frame/event
capture. It does not include microphone/transcription latency. No UART changes.

Live follow-up: run `2ef3c93f` failed after 43.027 s of agent execution with zero input sent. Detailed logs
show concurrent triage/speculative-model uploads each failed twice with `WriteTimeout` (~20–22 s each).
This is measured upload failure, not slow inference or launch settling. SSH also dropped; five ICMP pings
all failed and Tailscale reached the Pi only via DERP(mia), ~1–2 s ping responses. Pi reported LAN IP
10.178.204.138 (direct SSH refused); this laptop is 10.192.216.36/17. User asked to retry; a new live test is
running. Do not claim end-to-end speed improvement from the regression tests alone.

The nested-call regression was also run against the original `HEAD` implementation: the saturated-pool
test failed as expected (`work/nested-cancel-before.log`), and passes with the deployed fix. Added an
explicit stale-completion rejection test; all 42 runtime tests pass (`work/slow-fix-runtime-final.log`).

Retry checkpoint, 2026-10-18: run `840e36d9` failed at 53.407 s submission-to-result
(52.871 s model, zero input). Read-only same-frame upload probes then succeeded over both IPv4
(2.472 s) and default routing (2.181 s), so no routing change was made. These probes do not establish
reliable live performance. Latest full request `85867534` also failed: 22.888 s submission-to-result,
22.302 s model, zero input. Local evidence: `work/live-fixed-retry-2.log` and
`work/live-fixed-retry-2-journal.log`; remote capture directory `work/live-latency/20261018-084749`.
Do not claim Firefox launch success or a confirmed end-to-end improvement. Network upload stalls
remain observed; startup also misclassified this Linux target as Windows. Next investigation should
separate concurrent upload performance from single requests and inspect the stale launch/recovery
path, preserving fresh-screen checks and approval gates. No UART or routing changes.
Final full regression suite: 340 tests, OK (one optional skip), `work/slow-fix-tests-complete.log`.
No commits/pushes; voice page fixes remain local, core latency fixes/logging are deployed.

### Demo crunch checkpoint — 2026-10-18

Added `python -m bench.demo_smoke`: 17 selected runtime/routing/browser regressions covering launch,
completion gates, stale launch, irreversible approval, device swap, correction, scroll fallback,
streamed clauses, cancellation and nested workers. Fails on skipped browser checks. Passed in 5.206 s.
Targeted core suite also passed (127 tests, 6.680 s).

Fixed race launch recovery: an unexecuted launch discarded by the stale-screen guard now re-triages
on the fresh screen, including approval checks; it no longer evaluates completion with a falsely
executed launch step. Deployed only race.py after matching previous hash and confirming terminal idle;
service restarted active. No timeout/gate reduction, routing changes, or task-specific shortcuts.

Read-only paired upload probe: parallel two calls 4.136 s; sequential two calls 4.133 s. Both successful.
This does not establish a concurrency bottleneck; keep existing speculation. Remote CSV records both.
Live Firefox `9d58ad8d`: DONE in 16.748 s agent execution / 22.417 s from transcribed submission,
including 5.678 s initialization. Input skill ~2.138 s; model 6.969 + 3.621 + 2.589 s. One completion
proposal was rechecked after a screen change. Result screenshot inspected: Firefox foreground new-tab
window is genuinely open. Evidence `work/demo-live-launch.log`, `work/demo-firefox-result.jpg`.
A warm `open calculator` request is now running via `/tmp/exospace-demo-request.py`; inspect live
session/output before taking another action. Goal is still active: one successful launch does not prove
fast/reliable demo readiness. Voice fixes are still local only. No commits/pushes.

Warm calculator run `fdbd65ab` FAILED at 46.466 s with zero input: first model call 10.154 s, stale launch discarded; fresh re-triage 35.213 s, also discarded after a screen change; stall guard ended run. New stale-launch path is exercised live but demo reliability remains unproven. User reports separate local-only capture changes (60 fps, two buffers, frame-count diagnostic); inspect before integration.

Capture integration (user forwarded other agent's local changes): measured candidate with 30 fps/one
buffer, 60 fps/two buffers, then baseline again: 14.9995, 59.9980, 14.9993 fps; fresh nonblank video in
all three 3-second windows. `work/demo-capture-live.log`; mean frame intervals recorded in local
bench/results.csv. Added doctor elapsed-time normalization and stalled-frame failure; corrected capture
thread test setup for new counter. Deployed capture/config/app/doctor after matching remote HEAD
hashes and confirming idle; no .env copied. Agent restarted active. Full suite 342 tests, one optional
skip, all others pass (`work/demo-integrated-tests.log`).

60 fps live calculator `485da52e` DONE: 16.602 s agent execution / 27.328 s including 10.736 s startup.
First launch discarded on screen change; second launch executed, result verified. Model durations
5.619 + 6.379 + 1.431 s, launch ~1.726 s. Screenshot saved `work/demo-calculator-result.jpg`.
Warm arithmetic request `calculate 12 times 9` is running, run `57c4f31f`, local log
`work/demo-live-arithmetic.log`, shell session 92199. Inspect current state before another request.

User suggested a separate race from quoted aggregate stats. Current JevOnlyPlanner already batches
completion+next step, and JevFlash supports 0–12 stages. Remaining candidate: direct race completion
followed by fresh planner calls; compare batching/reuse as an isolated benchmark before more live
parallel uploads. Quoted 47%/53% and 0.4 s estimates are not yet validated for this checkout/Pi.

Warm arithmetic `57c4f31f` DONE in 12.359 s submission-to-result / 12.353 s execution. Planner selected
one stage typing `12*9=` (not a hardcoded recipe), then Jev verified 108. First model step 8.540 s,
completion 2.654 s; input 0.092 s. `work/demo-live-arithmetic.log`; result capture downloaded to
`work/demo-arithmetic-result.jpg`. All live test processes from this turn are terminal; no pending HID
request. Calculator screenshot independently confirmed KCalc foreground. Goal remains active: fresh
successful launches and arithmetic show recovery, but latency is variable and no broad readiness claim
is justified. Next optimization should benchmark current direct verification-to-planner path, not
assume quoted aggregate call counts apply to this version.

### Demo continuation: voice + measured planner alternatives

Pi voice files contain newer server-side Speaker/audio/narration code absent from this checkout.
Do NOT overwrite remote layer.py. Read remote versions into `work/pi-voice-before.*`; merged existing
`m.voiced` guard and AudioContext resume into local HTML, then deployed HTML only with exact remote
hash check. Voice service restarted active; served HTML verified remotely preserves audio and has
ordered commands. Existing browser tabs need reload to pick it up. Seven browser regressions now
include no double speech for server-voiced lines. Compact gate: 18 tests, 6.005 s, all pass.

New read-only `bench/arch/post_action_study.py` compares completion then next-step query against the
same questions batched. Uses JevOnlyPlanner's exact query bundle, NO HID, no runtime change. Laptop:
separate 14 calls / median .310 s; batched 8 calls / .214 s. Pi: separate 7 calls / median 1.640 s;
batched 4 calls / 1.002 s. Both variants falsely held a plain search for approval. Initial scoring missed
this; corrected local JSON/CSV and remote JSON/CSV verdicts: laptop 6/8, Pi 3/4 for each. Raw stdout
still has original incorrect `correct:true` lines; use corrected JSON/CSV plus this note. Checks do not
prove next-action grounding. No batching deployed.

Live navigation `a7b551f6` FAILED after 61.307 s: launch, then wait, 17.694 s field grounding/type,
14.848 s Enter proposal discarded on screen change, 6.781 s fresh Enter also discarded, then stall.
`work/demo-live-navigation.log`. No Enter sent. Screenshot replay comparison (`work/demo-planner-replay.log`):
Flash 10.017 s proposes address-bar click + type + Enter in one batch; Jev-first 5.824 s proposes
page-field click + type only. Both remain proposals, no success claim from replay.

Testing existing JEV_FIRST=0 setting via newly created Pi user-service drop-in
`~/.config/systemd/user/exospace-agent.service.d/90-demo-planner.conf` (only Environment=JEV_FIRST=0).
Agent was confirmed terminal idle before restart. This keeps race triage, approval and verification
unchanged; no .env edit. Verify live behavior, compare navigation and arithmetic, then either retain
with evidence or remove only this drop-in and daemon-reload/restart when idle. Runtime override
currently remains installed. No live test with this variant submitted yet at this checkpoint.

User additionally proposed early partial-transcript preparation, pending-answer classification racing,
5-second Jev timeout/retry, unchanged-screen completion reuse, and stable JSON key order. Inspected
4e52713: it executes quick prefix commands using spoken_commands/quick.py and blind browser keys;
not safe eager preparation, conflicts with no hardcoded command rules. Do not cherry-pick it.
Current timeout audit on Pi: 3343 Jev calls, 3279 successful; 35 successes >5 s, 14 >10 s, max32.828 s.
Recent100: 87 successful, 26 >5 s, 14 >10 s; failures include8 WriteTimeout and2 ReadTimeout. Keeping
20-second timeout: 5 s would cut off ~30% of recent successes. No eager execution or cache change.
Safe preparation would require exact final-text match, fresh screen, correction/interruption invalidation,
and no HID before final. Same-pixel reuse also needs same goal/history/approval/device state.

JEV_FIRST=0 confirmed in running service environment. Live navigation `36ce8c23` reports DONE:
36.646 s execution /47.045 s submission including 10.408 s initialization. 3steps vs previous61.307s
failure: launch; one batch click/type/Enter; verify. Source log `work/demo-live-navigation-flash.log`.
Result screenshot downloading to `work/demo-navigation-flash-result.jpg`; independently inspect before
claiming goal reached (completion text ambiguously mentioned main page OR search page). Override is
still installed. Agent run terminal. Goal active; do not claim demo fully ready or fast yet.

### 15-minute final improvement window
User clarified demo scope: general browser navigation, calculator, desktop settings opening/selection;
near-real-time is the target (10–36 seconds is NOT sufficient). User then limited this pass to under
15 minutes. Window started 2026-10-18 13:07:05 UTC /09:07:05 local; STOP by13:22:05 UTC. Do not extend
work beyond that without new authorization. Finish with honest measured state, no near-real-time claim.

Live /talk cancellation checked: 35.3 ms to cancelled result, zero input (`work/demo-live-cancel.log`).
Found and reproduced token reset reviving an abandoned worker (`work/cancel-reset-before.log`). Fixed
CancelToken internally with per-request epochs; reset invalidates old worker checks and waits forever,
while new requests work. Public cancel/reset path unchanged.94 targeted tests passed. Deployed cancel.py
with hash check/idle check, service active. Compact gate now19 tests,5.766 s.

JPEG q90 vsq70 read-only grounding experiment (`bench/arch/jpeg_study.py`): both8/14 correct; six
refusals each, so coverage too weak for production change. Payload610700→444246 bytes total (~27%).
No image-quality setting changed. Raw study includes preliminary two desktop cases before fixture id
correction; final JSON is complete28-case run. No HID. CSV includes raw timed runs.

Startup fix: connection monitor may not be ready at the one startup check, so first goal paid21.472 s
for identification. app._next_command now offers idle callback; one attempt per connection prepares
profile while waiting, never delays an already queued command, and does not retry after cancellation
until a new request/connection.6 app tests pass;51 app/runtime/cancel tests passed before final guard.
Deployed app.py after exact hash+terminal-idle check, service active. Verify target identification now
happens without submitting a goal. Need final full suite once before finishing.

Settings request54567c2e FAILED: launch16.732 s model +2.31 s input; subsequent model46.94 s then
JevError. SSH observation disconnected; authoritative journal confirmed terminal failure. Current
snapshot `work/demo-settings-current.jpg` shows Codex foreground, not Settings; no settings success.

Read-only speculation benchmark running: session583? actual local exec handle recorded in next update;
`work/demo-speculation-study.log`, remote `/tmp/exospace-speculation-study.py`. Compares eager current
race vs deferred planner on launch and arithmetic screenshots; no HID. Deferred is an isolated fake
pool in the benchmark, not a production modification. JEV_FIRST=0 drop-in remains in place.

### Timeboxed verification, 2026-10-18 09:12 local
- Verified idle preparation twice from the Pi journal: target identified at09:08:35 and09:12:24
  without submitting a goal. Latest test submitted at09:12:29 emitted its run event after0.010 s,
  removing identification from that request's startup path. This is not a total-latency claim.
- Reproduced another cancellation regression: discarded speculative future raising Cancelled
  aborted an uncancelled current request. `_settle_speculation` now checks current cancellation
  rather than propagating the obsolete future's cancellation. Tests cover both superseded work
  and real current interruption. Deployed after idle/hash checks; active service, race.py SHA256
  31f5a5c5966bfc23d8d58f51bcccc08846ed77846f53b298bd83c1b1b990f336.
- Final full suite:348 tests,347 passed,1 optional mss skip,24.676 s
  (`work/timebox-final-tests.log`). Compact demo gate:21 passed,5.739 s
  (`work/timebox-demo-smoke.log`). `git diff --check` clean.
- Read-only eager/deferred speculation study failed with Jev WriteTimeout before producing a
  usable comparison (`work/demo-speculation-study.log`). No race-default change justified;
  no image-quality or timeout reduction deployed. Pi still uses JEV_FIRST=0 service override.
- Live combined calculator check running:3c5b1952, "open calculator and calculate 18 times 7",
  `work/timebox-live-calculator.log`; remote capture work/live-latency/20261018-091229.
  Must record terminal outcome before finishing. Existing remote voice Python remains preserved.

Combined calculator run3c5b1952 succeeded:24.604 s submission-to-result,24.595 s benchmark total.
Screenshot `work/timebox-calculator-result.jpg` inspected: KCalc visibly shows126. Model steps
7.220+7.074+6.624 s; actions1.937+0.085 s. Raw Pi rows copied into bench/results.csv. Identification
was already finished before submission; run event0.010 s. No claim that this is near-real-time.
Final desktop test82f44d13 now running: "open system settings and select appearance";
`work/timebox-live-settings.log`, remote work/live-latency/20261018-091303.

Final settings run82f44d13 FAILED after 34.985 s (model step34.491 s), with no action event in the
observer log. Do not present Settings/Appearance as verified. Both live requests are terminal;
no test intentionally left running. README and architecture notes now distinguish synthetic timings
from the live rig, document partial-transcript planning as unimplemented, and preserve the newer remote
voice layer. No UART change, timeout reduction, quality reduction, commit or push in this pass.

### Claude pass after Astra — 2026-10-18 (DEPLOYED 09:4x with the follow-up below)
Causes found in the Pi journal: settings run `54567c2e` had opened System Settings, then the completion
check's Jev upload hit `WriteTimeout` twice (~20 s each) and `JevError` ended the request. Startup still
reported `windows` for the Linux laptop on every restart this morning.
- `brain/agent/prompts.py` `OS_PROMPT`: name an OS only from visible evidence (start menu/launcher/dock,
  taskbar or menu-bar logo, phone bars); otherwise `unknown`, which triggers the existing OS-key probe in
  `brain/device.py`. Offline check with Gemini on saved frames (2 tries each): the old prompt said `windows`
  for 6 of 8 Linux-laptop frames; the new one says `unknown` for all 8. Unchanged: synthetic Windows desktop
  `windows`, phone frames `android`, BIOS and Blender `unknown`. **Not verified:** that the probe frame
  (KDE launcher open) classifies as `linux`. If startup still logs `unknown`/`windows`, pin with `OS=linux`.
- `brain/agent/jev.py`: connect and upload capped at 8 s (`UPLOAD_S`); waiting for the answer keeps
  `JEV_TIMEOUT_S` (20 s). A stalled upload now costs ~8 s before Jev's retry instead of 20 s.
- `brain/agent/loop.py`: a `JevError` from planning gets one more look per request instead of ending it;
  a second one fails as before. The stall limit still bounds the run. Two tests in `tests/test_runtime.py`.
- Full suite 350 tests OK (1 optional skip); `python -m bench.demo_smoke` OK; `git diff --check` clean.
To deploy (after checking the Pi copies differ from these only by the changes above, and that the agent is
idle): rsync the three files to `~/Exospace/brain/agent/` and restart `exospace-agent`.

**Follow-up, deployed** (Pi copies diffed first: only these edits differed; agent idle; restarted, active):
- `brain/agent/race.py`: the planner no longer starts speculatively alongside triage; it runs only when triage
  routes to "plan" or a direct action/launch needs more. On the Pi both uploads shared one slow link, and after
  a launch the next plan *waited for the discarded speculative Flash plan to finish*. Removed the worker pool and
  `_settle_speculation` (and its two tests); added tests that a direct click starts no plan and a "plan" route
  plans once. Unmeasured live: expected to cut step 1 and post-launch waits; measure before claiming numbers.
- `brain/device.py`: the OS probe waits up to 3 s for the launcher to appear (thumbnail change >= 1.5), then
  0.5 s, instead of a fixed 1.2 s. The Pi's 09:25 startup probed and still got `unknown` (launcher off; run
  `05d685c9` failed in 33 s).
- **OS pinned for the desktop demo:** `~/.config/systemd/user/exospace-agent.service.d/95-demo-os.conf` adds
  `--os linux`; startup now logs `Target: linux`. **Remove that file (daemon-reload, restart when idle) before
  using the phone**, or it will be driven as a Linux desktop. `90-demo-planner.conf` (`JEV_FIRST=0`) unchanged.
- Full suite 350 OK (1 optional skip). No live request was submitted after deploying.

**IPv4 for hosted APIs — deployed** (agent + voice restarted when idle):
Run `bcb5c376` ("open calculator", DONE in 10.9 s) spent 6.8 s on its first Jev call; the same call on an open
connection took 0.34 s. Traced on the Pi: httpx's `connect_tcp` took 6.2 s. The APIs are behind Cloudflare,
which lists IPv6 first; the hotspot gives the Pi IPv6 addresses with no working route, and Python tries both
IPv6 addresses before IPv4 (curl races them: 0.7 s for the same request). New `brain/net.py`: `ipv4_transport()`
for httpx (Jev in `jev.py`, Gemini via `HttpOptions.client_args` in `model.py`) and `IPV4` for websockets.
Measured on the Pi after the fix: a Jev call on a new client 0.37 s (was 6.5 s); a cold Flash call 3.2 s.
The Pi's `brain/voice/layer.py` was patched in place (newer remote code preserved; backup `/tmp/layer.py.before`):
Deepgram STT `connect(..., family=IPV4)` and `Speaker(key, transport=ipv4_transport())`. The local
`layer.py` has only the STT change (it has no Speaker). Describe mode's Live connection is unchanged.
Earlier, at 09:3x, the hotspot's upload also fell to ~13 KB/s (300 KB in 22 s) and recovered to ~200 KB/s:
when runs stall with WriteTimeout, check the link before the code.

**OS pin removed (09:50)** — `95-demo-os.conf` drove the phone as a Linux desktop. Detection is automatic again
(startup logged `android (touch)` for the phone). The laptop relies on the evidence-only OS prompt plus the
launcher-waiting probe; if it reports `unknown`/`windows`, re-add the pin only while demoing the laptop.

**Local stale-screen check — on the Pi, takes effect at the next agent restart** (not restarted: user was mid-test).
Profile of `000d347f` ("Open Firefox and go to YouTube and search up horse videos"): 46 s, 9 steps, 7 Flash
calls 23.8 s, 23 Jev calls 6.5 s; steps 2, 4, 5, 7, 8 executed nothing (24.5 s of plans), the pattern of the
whole-screen stale check discarding plans while YouTube loaded and a chat window streamed on the target.
`brain/agent/loop.py` `_stale`: an aimed action is stale only if the area within `AIM_SPAN` (0.08 of the screen)
of its point changed (`capture.area_change`, compared on a 24x16 grid); a `done` claim still uses the whole
screen; keys, typing and launches are not discarded. The mid-batch check uses the same local area. Two new
runtime tests; 352 tests OK; demo gate OK. **Not yet measured live**, including capture noise on the small
phone area (the display endpoint only serves the last step's frame); if plans on the phone are discarded
more often after restart, raise `AREA_THUMB_SIZE` averaging or the threshold for areas.

**Stale check active; Android launcher fix — deployed, agent restarted (10:1x).**
- Capture noise on a still phone screen measured with the agent stopped: worst local `area_change` 0.17, whole 0.00
  (threshold 1.5), so the local stale check is not tripped by noise.
- Phone "Open YouTube" (`23f0d5d9`) took 85 s / 17 steps. Frames sampled every 0.25 s showed why: in Samsung's
  launcher search, **Enter does nothing**; the results list stays (the 0.52 s "launch" was the keyboard closing).
  Tab moves focus: 1 Tab = first app icon (here YT Studio), 2 = YouTube, 3 = the Settings section, 5-6 = app info.
  Per the user, `open_app` now presses Tab once (first result) then Enter on Android (`FIRST_RESULT_KEYS` in
  `brain/skills/__init__.py`). **Known limit:** the first result can be a similar app (YT Studio ranked above
  YouTube); the launch check then hands the request to the planner. A Jev-checked Tab walk would fix that.
  Not yet verified live after deploying. App shortcuts (`APP_KEYS`: settings, messages, browser, ...) are unchanged.
- 352 tests OK. Pi drop-ins now: only `90-demo-planner.conf` (`JEV_FIRST=0`).

### Timing review from pulled logs (Pi disconnected at 10:04; logs in `work/pi-logs-1004/`)
33 requests 09:45-10:04 (after the IPv4 fix): model time 267 s, of which **Flash 199 s (63 calls, ~3.2 s each)**
and Jev 60 s (251 calls, 0.24 s each); settle 75 s; input 45 s. Requests Jev finished alone took 1-3 s
(settings 2.7 s); every slow one was slow from repeated Flash planning: "Open YouTube" on the phone 85 s / 14 Flash
calls (launcher bug, fixed), "search up lo fi beats" `b0a3a482` 46 s / 7 Flash calls, 4 of 8 steps sent no input
(one was a plan rejected twice: Flash's `open_app` had no name, ~7 s; 2 such rejections in the window).
- Flash `thinking_level="minimal"` is not accepted by gemini-3.8-flash (ClientError), so `low` is the floor:
  laptop benchmark of the real planning prompt, 4 saved screens x 3: medians 2.9 / 3.3 / 5.4 / 4.6 s.
- Same 4 screens through `JevOnlyPlanner` (laptop, 3 runs): medians 1.0 / 0.8 / 0.4 / 0.3 s. Proposals: Firefox
  new tab -> clicks the page's search box (a web search, not the YouTube URL Flash chose); KCalc "12 times 9" ->
  can't, hands off to Flash (+0.8 s); settings -> `open_app` "system settings"; inbox -> done (same as Flash).
- **Next experiment (needs the rig):** remove `90-demo-planner.conf` (back to `JEV_FIRST=1`) and run the demo
  paths. Astra chose Flash plans when Jev calls took 1-7 s on the Pi; they now take ~0.25 s. Expect ~2.5-5 s saved per
  step Jev can take; watch navigation (Jev's web-search path) and keep Flash if those regress.
- Local only, not deployed: `loop.py` now logs `Step N: discarded, the screen changed where it acts`, so steps
  that send nothing can be told apart from rejected or held plans. 352 tests OK; demo gate OK.
- **Switched to Jev-first on the Pi (user request, untested):** removed `90-demo-planner.conf`, so `JEV_FIRST`
  is back to its default 1; deployed the `loop.py` discard log line (Pi copy differed only by it). Agent restarted
  when idle, active. No service drop-ins remain. To go back to Flash plans, re-create that drop-in with
  `Environment=JEV_FIRST=0`, daemon-reload and restart when idle.
