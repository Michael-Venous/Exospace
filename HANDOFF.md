# Exospace status and handoff

**`main` is the source of truth.** Every feature branch is merged except `overnight-ultracode` (see below).
Earlier session logs and research notes live in git history; current evidence is in
[bench/arch/README.md](bench/arch/README.md).

## What works (2026-10-17)
- **Runtime:** race executor is the default (PLAN.md §10). 200 offline tests pass
  (`.venv/bin/python -m unittest discover -s tests`, no API calls).
- **Voice layer:** `brain/voice/layer.py` with the Deepgram relay (key stays server-side), smoke-tested
  through the real agent.
- **Protocol:** `tests/test_pico_loopback.py` runs the real `PicoHID` + `Controller` against the real
  firmware command handler over a pty.
- **Live checks on the browser backend:**
  - Direct click in 1.5 s; a remark gets "Noted." in 0.6 s with no input.
  - A destructive request asks, and "no" declines.
  - "No, open John's email instead" runs the new request.
  - YouTube → search → first video.
- **Public websites** (Wikipedia, GitHub, docs.python.org, OpenStreetMap): 16 of 23 runs passed before the
  fixes below; safety held on remarks, ambiguous requests and a donation.

## Fixed after the website runs
- Scroll amounts are wheel notches (the planner had sent pixel counts; on the Pico that flings to the
  page end) and the controller caps one scroll at 15.
- Starring, liking, following, subscribing, voting and sharing count as irreversible and ask first.
- Jev and Gemini retry once on a timeout or dropped connection.
- On the planner path, a second unchanged screen in a row counts as a failed step, so a stuck task stops early.

## Known issues (not fixed)
- **False "done":** the completion check accepted a merely visible link as success (2/2), and reported
  "Done" after a click redirected to GitHub's login page.
- **"Go back" is hard to confirm:** after a correct back navigation the completion score stayed low, so the
  planner went back again (2/2).
- **Planner-path clicks are less precise** than the direct path: 8/15 inside the target without zoom,
  12/15 with it. The planner path does not zoom yet.
- **Untested:** the complex web-desktop run crashed before reporting; nothing has run on the capture card,
  the real Pico, a Mac or the phone yet.

## Waiting on hardware (in order)
1. On the Pi with sudo: the steps `scripts/pi/setup.sh` prints (uart0, `video`/`dialout` groups, linger),
   reboot, copy `.env` (chmod 600).
2. Flash all eight `firmware/pico/*.py` files; plug in the capture card and Pico; run
   `.venv/bin/python -m brain.doctor`.
3. `systemctl --user enable --now exospace-agent exospace-voice`; tap near each corner of the phone
   ([firmware/pico/README.md](firmware/pico/README.md)); run [docs/demo.md](docs/demo.md) on real targets and
   re-measure latency with the capture card.

## `overnight-ultracode` (do not merge)
Built unattended on an old `main`; merging it would delete the voice layer, Pi scripts and firmware fixes.
Already ported: `EXECUTOR=direct` (z.ai GLM), OS hints, on-screen check of `done` claims, the mid-batch
guard, blank-screen resume, the `wait` skill, retries. Left there on purpose:
- Firmware setup (arrow-key menus, `boot_keys` taps through a restart, BIOS re-detection): out of scope
  (PLAN.md §3) and 4/21 in QEMU only.
- QEMU and Android-emulator backends: test harnesses, not the demo path.
- `quick.py` regex fast path: a hard-coded phrase list (PLAN.md §3); race triage covers it.
- `TARGET_OS`: pins the OS, which breaks swapping devices; `--os` exists for a manual override.
- chromeos/ios OS names: untested.

## Still to do
- Race vs direct in `bench/arch/e2e_study.py`, with an injection segment.
- `bench/guard_cost.py` and the full test suite on the Pi.
- Wire `brain/arbiter.py` for pipelined voice segments (the voice layer sends one segment at a time today).
- The push-to-talk GPIO button and 3.5 mm switch jack are wired in the bring-up guide but not read by
  software yet; the voice page's Talk button and Space bar work.
