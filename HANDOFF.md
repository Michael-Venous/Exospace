# Exospace status and handoff

**`main` is the source of truth.** Every branch, including `overnight-ultracode`, is merged into it.
Earlier session logs and research notes live in git history; current evidence is in
[bench/arch/README.md](bench/arch/README.md).

## What works (2026-10-17)
- **Runtime:** race executor is the default with either a Gemini or a z.ai key (PLAN.md §10).
  233 offline tests pass (`.venv/bin/python -m unittest discover -s tests`, no API calls); `test_capture` skips
  without `mss`, which only the `local` extra installs.
- **Voice layer:** `brain/voice/layer.py` with the Deepgram relay (key stays server-side), smoke-tested
  through the real agent; the newest request wins, and push to talk works.
- **Protocol:** `tests/test_pico_loopback.py` runs the real `PicoHID` + `Controller` against the real
  firmware command handler over a pty.
- **Live checks on the browser backend (Gemini):**
  - Direct click in 1.5 s; a remark gets "Noted." in 0.6 s with no input.
  - A destructive request asks, and "no" declines.
  - "No, open John's email instead" runs the new request.
  - YouTube → search → first video.
- **Public websites** (Wikipedia, GitHub, docs.python.org, OpenStreetMap): 16 of 23 runs passed before the
  fixes below; safety held on remarks, ambiguous requests and a donation.
- **Race on z.ai alone** (workstation study, one repeat): 7/10 by the fixture's oracle, 0 unsafe. Jev clicks
  and remarks take about 1 s with no GLM call; "go to youtube" ran out of steps after 8 GLM calls.
- **From `overnight-ultracode`** (live GLM, `direct`, each oracle checked outside the agent):

| Goal | Target | Result |
|---|---|---|
| Text editor: type a line, save `~/hello.txt` | nested XFCE desktop | PASS, 36.5 s |
| 1234*5678 in the calculator | nested XFCE desktop | PASS, 24.2 s |
| Turn on dark theme | Android emulator | 2 of 3 |
| Set an alarm for 6:45 AM | Android emulator | 1 of 3 |
| Add contact Ada Lovelace, 555 123 4567 | Android emulator | 0 of 9 |
| Restart into firmware setup, set boot time-out to 7 s | QEMU/OVMF | 4 of 21 |

## Fixed after the website runs
- Scroll amounts are wheel notches (the planner had sent pixel counts; on the Pico that flings to the
  page end) and the controller caps one scroll at 15.
- Starring, liking, following, subscribing, voting and sharing count as irreversible and ask first.
- Jev and Gemini retry once on a timeout or dropped connection (z.ai already did).
- A second unchanged screen in a row counts as a failed step on the planner path, so a stuck task that keeps
  trying *different* inputs stops early; the same batch from the same screen three times also stops a run.

## Known issues (not fixed)
- **False "done":** the completion check accepted a merely visible link as success (2/2), and reported
  "Done" after a click redirected to GitHub's login page.
- **"Go back" is hard to confirm:** after a correct back navigation the completion score stayed low, so the
  planner went back again (2/2).
- **Planner-path clicks are less precise** than the direct path: 8/15 inside the target without zoom,
  12/15 with it. The planner path does not zoom yet.
- **Race planner and `wait`:** the planner prompt offers `skill(name=wait)` and `boot_keys`, but its schema still
  removes `wait` from the allowed names, so race cannot actually ask for it.
- **Firmware is experimental:** the model often claims `done` before saving with F10 then Y. PCs with Fast Boot
  ignore USB keyboards during POST, a restart that never resets USB gets no key taps, and Mac boot keys must be
  held, which taps cannot do.
- **Thin re-detection margins:** the threshold (40) is close to the smallest firmware-to-desktop change
  measured (44.1); one firmware-to-phone pair changed by only 1.9, so a phone misread as firmware can stay `bios`.
- **Quick start uses a word list,** so "open a text editor" still goes to the model.
- **Untested:** the capture card, the real Pico, a Mac, a real phone (only the emulator), iOS and ChromeOS.
  The OS hints were tuned on Linux and Android only.

## Waiting on hardware (in order)
1. On the Pi with sudo: the steps `scripts/pi/setup.sh` prints (uart0, `video`/`dialout` groups, linger),
   reboot, copy `.env` (chmod 600).
2. Flash all eight `firmware/pico/*.py` files; plug in the capture card and Pico; run
   `.venv/bin/python -m brain.doctor`.
3. `systemctl --user enable --now exospace-agent exospace-voice`; tap near each corner of the phone
   ([firmware/pico/README.md](firmware/pico/README.md)); run [docs/demo.md](docs/demo.md) on real targets and
   re-measure latency with the capture card.

## Still to do
- Race vs direct in `bench/arch/e2e_study.py`, with an injection segment.
- `bench/guard_cost.py` and the full test suite on the Pi.
- Team sign-off on the contract notes added for firmware and new OS names (PLAN.md §5.3, §5.4, §5.6).
- Wire `brain/arbiter.py` for pipelined voice segments (the voice layer interrupts and resubmits today).
- The push-to-talk GPIO button and 3.5 mm switch jack are wired in the bring-up guide but not read by
  software yet; the voice page's Talk button and the Space bar work.
