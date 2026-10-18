# Exospace status and handoff

**`main` is the source of truth.** Every branch, including `overnight-ultracode`, is merged into it.
Earlier session logs and research notes live in git history; current evidence is in
[bench/arch/README.md](bench/arch/README.md).

## What works (2026-10-17)
- **Runtime:** race executor is the default with either a Gemini or a z.ai key (PLAN.md §10).
  236 offline tests pass (`.venv/bin/python -m unittest discover -s tests`, no API calls); `test_capture` skips
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
- **Laptop through the real Pico** (KDE, 2880x1800, spectacle capture): main suite 11/12 (211 s); judge
  stress suite 9/9 in 40 s (barge-in, gibberish, "delete my home folder", passwords, contradictions, a hidden
  page injection, Spanish, a long query, a question); Blender from closed: open, maximize, add a sphere, 34 s.
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

These runs used per-OS prompt hints and the emulator backend, both since removed (PLAN.md §10), so they are
history, not a measure of `main`.

## Fixed after the website runs
- Scroll amounts are wheel notches (the planner had sent pixel counts; on the Pico that flings to the
  page end) and the controller caps one scroll at 15.
- Starring, liking, following, subscribing, voting and sharing count as irreversible and ask first.
- Jev and Gemini retry once on a timeout or dropped connection (z.ai already did).
- A second unchanged screen in a row counts as a failed step on the planner path, so a stuck task that keeps
  trying *different* inputs stops early; the same batch from the same screen three times also stops a run.

## Known issues (not fixed)
- **False "done":** the completion check accepted a merely visible link as success (2/2), and reported
  "Done" after a click redirected to GitHub's login page. Still seen: "open the tutorial, then the page
  about data structures" stopped with the section only scrolled into view.
- **"Go back" is hard to confirm:** after a correct back navigation the completion score stayed low, so the
  planner went back again (2/2).
- **Click precision (top priority).** Jev answers boxes in 0.005 steps (~14 px on a 2880 px screen) and its
  full-screen boxes miss by 10-25 px at 1280 scale; one zoom lands 12/15 planner clicks inside the target,
  but controls a few pixels apart need a second tight zoom (Blender's maximize vs minimize: 0/3 → 4/4).
  After the black-wallpaper crop fix the agent sees the true full screen, so small windows mean small
  targets: wiki_link went 5 s → 25 s and close_calc took 5 tries. To measure on an offline target set,
  then keep what helps: an adaptive second zoom for small boxes, the median of 2-3 parallel looks, and a
  hover check before clicking a small control next to a destructive one.
- **Small windows are maximized only on the planner path.** Jev's direct actions skip the window check.
- **Deep mode is slow and shallow.** Argon low takes 4-24 s a step; "make me a house" built a cube with a
  pitched roof (153 s).
- **Planner slips:** once typed `xdg-open <url>` into a browser while recovering a link click.
- **Untested:** the capture card, the real Pico, a Mac, a real phone, iOS and ChromeOS.

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
- Wire `brain/arbiter.py` for pipelined voice segments (the voice layer interrupts and resubmits today).
- The push-to-talk GPIO button and 3.5 mm switch jack are wired in the bring-up guide but not read by
  software yet; the voice page's Talk button and the Space bar work.
