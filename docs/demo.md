# Demo script (about 3 minutes)

Judges will try to break it. The script leans into that: every "attack" below is something
we measured and handle.

**Setup:** Exospace box (Pi 5 + Pico + capture card) on its own power. Laptop A (Windows),
laptop B (Mac) and the Samsung phone, all mirrored, all pre-plugged once (first-plug driver
popups done). Projector shows the touch display page (`--display 8770`): live screen,
crosshair on each click, plain-language history. Voice page (`:8780`) on a tablet or the
display, big Talk button. Phone hotspot as backup network.

## 1. "It just works" (30 s)
Plug into laptop A. Press Talk:
- *"Open up Chrome."* Desktop icon, so Jev double-clicks it. About a second, no LLM call.
- *"Go to my email … open Janet's email about the report."*

Point out: nothing is installed on the laptop; Exospace is a keyboard and mouse.

## 2. It knows a remark from a command (20 s)
- *"Oh, John is asking for a report."* It says "Noted." and touches nothing.

The previous design clicked John's email here. Jev's triage has a "no action" route.

## 3. Interrupt and correct (20 s)
- *"Reply saying I'll have it by five…"* and while it's working, press Talk:
  *"Actually make it six."* It stops instantly and continues with the correction.

## 4. Try to break it: destructive requests (30 s)
Invite a judge to say anything destructive:
- *"Delete all of these emails."* It replies: "This can't be undone. Should I go ahead?"
- *"No."* It replies: "Okay, I won't do that."

Measured: 27/27 destructive requests caught (send, delete, pay, factory reset, BIOS
changes), 0/33 false alarms. The old design deleted a folder in 3/3 runs even with a
prompt telling it to ask first.

## 5. Swap devices (40 s)
Unplug, plug into the phone. It re-detects the device as Android with touch input, and
memory carries over.
- *"Open WhatsApp."* A tap, under a second.
- *"Text mom that I'm running late."* The planner path, each step checked by Jev.

## 6. The numbers (20 s, one slide)
- 89% task success vs 53% for an LLM-planner-only agent; 0 vs 5 unsafe actions
  (paired test, p = 0.00002, held-out scenario included).
- Direct commands: first click in about 1 s with no LLM call; remarks 0.7 s.
- We tested GPT-6 Luna, Claude Haiku 5.5 and others against Gemini 3.8 Flash. The
  faster ones were unsafe on messy speech, so Flash stays the planner.

## If something fails on stage
- **Hardware won't come up:** run the same agent against the isolated browser
  (`--backend browser`) with the display page; the voice, safety and speed beats all work.
- **Network is slow:** use the phone hotspot. Direct commands still need one Jev call (~0.3 s).
- **It asks a question:** that's the design. Answer it out loud.
