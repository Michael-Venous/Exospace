# Agent

**Owner:** Agent

The executors, the agent loop and memory. How a request runs and what each module does:
[brain/README.md](../README.md). Why it is built this way: PLAN.md §10 and [bench/arch/README.md](../../bench/arch/README.md).

## Interaction rules
- **Voice first.** The user may not be able to type; the on-screen log is for helpers, judges and debugging.
- **The button always means "listen to me."** Idle, it starts a request; while working, it is a barge-in through
  the one cancel path (PLAN.md §5.6), and what follows is a correction of the same task.
- **Speak little.** One line when work starts and the result, never per keystroke; speech never delays input.
- **Ask, don't guess.** Ambiguous requests get a question; irreversible ones need a "yes" bound to that request
  and device; a remark is remembered, not acted on.
- **Point and speak.** On the touch display, "Tap = this" says *which* object; the planner picks *how* for this
  app, and Jev grounds and checks every stage.

## Ideas not built (measure each before adopting)
1. **Zoom on the planner path.** The direct path's zoom refinement lifted planner clicks from 8/15 to 12/15 inside
   the target on real websites.
2. **Privacy mode.** Black out password fields and card/SSN patterns (local OCR during settle) before screenshots
   leave the Pi; ask the user instead of typing a password.
3. **Screen-fingerprint cache.** Settled-frame hash + stage → the action that worked; a hit skips both models.
4. **Type the user's own words.** The model points at which transcript words are the value and code copies them,
   so a spoken email address is never retyped by a model.
5. **macOS Voice Control overlay** (*Item Numbers*) puts a number on every clickable item, visible over HDMI with
   nothing installed; it could help Jev on dense Mac screens.
6. **Wake word** ("Hey Exo", openWakeWord on the Pi) for users who cannot press anything.
