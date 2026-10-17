# Exospace: Devpost draft

> Software runs in user space or kernel space. Exospace runs outside the machine entirely.

## Inspiration
People with limited hand mobility get by with head mice, mouth joysticks and switch
interfaces. Those replace the mouse, but the person still has to perform every click.
Built-in voice control exists, but it is different on every OS, needs setup on each device
(often admin rights), and works one command at a time. We wanted something you carry with
you: plug it into any computer or phone, say what you want, and it does the steps.

## What it does
Exospace is a small box that plugs into a target over one USB-C hub. It **sees** the screen
through HDMI capture and **acts** as a USB keyboard, mouse and touchscreen. Nothing is
installed on the target, so it works on locked-down library computers, a friend's laptop
or a phone. You press one button (or an accessibility switch) and talk.

- Direct requests ("open WhatsApp", "go back") happen in about a second.
- Multi-step requests ("reply to John saying I'll have it by five") are planned and checked
  step by step on the real screen.
- Remarks ("ugh, this folder is a mess") are remembered, not acted on.
- Anything irreversible (send, delete, pay, BIOS changes) waits for a spoken "yes".
- It remembers you as you unplug from one device and plug into the next.

## How we built it
- **Hardware:** Raspberry Pi 5 (brain, on its own power), Raspberry Pi Pico running
  CircuitPython as a composite USB HID device (boot keyboard, absolute mouse, multitouch),
  an HDMI capture card and a USB-C hub. The Pi talks to the Pico over UART with a small
  JSON protocol; the Pico is the only thing that touches the target.
- **The race architecture:** every request goes to **Jev 1.14** (TypeSafe's vision model,
  ~0.3 s) and **Gemini 3.8 Flash** at the same time. In one call, Jev decides "act now",
  "plan" or "no action", locates the target, picks click, double-click or right-click,
  and flags irreversible requests. A confident Jev action wins and Flash's plan is
  discarded; otherwise Flash's plan, already running, is executed stage by stage with Jev
  checking each stage on a fresh screenshot.
- **Knowing when it's done:** one Jev call judges the goal against the list of steps
  actually executed plus the screen. That catches skipped steps and searches that were
  typed but never submitted, and still accepts history-dependent goals like "the first video".
- **One writer:** a single controller is the only path to the keyboard and mouse, with one
  cancel path for unplugging, barge-in and failed checks.
- **Voice layer:** Deepgram transcribes server-side, each pause becomes a request, talking
  while it works interrupts it, and the agent's questions and results are spoken back.

## Challenges we ran into
- **Measuring instead of guessing.** We built screenshot-only benchmark fixtures with
  independent state checks and ran over a thousand model calls. Several intuitions were
  wrong: a cheaper or faster model was usually less safe, and the biggest speed win came
  from fixing a completion check that rejected finished work, not from a faster model.
- **A safety bug in our own first design:** told to "delete the downloads folder", it did,
  in 3 of 3 runs, despite a prompt saying to ask first. A dedicated Jev check caught 27/27
  destructive requests with 0/33 false alarms.

## Accomplishments we're proud of
- 89% task success vs 53% for an LLM-planner-only agent, with 0 vs 5 unsafe actions,
  across two interactive scenarios including a held-out one (paired sign test p = 0.00002).
- First click in about 1 s for direct requests, with no LLM call on that path.
- No hard-coded command rules: the same pipeline handles any screen.

## What we learned
Vision models that answer narrow, structured questions (Jev) make better gatekeepers than
general LLMs: faster, and more conservative about acting. The LLM is best kept as the
planner for the cases that truly need one.

## What's next
Hands-free continuous listening on top of the commit gate we already built (pipelined
speech with in-order execution); per-user preferences; more on-device testing with real
assistive-tech users.

## Honest limits
Benchmarks ran on synthetic screens in a browser; hardware tests are in progress. Models are
hosted APIs, so it needs a network connection.
