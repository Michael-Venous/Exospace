# Agent design: how everything works together

**Owner:** Agent. **Status:** proposal. Section 8 lists what needs team approval before it touches PLAN.md §5.

This covers the parts around the model loop that aren't built yet: who it's for, what the user hears and sees, how
speech gets in, and how voice and the projector view plug into the agent. The model orchestration itself is built
(Jev 1.14 + Gemini 3.8 Flash, `brain/agent/jev_flash.py`); section 6 summarizes it and lists ideas to try on top.

## 1. Who it's for
**Lead with limited hand mobility.** The box replaces the *hands*, so it fits people who can see the screen and
speak but can't use a keyboard and mouse well:
- spinal cord injury, ALS (while speech holds up), MS, muscular dystrophy, cerebral palsy
- tremor or Parkinson's, arthritis, amputation, severe RSI or carpal tunnel

Two things make this a strong fit:
1. **Nothing to install.** It works on a library computer, a work laptop with no admin rights, or a family member's
   phone. Built-in voice control only works on your own, already set-up device.
2. **One input.** Everything starts from a single button, and assistive switch users already have a button they
   can press (section 3).

**Second: low vision and blind users.** "Read me what's on screen," full read-back of results (verbosity on), and
large text on the box's touchscreen. It's second, not first, because operating systems ship screen readers and our
demo is visual. Our edge is the same: no setup on a device that isn't yours.

**Not targeted:** deaf and hard-of-hearing users can use a keyboard and mouse fine. Still, every spoken line also
shows up in the on-screen log, so nothing is audio-only.
**Be honest about the limit:** speech-to-text struggles with dysarthria (slurred speech), so people who can't speak
clearly aren't served yet. Wording from PLAN.md §11: "designed for limited hand mobility," never "validated."

## 2. What the user experiences
**Voice first, no chat window for the user.** The person we're serving can't comfortably type, so typing to the agent
defeats the purpose. There *is* a conversation log on screen (section 5), but it's for sighted helpers, judges and
debugging.

```
 idle ──hold button──► LISTENING ──release──► THINKING ──► ACTING ──► DONE
   ▲                     (rising tone)        "Okay, opening    step-by-step   (chime + one-line
   │                                           Calendar"        on screen       result read back)
   │                                                │              │
   │                       press button at any time ┴──────────────┴──► CANCEL (§5.6) ──► LISTENING
   │                                                                     ("Actually make it 7:30")
   └──────────── agent needs something: speaks the question, mic opens ◄─┘
                 by itself for 6 s (no button), "yes" for risky steps
```

Rules:
- **The button always means "listen to me."** In idle it starts a request. While the agent is working it's a
  barge-in, which uses the one cancel path (§5.6), then listens. A correction goes to the planner along with
  memory, so it continues the task instead of starting over.
- **Questions don't need the button.** When the executor returns `ask_user`, or a risky action needs confirming,
  the agent speaks and opens the mic for about 6 seconds. Making a motor-impaired user press a button to answer a
  question we just asked is bad design.
- **Sounds before speech.** Short sounds for listening, working, done, error and "I need you" are faster than words.
  Speech is kept for content.
- **Speak little.** One short line when each subtask starts ("Opening Calendar…"), not one per keystroke. Never wait
  for text-to-speech to finish before acting. A verbosity setting turns on full read-back for low-vision users.

## 3. Inputs (Hardware + Demo build these; Agent only consumes events)
| Input | Why | Priority |
|---|---|---|
| Big push-to-talk button on GPIO | The main control | Must |
| **3.5 mm switch jack wired to the same GPIO** | Standard assistive switches (buddy buttons, head switches, sip-and-puff interfaces) use a 3.5 mm mono plug. Any user's existing switch becomes our button. A jack and two wires. | Should, great pitch line |
| Typed-goal box on the projector page | Development without talking, plus a backup in a loud expo hall | Should |
| Wake word ("Hey Exo", openWakeWord on the Pi) | For users who can't press anything | Stretch |

## 4. Speech in and out
**Push-to-talk makes speech-to-text easy:** releasing the button marks the end of the sentence, so we don't need
silence detection (which waits for a pause) or an "end of turn" model.

| Option | Latency after release | Notes |
|---|---|---|
| **Deepgram streaming** (recommended first) | About instant, because text arrives while the user is still talking | Cloud, API key, free credit, about 20 lines of code |
| Whisper large-v3-turbo on the B200 | Network round trip plus a fraction of a second | Fits "models run on our own GPU server"; adds work for Models |
| whisper.cpp `base.en` on the Pi | Roughly 1–2 s for a short clip (measure it) | Offline backup when Wi-Fi dies |

**Text-to-speech:** Piper on the Pi (local, fast, works offline). The sounds are WAV files, so no model is involved.

## 5. What shows on screen (Demo owns the view; Agent publishes events)
One small page, served by the Pi, shown on the projector and on the box's touchscreen:
- the live captured frame with a **crosshair on the next target**
- a **conversation log** (what the user said / what the agent said)
- the current plan with subtasks ticked off, plus the step log with per-step timings
- the state (listening / thinking / acting / waiting for you) and a big task timer

The agent publishes newline-delimited JSON events over a local WebSocket. The page only displays them, so the view
can't break the agent.
```json
{"ev":"state","state":"listening|thinking|acting|asking|done|cancelled"}
{"ev":"user_said","text":"draft an invite to my team"}
{"ev":"agent_said","text":"Opening Mail…"}
{"ev":"plan","subtasks":["open Mail","new message","fill fields"],"done":0}
{"ev":"step","subtask":1,"tier":"skill|cache|executor|planner","action":{"type":"click","x":0.42,"y":0.13},
 "t_model":0.31,"t_total":0.74}
{"ev":"device","profile":{"os":"macos","rect":[0,0,1920,1080],"input":"mouse","connected_at":0}}
```

## 6. Orchestration (built: see PLAN.md §13 and HANDOFF.md)
**Jev 1.14 + Gemini 3.8 Flash, hosted APIs, B200 deferred.** In `brain/agent/jev_flash.py`:
- Flash plans **cached stages** from the screenshot. Each stage has a `ready` condition, an `expected` result and
  1–5 actions, plus one `completion` condition for the whole goal.
- Jev checks each stage on a fresh screenshot in one batched request: routing (`execute` / `ask_flash`), target
  presence, uniqueness and clarity, a `box` for the click, and verification of the previous stage.
- Explicit single pointer commands ("click Save") skip Flash entirely.
- Uncertainty or an unexpected screen means Flash replans. Real ambiguity means asking the user.

Measured so far (HANDOFF.md, local browser backend): Jev about 0.20–0.22 s per request; a direct click in 1.3 s with
no Flash call; a 6-field calendar task in 27–31 s with 3–4 Flash calls.

### Ideas to try on top (proposals, measure each)
1. **Zoom and ask again for small targets.** In our first manual test, "Code" and "Star" boxes were centered, but the
   small "Sign in" box drifted to the button's edge. When a box is small or its confidence is low, crop about 3×
   around it at full resolution, ask `box` again and map back.
2. **Screen-fingerprint cache.** Hash of the settled frame plus the stage → the action that worked. A hit skips
   Jev and Flash, so repeat tasks get faster (good for the benchmark slide).
3. **Race skills against Flash.** Start the Flash plan and the skill check together; if `open_app` covers the
   request, act right away and drop the plan.
4. **Fill silence.** Speak the first stage ("Opening Calendar…") as soon as the plan arrives. The calendar task
   takes about 30 s, so narration matters.
5. **Type the user's own words** (from computah): the model points at which transcript words are the value and code
   copies them exactly, so a spoken email address isn't retyped by a model.
6. **macOS Voice Control overlay.** Setting Overlay to *Item Numbers* puts a number badge on every clickable item,
   taken from the Mac's accessibility tree, and we can see it through HDMI with nothing installed. It could help Jev
   on dense Mac screens. Test it on the MacBook.

### Sensitive data
Screenshots go to TypeSafe and Google. Before sending, black out password fields (rows of dots) and SSN, card and
routing-number patterns found by local OCR on the settled frame. OCR runs while we wait for settle anyway, and the
mask is reused until the screen changes. Make it a **privacy mode** toggle and benchmark it on and off. If a password
field has focus, ask the user instead of typing it.

## 7. Interfaces still needed (voice and the projector view)
```python
class Voice(Protocol):                      # brain/voice/ (Demo)
    def say(self, text: str) -> None: ...   # non-blocking
    def earcon(self, name: Literal["listen", "working", "done", "error", "need_you"]) -> None: ...
    def ask(self, question: str, timeout_s: float = 6.0) -> str | None: ...  # opens mic, no button
    def confirm(self, prompt: str) -> bool: ...

# Voice → agent: utterances feed the same path the typed-goal REPL uses today; a button press during execution
# triggers the existing cancel token (§5.6), then the next utterance is handled as a correction.
```
The agent's `ask_user` action maps to `Voice.ask`, and destructive steps to `Voice.confirm`. The projector view
subscribes to the event stream in section 5.

## 8. Needs team approval
1. **Event stream format** (section 5): a new contract between Agent and Demo.
2. **Speech-to-text provider** (section 4): Deepgram streaming first, whisper.cpp on the Pi as backup?
3. **Voice interface** (section 7): agree on it with Demo before either side builds against it.
