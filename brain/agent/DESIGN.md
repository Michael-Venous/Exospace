# Agent design: how everything works together

**Owner:** Agent. **Status:** proposal. Section 8 lists what needs team approval before it touches PLAN.md §5.

This covers what happens from the moment the user wants something until it's done: who it's for, what they hear and
see, how speech gets in, and how the smart model (Gemini) and the fast model (UI-TARS on the B200s, or Jev picking from a numbered list) share
the work.

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

## 6. Orchestration: smart model plans, fast model acts
Each tier is only used when the cheaper one can't handle the moment.

| Tier | Who | Target latency | Job |
|---|---|---|---|
| 0. Skill / cache | Controller (no AI) | < 0.1 s | Per-OS keyboard skills (§6 in PLAN.md); replay a cached action for a screen seen before |
| 1. Executor | UI-TARS on the B200s (candidate), **Jev picking from a numbered list (candidate)**, Gemini (baseline) | ~0.3–1 s | Screen + subtask → ONE next action (§5.3) |
| 2. Planner | Gemini | 1.5–3 s | Goal + memory + device → subtasks; repair when stuck |

**Rule: Gemini never runs between individual actions.** It runs on a new request, a device change, or when the
executor needs help.

### Escalate to the planner when
1. The executor returns `fail` or `ask_user`, or its confidence is below the threshold.
2. The screen doesn't match the current subtask (surprise dialog, wrong app).
3. Three verifications in a row fail. This is already a cancel trigger in §5.6.

The planner then returns a revised list of remaining subtasks. **Risky actions** (send, delete, pay) go to the
*user* for a spoken "yes," not to a model.

### Hide latency
- **Race at the start:** when the transcript arrives, start the Gemini plan call *and* check skills and the cache at
  the same time. If a skill can do the whole request ("open Calendar"), act now and cancel Gemini.
- **Fill silence:** say "Okay, ⟨first subtask⟩" the moment the plan arrives.
- **Look ahead:** while the executor works on subtask k, the planner can prepare k+1 (only after it's stable).
- **Screen-fingerprint cache:** hash the settled frame (downscaled perceptual hash + the subtask text) → the action
  that worked. A cache hit skips the model entirely. Repeat runs get faster, which suits the benchmark slide.

### Pick from a numbered list instead of guessing pixels (benchmark it, don't assume it)
Asking a model to output exact click coordinates from a screenshot is slow and error-prone. Turning the screen into
a **numbered list of clickable things** lets a small, fast *text* model answer with one number:
1. **Where the list comes from:**
   - **macOS Voice Control overlay (best on Macs).** System Settings → Accessibility → Voice Control → Overlay:
     *Item Numbers* keeps a number badge on every clickable element, all the time. Those numbers come from the
     Mac's own accessibility tree, so we get the tree **through the HDMI pixels** with nothing installed, only a
     setting turned on once. OCR the badges, and you know the position of every button.
   - **Everywhere else:** local OCR text boxes (a button labeled "Save" becomes option 7).
2. **Jev** picks the number for the subtask (about 0.3 s, with a probability per option).
3. The badge or box center becomes normalized coordinates (§5.2), then a normal `click`.

It can't see unlabeled icons, so UI-TARS or Gemini handle those screens. Add it as a `config` in
`bench/results.csv` and let the numbers decide (PLAN.md §8).

**Jev call** (format from computah's `docs/JEV_REQUESTS.md`): `POST https://api.typesafe.ai/v1/systemone`,
model `jev-1.13.0`, a `state` (subtask, scene summary, candidate list) and a `choice` question whose `criteria` map
IDs to options, plus `none_here`. Below 0.7, or `none_here`, means escalate. Also available on OpenRouter as
`typesafe/jev-1.13`. Put the key in `.env` as `TYPESAFE_API_KEY`.

**Why not the real accessibility API?** That's what makes Mac-only voice agents fast, but it requires software
running *on* the target, which breaks "nothing installed" and our hardware story. The Voice Control overlay is the
closest we can get from outside the machine. Test it tonight on the MacBook.

### What to copy from computah (our speed baseline)
[musubipapi/computah](https://github.com/musubipapi/computah) (MIT) is the fast Mac voice agent the team wants to
match. It reads the macOS accessibility tree (no screenshots), lets Deepgram Flux handle speech, and asks Jev one
*choice question* per step. Ideas that carry over even though we only have pixels and keyboard/mouse:
1. **Ask a fast model to choose, never to write.** Each step is "here are the options, pick one ID," with a
   `none_here` answer (our escalation) and a probability per option. Code maps the ID back to a real target.
   We use Jev itself. **Backup if Jev goes down again:** vLLM on the B200 supports constrained choice output (the model can only answer one of the given
   IDs) plus logprobs, which gives the probability. A small, fast model served this way plays Jev's role.
2. **Group the screen into choices first.** They don't send the whole tree, just a short list of actionable
   candidates ("AXButton; label=Done"). Ours: Voice Control badges or OCR boxes, as above.
3. **First pick the route:** `application` (open an app) / `navigation` (open a URL) / `controls`. Our first two
   are controller skills, so most requests never need the screen at all.
4. **Type the user's own words, not the model's.** The model picks *which words* of the transcript are the value
   (by token position), and code copies them exactly. No retyped or made-up text.
5. **Never repeat an uncertain action** to see whether it worked. A keystroke over HID can't be undone, so read the
   screen again and escalate instead.
6. **A new speech turn revokes permission to send input**, which is our barge-in plus cancel path (§5.6). Key down
   and key up always stay paired (`release_all`).
7. **"Sent" is not "done."** Only evidence on the fresh screen counts as success.
8. **Prepare during speech, act after.** Deepgram's early end-of-turn event lets them read controls and pick an
   action before the user finishes, with no input sent until the final transcript matches. With push-to-talk, we can
   prepare from the partial transcript while the button is still held.

### Reaching the B200s (HiPerGator) from the Pi
Running a model there is fine, not just training:
1. Start a Slurm job on the B200 partition that runs vLLM (`vllm serve <model> --port 8000`). Never run it on a login
   node. Ask for a long `--time` so it lasts through judging.
2. From the Pi, tunnel through a login node: `ssh -N -L 8000:<compute-node>:8000 <gatorlink>@hpg.rc.ufl.edu`.
   The Pi then calls `EXECUTOR_BASE_URL=http://localhost:8000/v1`, with no code change.
3. The Pi needs an SSH key on the account for this. Use `autossh` so the tunnel reconnects after Wi-Fi drops.
4. Measure the round trip on UF Wi-Fi early (PLAN.md §10). Staying on campus should keep it small.

**Easier backup:** UF's **NaviGator Toolkit** gives an OpenAI-compatible API (`https://api.ai.it.ufl.edu/v1`) with a
GatorLink-made key, including models hosted on HiPerGator. No Slurm, no tunnel, but only the models they offer.

### Sensitive data never leaves the Pi unmasked
Gemini is a cloud model, and frames sent to it can contain passwords, card numbers, SSNs or bank routing numbers.
Before a frame goes to Gemini, the Pi runs local OCR and blacks out matching boxes: password fields, 9-digit
SSN-like numbers, 16-digit card-like numbers, and 9-digit routing numbers next to "routing". If a password field has
focus, the agent asks the user instead of typing it. Frames sent to our own B200 server are a lower risk, but get the
same masking if it's cheap enough. A good line for judges: "it blurs your SSN before any AI sees it."

## 7. Interfaces between modules (typed, small)
```python
class Voice(Protocol):                      # brain/voice/ (Demo)
    def say(self, text: str) -> None: ...   # non-blocking
    def earcon(self, name: Literal["listen", "working", "done", "error", "need_you"]) -> None: ...
    async def ask(self, question: str, timeout_s: float = 6.0) -> str | None: ...  # opens mic, no button
    async def confirm(self, prompt: str) -> bool: ...

# Voice → agent: one asyncio.Queue of input events
Utterance = TypedDict("Utterance", {"kind": Literal["utterance"], "text": str})
BargeIn = TypedDict("BargeIn", {"kind": Literal["barge_in"]})

class Executor(Protocol):                   # brain/agent/ calls server/ (Models)
    async def next_action(self, frame: Frame, subtask: str, profile: DeviceProfile) -> Step: ...

@dataclass
class Step:
    action: Action          # §5.3 schema
    confidence: float | None  # from vLLM logprobs where available, else None
    t_model: float

class Planner(Protocol):                    # brain/agent/ (Agent)
    async def plan(self, goal: str, memory: Session, profile: DeviceProfile) -> list[Subtask]: ...
    async def repair(self, memory: Session, profile: DeviceProfile, problem: str) -> list[Subtask]: ...
```

## 8. Needs team approval before it goes in PLAN.md §5
1. **Subtask fields** (§5.5 `task.subtasks` items): `{"text": str, "done_when": str | null}`. `done_when` is text
   expected on screen, so verification is cheap code instead of a model call.
2. **Executor confidence:** UI-TARS doesn't return a probability. Either Models exposes logprobs from vLLM, or we
   treat parse failures and `fail` as low confidence.
3. **Event stream format** (section 5): a new contract between Agent and Demo.
4. **Speech-to-text provider** (section 4): Deepgram first, Whisper on the Pi as backup?
