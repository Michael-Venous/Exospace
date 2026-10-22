"""Talk to the agent on the laptop: hold Home, speak, let go (bench only, runs ON the laptop).

While Home is held the microphone streams (pw-record) to Deepgram, starting half a second before the key went down
(the microphone stays open; only that half second stays in memory), and the agent hears the words as they come, as
`laptop_tests.py` speaks them: it can start before you finish. On release Deepgram finalizes the last words (about
0.1 s) and the request is complete. The agent acts on the screen through the uinput keyboard and mouse; what it says
or asks is spoken back by Piper on the laptop (no key, about 0.05 s a sentence). Holding Home again while it works is
a new request: the newest wins. Home still reaches the app in front (the keyboard isn't grabbed); the agent's own
virtual keyboard is never listened to. Reading the keyboard needs the `input` group. Ctrl+C stops.

Each model reply and action prints with its time, and each request ends with a timing line (first action and end,
from letting go of Home), also appended to ~/exo-bench/talk-logs/timings.csv.

  ~/exo-bench/talk.sh
"""

from __future__ import annotations

import csv
import fcntl
import io
import json
import os
import queue
import select
import struct
import subprocess
import threading
import time
import wave
from collections import deque
from pathlib import Path

from piper import PiperVoice
from websockets.sync.client import connect

from bench.arch.laptop import SESSION, LaptopTarget
from bench.arch.queue_agent import QueueAgent

# Home, and the keypad's 7/Home, which this laptop's Home key sends (code 71; Home while Num Lock is off).
HOME_KEYS = {102, 71}
EV_SYN, EV_KEY, EV_LED, KEY_A, KEY_MAX = 0, 1, 0x11, 30, 0x2FF
EVIOCGRAB, UI_SET_EVBIT, UI_SET_KEYBIT, UI_SET_LEDBIT = 0x40044590, 0x40045564, 0x40045565, 0x40045569
UI_DEV_SETUP, UI_DEV_CREATE, UI_DEV_DESTROY = 0x405C5503, 0x5501, 0x5502
EVENT = struct.Struct("llHHi")  # struct input_event on 64-bit Linux
RATE = 16000
PREROLL_S = 0.5  # audio from before Home went down that each request keeps
DEEPGRAM = (f"wss://api.deepgram.com/v1/listen?model=nova-3&encoding=linear16&sample_rate={RATE}&channels=1"
            "&interim_results=true&smart_format=true&endpointing=false")
VOICE = Path.home() / "exo-bench/voices/en_US-lessac-medium.onnx"  # Piper's voice (rhasspy/piper-voices)
MODEL = "deepseek/deepseek-v4.1-flash-1017"
LOGS = Path.home() / "exo-bench/talk-logs"
TIMINGS = LOGS / "timings.csv"  # one row per request, every session
FIELDS = ["when", "words", "spoken_s", "first_action_s", "finished_s", "model_calls", "actions"]


def keyboards() -> list[str]:
    """Event devices of the physical keyboards (they have letter keys: not power buttons or hotkey strips), not
    virtual ones (the agent's, this script's pass-through, other software's)."""
    found = []
    for block in Path("/proc/bus/input/devices").read_text().split("\n\n"):
        fields = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        bits = int("".join(w.zfill(16) for w in fields.get("B: KEY", "0").split()), 16)
        if fields.get("S: Sysfs", "").startswith("/devices/virtual") or not bits >> KEY_A & 1:
            continue  # other software's virtual keyboards (this agent's, an input switcher) are left alone
        found += [f"/dev/input/{h}" for h in fields.get("H: Handlers", "").split() if h.startswith("event")]
    return found


class PassThrough:
    """The real keyboards held by this script alone (EVIOCGRAB), every key but Home sent on through a virtual one:
    Home talks to the agent and never reaches the app in front. The kernel lets go of them when this process ends,
    however it ends. Caps/Num Lock lights are passed back."""

    def __init__(self, paths: list[str]) -> None:
        self.real = [os.open(p, os.O_RDWR) for p in paths]
        self.virtual = os.open("/dev/uinput", os.O_RDWR | os.O_NONBLOCK)
        for kind in (EV_KEY, EV_LED):
            fcntl.ioctl(self.virtual, UI_SET_EVBIT, kind)
        for code in range(1, KEY_MAX):
            fcntl.ioctl(self.virtual, UI_SET_KEYBIT, code)
        for led in range(3):  # Num, Caps, Scroll Lock
            fcntl.ioctl(self.virtual, UI_SET_LEDBIT, led)
        fcntl.ioctl(self.virtual, UI_DEV_SETUP, struct.pack("4H80sI", 0x03, 0x1209, 0xE506, 1,
                                                             b"Exospace pass-through keyboard", 0))
        fcntl.ioctl(self.virtual, UI_DEV_CREATE)
        time.sleep(0.3)  # the compositor picks the new keyboard up before the real ones go quiet
        for fd in self.real:
            fcntl.ioctl(fd, EVIOCGRAB, 1)

    def forward(self, data: bytes) -> list[tuple[int, int]]:
        """Pass a real keyboard's events on, all but Home's; Home's (code, value) presses come back."""
        out, home = [], []
        for i in range(0, len(data) - EVENT.size + 1, EVENT.size):
            sec, usec, kind, code, value = EVENT.unpack_from(data, i)
            if kind == EV_KEY and code in HOME_KEYS:
                home.append((code, value))
            elif kind in (EV_KEY, EV_SYN):
                out.append(EVENT.pack(sec, usec, kind, code, value))
        if out:
            os.write(self.virtual, b"".join(out))
        return home

    def lights(self) -> None:
        """Lock lights the desktop set on the virtual keyboard, shown on the real ones."""
        try:
            data = os.read(self.virtual, EVENT.size * 16)
        except BlockingIOError:
            return
        for i in range(0, len(data) - EVENT.size + 1, EVENT.size):
            sec, usec, kind, code, value = EVENT.unpack_from(data, i)
            if kind == EV_LED:
                for fd in self.real:
                    os.write(fd, EVENT.pack(sec, usec, kind, code, value) + EVENT.pack(sec, usec, EV_SYN, 0, 0))

    def close(self) -> None:
        for fd in self.real:
            os.close(fd)  # closing lets go of the grab
        fcntl.ioctl(self.virtual, UI_DEV_DESTROY)
        os.close(self.virtual)


class Mic:
    """The microphone, always open, keeping only its last PREROLL_S in memory: the first word, said as Home goes
    down, is in it ("close firefox" was heard as "firefox": the microphone and Deepgram took ~0.5 s to start).
    Nothing leaves the laptop until Home is held."""

    def __init__(self, env: dict) -> None:
        self.proc = subprocess.Popen(["pw-record", "--rate", str(RATE), "--channels", "1", "--format", "s16", "-"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
        self.recent: deque[bytes] = deque(maxlen=round(PREROLL_S * 10))
        self.listener: queue.Queue | None = None
        self.lock = threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        while chunk := self.proc.stdout.read(RATE // 10 * 2):  # 100 ms of 16-bit samples
            with self.lock:
                self.recent.append(chunk)
                if self.listener is not None:
                    self.listener.put(chunk)

    def listen(self) -> queue.Queue:
        """From the last PREROLL_S on, until `stop`."""
        with self.lock:
            self.listener = queue.Queue()
            for chunk in self.recent:
                self.listener.put(chunk)
            return self.listener

    def stop(self) -> None:
        with self.lock:
            if self.listener is not None:
                self.listener.put(None)
            self.listener = None


class Utterance:
    """One press of Home: the microphone streamed to Deepgram; `heard(words, speaking)` gets the words so far."""

    def __init__(self, key: str, mic: Mic, heard) -> None:
        self.heard, self.final, self.interim, self.mic = heard, [], "", mic
        self.audio = mic.listen()  # audio buffers here while Deepgram connects
        self.ws = connect(DEEPGRAM, additional_headers={"Authorization": f"Token {key}"})
        self.done = threading.Event()
        threading.Thread(target=self._send, daemon=True).start()
        self.reader = threading.Thread(target=self._receive, daemon=True)
        self.reader.start()

    def words(self) -> str:
        return " ".join([*self.final, self.interim]).strip()

    def _send(self) -> None:
        while (chunk := self.audio.get()) is not None:
            self.ws.send(chunk)
        self.ws.send(json.dumps({"type": "Finalize"}))

    def _receive(self) -> None:
        try:
            while True:
                message = json.loads(self.ws.recv(timeout=15))
                if message.get("type") != "Results":
                    continue
                text = message["channel"]["alternatives"][0]["transcript"]
                if message.get("is_final"):
                    self.final += [text] if text else []
                    self.interim = ""
                else:
                    self.interim = text
                if message.get("from_finalize"):
                    break
                if self.words():
                    self.heard(self.words(), True)
        except Exception:  # noqa: BLE001 - a dropped stream ends the utterance with the words it had
            pass
        self.done.set()

    def finish(self) -> str:
        """Let go of Home: the last words, once Deepgram has finalized them (at most 2 s)."""
        self.mic.stop()
        self.done.wait(2)
        self.ws.close()
        return self.words()


class Speaker:
    def __init__(self, env: dict) -> None:
        self.voice, self.env = PiperVoice.load(str(VOICE)), env

    def say(self, text: str) -> None:
        sound = io.BytesIO()
        with wave.open(sound, "wb") as out:
            self.voice.synthesize_wav(text, out)
        subprocess.run(["pw-play", "-"], input=sound.getvalue(), env=self.env, check=False)


def main() -> None:
    try:
        keys = PassThrough(keyboards())
    except PermissionError:
        raise SystemExit("Can't read the keyboard: add yourself to the input group (sudo usermod -aG input $USER), "
                         "then log out and back in.") from None
    env = {**os.environ, **SESSION}
    speaker, mic = Speaker(env), Mic(env)
    session = LOGS / time.strftime("%Y%m%d-%H%M%S")  # each call's screenshot and Jev's log land here
    session.mkdir(parents=True, exist_ok=True)
    agent = QueueAgent(LaptopTarget(), MODEL, "", os.environ["TYPESAFE_API_KEY"], session, "qwen")
    print("Ready: hold Home and speak; let go when you're done. Ctrl+C stops.", flush=True)

    def speak_back() -> None:  # what the agent says or asks, aloud
        spoken = 0
        while True:
            said = agent.brain.said
            while spoken < len(said):
                print(f"  agent: {said[spoken]}", flush=True)
                speaker.say(said[spoken])
                spoken += 1
            time.sleep(0.2)

    requests: list[dict] = []  # each press of Home: when it was pressed and let go, in agent seconds

    def summarize(request: dict, finished: float | None) -> None:
        """One request's timings from letting go of Home: its first action (negative: it started while you still
        spoke) and its end (the agent said, asked or confirmed done)."""
        pressed, released = request["pressed"], request["released"]
        actions = [r for r in agent.shared.log if r.t >= pressed]
        calls = [c for c in agent.brain.calls if c["t"] >= pressed and not c.get("cancelled")]
        row = {"when": time.strftime("%H:%M:%S"), "words": request["words"],
               "spoken_s": round(released - pressed, 2),
               "first_action_s": round(actions[0].t - released, 2) if actions else "",
               "finished_s": round(finished - released, 2) if finished else "",
               "model_calls": len(calls), "actions": len(actions)}
        print(f"  timing: first action {row['first_action_s']} s, finished {row['finished_s'] or 'not yet'} s after "
              f"you let go; {row['model_calls']} model calls, {row['actions']} actions", flush=True)
        new = not TIMINGS.exists()
        with TIMINGS.open("a", newline="") as f:
            out = csv.DictWriter(f, FIELDS)
            if new:
                out.writeheader()
            out.writerow(row)

    def save(folder: Path) -> None:  # every call, action and click point so far, for looking into a run later
        (folder / "calls.json").write_text(json.dumps(agent.brain.calls, indent=0, default=str))
        (folder / "log.json").write_text(json.dumps([{"t": r.t, "action": r.action, "result": r.result,
                                                      "detail": r.detail} for r in agent.shared.log], indent=0))
        (folder / "trace.json").write_text(json.dumps(agent.trace, indent=0, default=str))

    def trace() -> None:  # each model reply and each action as it runs, in seconds since the agent started
        calls = ran = 0
        saved = (0, 0)
        while True:
            open_ = [r for r in requests if "words" in r and not r.get("summarized")]
            for request in open_:
                if agent.shared.finished or request is not requests[-1]:  # done, or overtaken by a new request
                    summarize(request, time.monotonic() - agent.started if agent.shared.finished else None)
                    request["summarized"] = True
            for call in agent.brain.calls[calls:]:
                if call.get("total_s") is not None or call.get("error") or call.get("cancelled"):
                    # a cancelled call's actions that had already streamed still run: show them too
                    print(f"  [{call['t']:.2f}] model {'(cancelled by new words) ' if call.get('cancelled') else ''}"
                          f"first token {call.get('ttft_s')} s: "
                          f"{[{k: v for k, v in a.items() if k != 'at_s'} for a in call['lines']] or call.get('error', '')}",
                          flush=True)
                    calls += 1
                else:
                    break
            for record in agent.shared.log[ran:]:
                point = next((t["point"] for t in reversed(agent.trace)
                              if t.get("label") == "ground" and t.get("target") == next(iter(record.action.values()))),
                             None) if {"click", "double_click", "right_click"} & record.action.keys() else None
                print(f"  [{record.t:.2f}] ran {record.action}{f' at {point}' if point else ''} -> {record.result} "
                      f"{record.detail}", flush=True)
                ran += 1
            if (calls, ran) != saved:  # only when something new happened
                save(session)
                saved = (calls, ran)
            time.sleep(0.1)

    threading.Thread(target=speak_back, daemon=True).start()
    threading.Thread(target=trace, daemon=True).start()
    talking: Utterance | None = None
    try:
        while True:
            ready, _, _ = select.select([*keys.real, keys.virtual], [], [])
            if keys.virtual in ready:
                keys.lights()
            for fd in ready:
                if fd == keys.virtual:
                    continue
                for code, value in keys.forward(os.read(fd, EVENT.size * 64)):
                    if value == 2:  # key repeat
                        continue
                    if value == 1 and talking is None:
                        agent.brain.end_turn()  # the last request is over; this is the next one
                        requests.append({"pressed": time.monotonic() - agent.started})
                        talking = Utterance(os.environ["DEEPGRAM_API_KEY"], mic, agent.brain.hear)
                        print("listening...", flush=True)
                    elif value == 0 and talking is not None:
                        released = time.monotonic() - agent.started
                        words, talking = talking.finish(), None
                        print(f"[{released:.2f}] you: {words or '(nothing heard)'}", flush=True)
                        if words:
                            requests[-1].update(released=released, words=words)
                            agent.brain.hear(words, speaking=False)
                        else:
                            requests.pop()
    except KeyboardInterrupt:
        pass
    finally:
        keys.close()
        agent.close()
        agent.controller.hid.close()


if __name__ == "__main__":
    main()
