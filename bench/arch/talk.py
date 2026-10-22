"""Talk to the agent on the laptop: hold Home, speak, let go (bench only, runs ON the laptop).

While Home is held the microphone streams (pw-record) to Deepgram, and the agent hears the words as they come, as
`laptop_tests.py` speaks them: it can start before you finish. On release Deepgram finalizes the last words (about
0.1 s) and the request is complete. The agent acts on the screen through the uinput keyboard and mouse; what it says
or asks is spoken back by Piper on the laptop (no key, about 0.05 s a sentence). Holding Home again while it works is
a new request: the newest wins. Home still reaches the app in front (the keyboard isn't grabbed); the agent's own
virtual keyboard is never listened to. Reading the keyboard needs the `input` group. Ctrl+C stops.

  ~/exo-bench/talk.sh
"""

from __future__ import annotations

import io
import json
import os
import select
import struct
import subprocess
import threading
import time
import wave
from pathlib import Path

from piper import PiperVoice
from websockets.sync.client import connect

from bench.arch.laptop import SESSION, LaptopTarget
from bench.arch.queue_agent import QueueAgent

KEY_HOME, EV_KEY = 102, 1
EVENT = struct.Struct("llHHi")  # struct input_event on 64-bit Linux
RATE = 16000
DEEPGRAM = (f"wss://api.deepgram.com/v1/listen?model=nova-3&encoding=linear16&sample_rate={RATE}&channels=1"
            "&interim_results=true&smart_format=true&endpointing=false")
VOICE = Path.home() / "exo-bench/voices/en_US-lessac-medium.onnx"  # Piper's voice (rhasspy/piper-voices)
MODEL = "deepseek/deepseek-v4.1-flash-1017"
LOGS = Path.home() / "exo-bench/talk-logs"


def keyboards() -> list[str]:
    """Event devices of the real keyboards (any device with keys), not the agent's virtual one."""
    found, name = [], ""
    for line in Path("/proc/bus/input/devices").read_text().splitlines():
        if line.startswith("N: Name="):
            name = line.split("=", 1)[1].strip('"')
        elif line.startswith("H: Handlers=") and "kbd" in line and not name.startswith("Exospace"):
            found += [f"/dev/input/{h}" for h in line.split("=", 1)[1].split() if h.startswith("event")]
    return found


class Utterance:
    """One press of Home: the microphone streamed to Deepgram; `heard(words, speaking)` gets the words so far."""

    def __init__(self, key: str, env: dict, heard) -> None:
        self.heard, self.final, self.interim = heard, [], ""
        self.ws = connect(DEEPGRAM, additional_headers={"Authorization": f"Token {key}"})
        self.mic = subprocess.Popen(["pw-record", "--rate", str(RATE), "--channels", "1", "--format", "s16", "-"],
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
        self.done = threading.Event()
        threading.Thread(target=self._send, daemon=True).start()
        self.reader = threading.Thread(target=self._receive, daemon=True)
        self.reader.start()

    def words(self) -> str:
        return " ".join([*self.final, self.interim]).strip()

    def _send(self) -> None:
        while chunk := self.mic.stdout.read(RATE // 10 * 2):  # 100 ms of 16-bit samples
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
        self.mic.terminate()
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
        files = [open(d, "rb", buffering=0) for d in keyboards()]
    except PermissionError:
        raise SystemExit("Can't read the keyboard: add yourself to the input group (sudo usermod -aG input $USER), "
                         "then log out and back in.") from None
    env = {**os.environ, **SESSION}
    speaker = Speaker(env)
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

    threading.Thread(target=speak_back, daemon=True).start()
    talking: Utterance | None = None
    try:
        while True:
            ready, _, _ = select.select(files, [], [])
            for f in ready:
                data = f.read(EVENT.size * 64)
                for i in range(0, len(data) - EVENT.size + 1, EVENT.size):
                    _, _, kind, code, value = EVENT.unpack_from(data, i)
                    if kind != EV_KEY or code != KEY_HOME or value == 2:  # 2: key repeat
                        continue
                    if value == 1 and talking is None:
                        agent.brain.end_turn()  # the last request is over; this is the next one
                        talking = Utterance(os.environ["DEEPGRAM_API_KEY"], env, agent.brain.hear)
                        print("listening...", flush=True)
                    elif value == 0 and talking is not None:
                        words, talking = talking.finish(), None
                        print(f"you: {words or '(nothing heard)'}", flush=True)
                        if words:
                            agent.brain.hear(words, speaking=False)
    except KeyboardInterrupt:
        pass
    finally:
        agent.close()
        agent.controller.hid.close()


if __name__ == "__main__":
    main()
