"""Talk to the agent on the laptop: hold Home, speak, let go (bench only, runs ON the laptop).

While Home is held the microphone records (pw-record); on release Gemini Flash-Lite transcribes it (about 1 s) and
the agent hears the words, as `laptop_tests.py` speaks them, and acts on the screen through the uinput keyboard and
mouse. What the agent says or asks is spoken back by Piper on the laptop (no key, about 0.05 s a sentence). Holding Home again while it works is a new request: the
newest wins. Home still reaches the app in front (the keyboard isn't grabbed); the agent's own virtual keyboard is
never listened to. Reading the keyboard needs the `input` group. Ctrl+C stops.

  ~/exo-bench/talk.sh
"""

from __future__ import annotations

import os
import select
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import io
import wave

from google import genai
from google.genai import types
from piper import PiperVoice

from bench.arch.laptop import SESSION, LaptopTarget
from bench.arch.queue_agent import QueueAgent

KEY_HOME, EV_KEY = 102, 1
EVENT = struct.Struct("llHHi")  # struct input_event on 64-bit Linux
STT_MODEL = "gemini-3.5-flash-lite"  # minimal thinking: 1.1 s for a sentence; 3.8 Flash took 3.3 s
STT_PROMPT = "Write down exactly what is said, word for word. Reply with the words only."
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


class Voice:
    def __init__(self, key: str) -> None:
        self.stt = genai.Client(api_key=key)
        self.tts = PiperVoice.load(str(VOICE))
        self.env = {**os.environ, **SESSION}

    def record(self) -> subprocess.Popen:
        path = Path(tempfile.gettempdir()) / "exo_talk.wav"
        return subprocess.Popen(["pw-record", "--rate", "16000", "--channels", "1", str(path)], env=self.env)

    def transcribe(self, rec: subprocess.Popen) -> str:
        rec.terminate()
        rec.wait(timeout=5)
        path = Path(rec.args[-1])
        if path.stat().st_size < 8000:  # under a quarter second: a tap, not speech
            return ""
        reply = self.stt.models.generate_content(
            model=STT_MODEL, contents=[types.Part.from_bytes(data=path.read_bytes(), mime_type="audio/wav"), STT_PROMPT],
            config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="minimal")))
        return (reply.text or "").strip()

    def say(self, text: str) -> None:
        sound = io.BytesIO()
        with wave.open(sound, "wb") as out:
            self.tts.synthesize_wav(text, out)
        subprocess.run(["pw-play", "-"], input=sound.getvalue(), env=self.env, check=False)


def main() -> None:
    devices = keyboards()
    try:
        files = [open(d, "rb", buffering=0) for d in devices]
    except PermissionError:
        raise SystemExit("Can't read the keyboard: add yourself to the input group (sudo usermod -aG input $USER), "
                         "then log out and back in.") from None
    voice = Voice(os.environ["GEMINI_API_KEY"])
    LOGS.mkdir(parents=True, exist_ok=True)
    agent = QueueAgent(LaptopTarget(), MODEL, "", os.environ["TYPESAFE_API_KEY"],
                       LOGS / time.strftime("%Y%m%d-%H%M%S"), "qwen")
    print("Ready: hold Home and speak; let go when you're done. Ctrl+C stops.", flush=True)

    def speak_back() -> None:  # what the agent says or asks, aloud
        spoken = 0
        while True:
            said = agent.brain.said
            while spoken < len(said):
                print(f"  agent: {said[spoken]}", flush=True)
                voice.say(said[spoken])
                spoken += 1
            time.sleep(0.2)

    threading.Thread(target=speak_back, daemon=True).start()
    rec = None
    try:
        while True:
            ready, _, _ = select.select(files, [], [])
            for f in ready:
                data = f.read(EVENT.size * 64)
                for i in range(0, len(data) - EVENT.size + 1, EVENT.size):
                    _, _, kind, code, value = EVENT.unpack_from(data, i)
                    if kind != EV_KEY or code != KEY_HOME or value == 2:  # 2: key repeat
                        continue
                    if value == 1 and rec is None:
                        agent.brain.end_turn()  # the last request is over; this is the next one
                        rec = voice.record()
                        print("listening...", flush=True)
                    elif value == 0 and rec is not None:
                        words, rec = voice.transcribe(rec), None
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
