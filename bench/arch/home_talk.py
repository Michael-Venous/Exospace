"""Hold Home anywhere to talk to the agent server (serve_agent.py), with no browser: the laptop rig's own push-to-talk.

Pressing Home stops whatever the agent is doing; the microphone streams to Deepgram while it is held (talk.py's Mic
and Utterance, half a second of pre-roll included); letting go sends the whole request to POST /goal. What the agent
says comes back on GET /events and is spoken with Piper, never while Home is held. The voice page and its Space key
are a separate front end (`serve_laptop.sh --page`); this one never touches the page.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request

AGENT = os.environ.get("EXO_AGENT", "http://127.0.0.1:8770")


def post(path: str, body: dict | None = None) -> None:
    """A request to the agent; one it can't take is reported, never fatal (the key must keep working)."""
    request = urllib.request.Request(AGENT + path, json.dumps(body or {}).encode(), {"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(request, timeout=5).close()
    except OSError as exc:
        print(f"  (agent not reachable at {AGENT}: {exc})", flush=True)


class HomeTalk:
    """Home's press and release, turned into /stop and /goal; the agent's lines, spoken."""

    def __init__(self, post, speaker, listen) -> None:
        """`listen()` starts one utterance and returns it (its `finish()` gives the words heard)."""
        self.post, self.speaker, self.listen = post, speaker, listen
        self.held = threading.Event()
        self.talking = None

    def changed(self, down: bool) -> None:
        if down and self.talking is None:
            self.held.set()
            self.speaker.stop()
            self.post("/stop")  # Home stops the agent at once, as in talk.py
            self.talking = self.listen()
            print("listening...", flush=True)
        elif not down and self.talking is not None:
            words, self.talking = self.talking.finish(), None
            self.held.clear()
            print(f"you: {words or '(nothing heard)'}", flush=True)
            if words:
                self.post("/goal", {"text": words})

    def hear(self, event: dict) -> None:
        """One agent event: a line it says (a result the page would speak word for word) is spoken."""
        if event.get("ev") == "result" and event.get("status") == "ask_user" and event.get("message"):
            print(f"  agent: {event['message']}", flush=True)
            if not self.held.is_set():
                self.speaker.say(event["message"])

    def follow(self) -> None:
        """The agent's event stream, reconnected whenever it drops (the agent may start after this)."""
        while True:
            try:
                with urllib.request.urlopen(AGENT + "/events", timeout=60) as stream:
                    for raw in stream:
                        line = raw.decode().strip()
                        if line.startswith("data: "):
                            self.hear(json.loads(line[6:]))
            except OSError:
                time.sleep(1)


def main() -> None:
    from bench.arch.laptop import SESSION
    from bench.arch.talk import Mic, Speaker, Utterance
    from brain.voice import talk_key

    env = {**os.environ, **SESSION}
    speaker, mic = Speaker(env), Mic(env)
    key = os.environ["DEEPGRAM_API_KEY"]
    home = HomeTalk(post, speaker, lambda: Utterance(key, mic, lambda words, speaking: None))
    threading.Thread(target=home.follow, daemon=True).start()
    print("Ready: hold Home anywhere and speak; let go when you're done. Ctrl+C stops.", flush=True)
    try:
        talk_key.watch(home.changed)
    except PermissionError:
        raise SystemExit("Can't read the keyboard: add yourself to the input group (sudo usermod -aG input $USER), "
                         "then log out and back in.") from None
    except KeyboardInterrupt:
        pass
    finally:
        speaker.stop()
        mic.close()


if __name__ == "__main__":
    main()
