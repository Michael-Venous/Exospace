"""Spoken narration (PLAN.md §11): Deepgram Aura reads the agent's lines aloud through the page.

Chromium on the Pi has no speech-dispatcher, so the browser's own speechSynthesis is silent there.
Aura streams 24 kHz 16-bit PCM back on the same DEEPGRAM_API_KEY (first audio in 0.3 s on the Pi),
and the page plays it with the player Describe mode already uses. Talking cuts a line off.
"""

from __future__ import annotations

import base64
import threading
from typing import Any, Callable

import httpx

SPEAK_URL = "https://api.deepgram.com/v1/speak"
VOICE = "aura-2-thalia-en"
RATE = 24000
CHUNK = RATE // 5 * 2  # 0.2 s of 16-bit mono: small enough that playback starts on the first one


class Speaker:
    def __init__(self, key: str, voice: str = VOICE, transport: httpx.BaseTransport | None = None) -> None:
        self.client = httpx.Client(timeout=10.0, transport=transport,
                                   headers={"Authorization": f"Token {key}", "Content-Type": "application/json"})
        self.voice = voice
        self._lock = threading.Lock()
        self._line = 0  # bumped by cancel(): a line still streaming stops at its next chunk

    def cancel(self) -> None:
        with self._lock:
            self._line += 1

    def say(self, text: str, emit: Callable[[dict[str, Any]], None]) -> bool:
        """Stream `text` to emit as {"pcm", "rate"} chunks; False if nothing could be spoken."""
        with self._lock:
            self._line += 1
            mine = self._line
        params = {"model": self.voice, "encoding": "linear16", "sample_rate": RATE, "container": "none"}
        sent, carry = False, b""
        try:
            with self.client.stream("POST", SPEAK_URL, params=params, json={"text": text}) as response:
                if response.status_code != 200:
                    return False
                for data in response.iter_bytes(CHUNK):
                    if self._line != mine:  # the user started talking, or a newer line took over
                        return sent
                    data, carry = (carry + data), b""
                    if len(data) % 2:  # never split a 16-bit sample between chunks
                        data, carry = data[:-1], data[-1:]
                    if data:
                        emit({"pcm": base64.b64encode(data).decode(), "rate": RATE})
                        sent = True
        except httpx.HTTPError:
            return sent
        return sent

    def close(self) -> None:
        self.client.close()
