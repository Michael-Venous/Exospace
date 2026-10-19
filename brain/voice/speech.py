"""Spoken narration (PLAN.md §11): Deepgram Aura reads the agent's lines aloud through the page.

Chromium on the Pi has no speech-dispatcher, so the browser's own speechSynthesis is silent there.
Aura streams 24 kHz 16-bit PCM back on the same DEEPGRAM_API_KEY (first audio in 0.3 s on the Pi),
and the page plays it with the player Describe mode already uses. Talking cuts a line off.

Each line's audio is kept (in memory and in tts_cache/), so a repeated line plays at once. Aura reads
the same text a little differently each time, so the fixed replies get TAKES renderings at startup and
one is picked at random: "Done." never sounds canned, and even the first one is instant.
"""

from __future__ import annotations

import base64
import hashlib
import random
import threading
from pathlib import Path
from typing import Any, Callable

import httpx

SPEAK_URL = "https://api.deepgram.com/v1/speak"
VOICE = "aura-2-thalia-en"
RATE = 24000
CHUNK = RATE // 5 * 2  # 0.2 s of 16-bit mono: small enough that playback starts on the first one
CACHE_DIR = Path("tts_cache")
MAX_CACHED = 300  # lines kept in memory; questions vary, the fixed replies are a handful
TAKES = 5  # renderings kept per fixed reply


class Speaker:
    def __init__(self, key: str, voice: str = VOICE, transport: httpx.BaseTransport | None = None,
                 cache_dir: Path | None = CACHE_DIR) -> None:
        self.client = httpx.Client(timeout=10.0, transport=transport,
                                   headers={"Authorization": f"Token {key}", "Content-Type": "application/json"})
        self.voice = voice
        self._lock = threading.Lock()
        self._line = 0  # bumped by cancel(): a line still streaming stops at its next chunk
        self.cache_dir = cache_dir
        self._audio: dict[str, list[bytes]] = {}  # text -> whole renderings of it, once spoken in full

    def cancel(self) -> None:
        with self._lock:
            self._line += 1

    def prerender(self, lines: list[str], takes: int = TAKES) -> None:
        """Fetch `takes` renderings of each line ahead of time (on a background thread at startup)."""
        for text in lines:
            for _ in range(takes - len(self._takes(text))):
                self._fetch(text, lambda chunk: None, cancellable=False)

    def say(self, text: str, emit: Callable[[dict[str, Any]], None], cancellable: bool = True) -> bool:
        """Stream `text` to emit as {"pcm", "rate"} chunks; False if nothing could be spoken."""
        mine = 0
        if cancellable:
            with self._lock:
                self._line += 1
                mine = self._line
        if takes := self._takes(text):
            audio = random.choice(takes)
            for start in range(0, len(audio), CHUNK):
                if cancellable and self._line != mine:
                    return start > 0
                emit({"pcm": base64.b64encode(audio[start:start + CHUNK]).decode(), "rate": RATE})
            return bool(audio)
        return self._fetch(text, emit, cancellable, mine)

    def _fetch(self, text: str, emit: Callable[[dict[str, Any]], None], cancellable: bool, mine: int = 0) -> bool:
        """Stream one new rendering of `text` from Aura and keep it once heard in full."""
        params = {"model": self.voice, "encoding": "linear16", "sample_rate": RATE, "container": "none"}
        sent, carry, whole = False, b"", bytearray()
        try:
            with self.client.stream("POST", SPEAK_URL, params=params, json={"text": text}) as response:
                if response.status_code != 200:
                    return False
                for data in response.iter_bytes(CHUNK):
                    if cancellable and self._line != mine:  # the user started talking, or a newer line took over
                        return sent
                    data, carry = (carry + data), b""
                    if len(data) % 2:  # never split a 16-bit sample between chunks
                        data, carry = data[:-1], data[-1:]
                    if data:
                        emit({"pcm": base64.b64encode(data).decode(), "rate": RATE})
                        whole += data
                        sent = True
        except httpx.HTTPError:
            return sent
        if sent:  # only a line heard to the end is kept: a cut-off one would replay cut off
            self._keep(text, bytes(whole))
        return sent

    def _dir(self, text: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / hashlib.sha256(f"{self.voice}|{text}".encode()).hexdigest()[:24]

    def _takes(self, text: str) -> list[bytes]:
        if text not in self._audio:
            folder = self._dir(text)
            files = sorted(folder.glob("*.pcm")) if folder is not None and folder.is_dir() else []
            if files:
                self._audio[text] = [f.read_bytes() for f in files]
        return self._audio.get(text, [])

    def _keep(self, text: str, audio: bytes) -> None:
        if text not in self._audio and len(self._audio) >= MAX_CACHED:
            self._audio.pop(next(iter(self._audio)))
        takes = self._audio.setdefault(text, [])
        takes.append(audio)
        folder = self._dir(text)
        if folder is not None:
            try:
                folder.mkdir(parents=True, exist_ok=True)
                tmp = folder / f"{len(takes)}.tmp"
                tmp.write_bytes(audio)
                tmp.replace(folder / f"{len(takes)}.pcm")  # never a half-written clip
            except OSError:
                pass

    def close(self) -> None:
        self.client.close()
