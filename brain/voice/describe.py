"""Describe mode: Gemini Live says what is on the screen after each request (PLAN.md §11).

For a user who cannot see the target, a result line ("Done.") is not enough. With the voice page's
Describe toggle on, each finished request's screen goes to a Gemini Live model with the request and
its outcome, and the spoken reply streams back to the page as 24 kHz 16-bit PCM chunks: the first
audio arrived in 1.5 s on a YouTube results page. Talking cancels a description mid-sentence.
"""

from __future__ import annotations

import asyncio
import base64
import threading
from typing import Any, Callable

DESCRIBE_MODEL = "gemini-3.8-live"
SAMPLE_RATE = 24000  # Gemini Live's output audio
TIMEOUT_S = 20.0
SYSTEM = ("You describe a computer or phone screen aloud for a blind user, right after an assistant acted on "
          "their request. In one or two short sentences, say whether the request looks done and what is on "
          "screen that matters to them. Never mention coordinates, pixels, or colors unless they matter.")

Emit = Callable[[dict[str, Any]], None]


class Describer:
    def __init__(self, api_key: str, model: str = DESCRIBE_MODEL) -> None:
        from google import genai
        from google.genai import types

        self._client, self._types, self.model = genai.Client(api_key=api_key), types, model
        self._generation = 0
        self._lock = threading.Lock()

    def cancel(self) -> None:
        """Stop any description in progress (the user started talking)."""
        with self._lock:
            self._generation += 1

    def describe(self, jpeg: bytes, request: str, outcome: str, emit: Emit) -> bool:
        """Stream a spoken description through `emit`; False if nothing was said (the caller falls back)."""
        with self._lock:
            generation = self._generation
        try:
            return asyncio.run(asyncio.wait_for(self._stream(jpeg, request, outcome, emit, generation), TIMEOUT_S))
        except Exception:  # noqa: BLE001 - an unreachable or slow model must never cost the result line
            return False

    async def _stream(self, jpeg: bytes, request: str, outcome: str, emit: Emit, generation: int) -> bool:
        types = self._types
        config = types.LiveConnectConfig(response_modalities=["AUDIO"], system_instruction=SYSTEM)
        prompt = f"The user asked: {request}\nThe assistant reported: {outcome}\nDescribe the screen now."
        spoke = False
        async with self._client.aio.live.connect(model=self.model, config=config) as session:
            await session.send_client_content(turns=types.Content(role="user", parts=[
                types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), types.Part(text=prompt)]),
                turn_complete=True)
            async for message in session.receive():
                if self._generation != generation:
                    return True  # cut off by the user: they heard something or chose not to
                if message.data:
                    emit({"pcm": base64.b64encode(message.data).decode(), "rate": SAMPLE_RATE})
                    spoke = True
                if message.server_content and message.server_content.turn_complete:
                    break
        return spoke
