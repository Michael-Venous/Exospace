"""Spoken narration: Deepgram Aura audio streamed to the page as PCM chunks."""

import base64
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import httpx

from brain.voice.layer import Agent
from brain.voice.speech import CHUNK, RATE, Speaker


def aura(body: bytes, status: int = 200, calls: list | None = None) -> httpx.MockTransport:
    def reply(request):
        if calls is not None:
            calls.append(request)
        return httpx.Response(status, content=body)
    return httpx.MockTransport(reply)


def speaker(transport, cache_dir=None):
    return Speaker("k", transport=transport, cache_dir=cache_dir)


def audio(chunks) -> bytes:
    return b"".join(base64.b64decode(chunk["pcm"]) for chunk in chunks)


class SpeakerTests(unittest.TestCase):
    def test_a_line_streams_as_whole_sample_chunks(self):
        sent = []
        self.assertTrue(speaker(aura(b"\x01\x02" * CHUNK)).say("Done.", sent.append))
        audio = b"".join(base64.b64decode(chunk["pcm"]) for chunk in sent)
        self.assertEqual(audio, b"\x01\x02" * CHUNK)
        self.assertTrue(all(chunk["rate"] == RATE and len(base64.b64decode(chunk["pcm"])) % 2 == 0 for chunk in sent))

    def test_talking_cuts_a_line_off(self):
        voice, sent = speaker(aura(b"\x00\x00" * CHUNK * 4)), []

        def emit(chunk):
            sent.append(chunk)
            voice.cancel()  # the user pressed talk after the first chunk

        voice.say("This can't be undone. Should I go ahead?", emit)
        self.assertEqual(len(sent), 1)

    def test_an_error_says_nothing(self):
        sent = []
        self.assertFalse(speaker(aura(b"nope", 401)).say("Done.", sent.append))
        self.assertEqual(sent, [])


class SpeechCacheTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_a_repeated_line_plays_the_same_audio_without_asking_again(self):
        calls, clip = [], bytes(range(256)) * 40
        voice = speaker(aura(clip, calls=calls), self.dir)
        first, second = [], []
        voice.say("Done.", first.append)
        voice.say("Done.", second.append)
        self.assertEqual((audio(first), audio(second), len(calls)), (clip, clip, 1))

    def test_fixed_replies_get_five_takes_ahead_of_time_and_vary(self):
        calls, n = [], iter(range(100))
        transport = httpx.MockTransport(lambda r: calls.append(r) or httpx.Response(200, content=bytes([next(n)]) * 64))
        voice = speaker(transport, self.dir)
        voice.prerender(["Done.", "On it."])
        heard = set()
        for _ in range(40):
            chunks = []
            voice.say("Done.", chunks.append)
            heard.add(audio(chunks))
        self.assertEqual(len(calls), 10)  # five takes each, fetched once; every "Done." came from them
        self.assertGreater(len(heard), 1)  # and they rotate

    def test_a_cut_off_line_is_not_kept(self):
        calls, voice = [], None
        voice = speaker(aura(b"\x00\x00" * CHUNK * 4, calls=calls), self.dir)
        voice.say("Should I go ahead?", lambda chunk: voice.cancel())
        voice.say("Should I go ahead?", [].append)
        self.assertEqual(len(calls), 2)

    def test_the_cache_survives_a_restart(self):
        speaker(aura(b"\x05\x06" * 50), self.dir).say("Done.", [].append)
        calls, again = [], []
        speaker(aura(b"other", calls=calls), self.dir).say("Done.", again.append)
        self.assertEqual((audio(again), calls), (b"\x05\x06" * 50, []))


class NarrationVoiceTests(unittest.TestCase):
    def test_with_a_speaker_the_page_shows_the_line_and_the_server_says_it(self):
        speaker, lines = Mock(), queue.Queue()
        Agent("http://127.0.0.1:1", speaker=speaker)._voice(lines, "Chrome is open")
        self.assertEqual(lines.get_nowait(), {"say": "Chrome is open", "voiced": True})
        speaker.say.assert_called_once()  # on its own thread; Mock records the call either way

    def test_without_a_speaker_the_browser_speaks(self):
        lines = queue.Queue()
        Agent("http://127.0.0.1:1")._voice(lines, "Chrome is open")
        self.assertEqual(lines.get_nowait(), {"say": "Chrome is open"})


if __name__ == "__main__":
    unittest.main()
