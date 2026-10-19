"""Spoken narration: Deepgram Aura audio streamed to the page as PCM chunks."""

import base64
import queue
import unittest
from unittest.mock import Mock

import httpx

from brain.voice.layer import Agent
from brain.voice.speech import CHUNK, RATE, Speaker


def aura(body: bytes, status: int = 200) -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: httpx.Response(status, content=body))


class SpeakerTests(unittest.TestCase):
    def test_a_line_streams_as_whole_sample_chunks(self):
        sent = []
        self.assertTrue(Speaker("k", transport=aura(b"\x01\x02" * CHUNK)).say("Done.", sent.append))
        audio = b"".join(base64.b64decode(chunk["pcm"]) for chunk in sent)
        self.assertEqual(audio, b"\x01\x02" * CHUNK)
        self.assertTrue(all(chunk["rate"] == RATE and len(base64.b64decode(chunk["pcm"])) % 2 == 0 for chunk in sent))

    def test_talking_cuts_a_line_off(self):
        speaker, sent = Speaker("k", transport=aura(b"\x00\x00" * CHUNK * 4)), []

        def emit(chunk):
            sent.append(chunk)
            speaker.cancel()  # the user pressed talk after the first chunk

        speaker.say("This can't be undone. Should I go ahead?", emit)
        self.assertEqual(len(sent), 1)

    def test_an_error_says_nothing(self):
        sent = []
        self.assertFalse(Speaker("k", transport=aura(b"nope", 401)).say("Done.", sent.append))
        self.assertEqual(sent, [])


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
