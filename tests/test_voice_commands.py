"""Commands for the box itself (talking speed, repeat): recognized whole, kept across restarts, never sent on."""

import json
import queue
import urllib.request
from unittest.mock import patch

import pytest

from brain.voice import settings
from brain.voice.commands import (MAX_SPEED, MIN_SPEED, STEP, Dictate, Echo, Private, Repeat, Speed, leave_private,
                                  parse)
from brain.voice.controls import PRIVATE_LINE, RESUME_GOAL, Controls, Reply
from brain.voice.layer import Agent, page_server, private_transcript


@pytest.mark.parametrize("words, speed", [
    ("talk faster", STEP),
    ("Talk faster.", STEP),
    ("Please speak a bit faster", STEP),
    ("talk slower", 1 / STEP),
    ("Speak more slowly, please.", 1 / STEP),
    ("Talk at 2x.", 2.0),
    ("talk at 2 x speed", 2.0),
    ("Speak two times faster.", 2.0),
    ("read at 1.5 times speed", 1.5),
    ("Talk at two and a half times speed", 2.5),
    ("speak 2 times slower", 0.5),
    ("Talk at half speed.", 0.5),
    ("Normal speed.", 1.0),
    ("Can you talk normally?", 1.0),
    ("set speech speed to 3", 3.0),
    ("Talk at 9x", MAX_SPEED),
])
def test_speed_commands(words, speed):
    command = parse(words)
    assert isinstance(command, Speed)
    assert command.apply(1.0) == round(speed, 2)


@pytest.mark.parametrize("words", ["Repeat that.", "say that again", "What did you say?", "Could you repeat that?"])
def test_repeat_commands(words):
    assert parse(words) == Repeat()


@pytest.mark.parametrize("words", [
    "type talk faster in the doc",
    "talk faster than the video",
    "open the speed test site",
    "search for normal speed of sound",
    "repeat that email to Janet",
    "what did you say in the message to Priya",
    "play it at 2x speed",
    "talk at 0x",
])
def test_requests_that_only_contain_the_words_go_to_the_agent(words):
    assert parse(words) is None


def test_speed_steps_stay_between_the_limits():
    assert Speed(factor=STEP).apply(MAX_SPEED) == MAX_SPEED
    assert Speed(factor=1 / STEP).apply(MIN_SPEED) == MIN_SPEED


def test_settings_round_trip_and_survive_a_broken_file(tmp_path):
    path = tmp_path / "exospace" / "voice.json"
    assert settings.load(path) == settings.Settings()
    settings.save(settings.Settings(speed=2.5), path)
    assert settings.load(path).speed == 2.5
    path.write_text("not json")
    assert settings.load(path) == settings.Settings()


def test_controls_answer_commands_and_remember_the_speed(tmp_path):
    path = tmp_path / "voice.json"
    controls = Controls(path)
    assert controls.handle("open firefox") == Reply(goal="open firefox")
    assert controls.handle("repeat that") == Reply(say="I haven't said anything yet.")
    assert controls.handle("Talk at 2x.") == Reply(say="Speaking at 2 times speed.")
    assert controls.handle("talk at 5x").say == "Speaking at 5 times speed."
    assert controls.handle("talk faster").say == "That's already the fastest I can talk."
    controls.said("Firefox is open.")
    assert controls.handle("Say that again.") == Reply(say="Firefox is open.")
    assert Controls(path).speed == MAX_SPEED  # a restart keeps it


def test_the_voice_page_answers_a_command_itself_and_sends_a_request_on(tmp_path):
    agent = Agent("http://127.0.0.1:9", controls=Controls(tmp_path / "voice.json"))
    sent = []
    agent.post = lambda path, body: sent.append((path, body))
    lines: queue.Queue = queue.Queue()
    server = page_server(agent, 0, 0, lines)
    port = server.server_address[1]

    def segment(text):
        request = urllib.request.Request(f"http://127.0.0.1:{port}/segment", json.dumps({"text": text}).encode(),
                                         {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{port}"})
        urllib.request.urlopen(request, timeout=5).close()

    try:
        segment("Talk at 2x.")
        assert sent == []
        assert lines.get_nowait() == {"say": "Speaking at 2 times speed.", "rate": 2.0}
        segment("open firefox")
        assert sent == [("/goal", {"text": "open firefox", "continues": False})]
    finally:
        server.shutdown()


@pytest.mark.parametrize("words, command", [
    ("Start dictating.", Dictate(True)),
    ("start dictation", Dictate(True)),
    ("Stop dictating.", Dictate(False)),
    ("I'm done dictating.", Dictate(False)),
    ("Echo off.", Echo(False)),
    ("turn echo on", Echo(True)),
])
def test_dictation_commands(words, command):
    assert parse(words) == command


@pytest.mark.parametrize("words", ["start dictating a letter to Janet", "open the dictation settings",
                                   "type echo off in the terminal"])
def test_dictation_words_inside_a_request_go_to_the_agent(words):
    assert parse(words) is None


def test_dictation_types_each_utterance_reads_it_back_and_stops(tmp_path):
    controls = Controls(tmp_path / "voice.json")
    assert controls.handle("Start dictating.") == Reply(say="Dictating. Say stop dictating when you're done.")
    first = controls.handle("Dear Janet,")
    assert first.say == "Dear Janet," and first.goal.endswith(": Dear Janet,") and "a space" not in first.goal
    assert "a space" in controls.handle("thanks for the notes.").goal  # later lines don't run into the last
    assert controls.handle("talk faster").goal is None  # commands still work while dictating
    assert controls.handle("echo off") == Reply(say="I won't read back what you dictate.")
    assert controls.handle("See you Friday.").say is None
    assert controls.handle("Stop dictating.") == Reply(say="Stopped dictating.")
    assert controls.handle("open firefox") == Reply(goal="open firefox")
    assert settings.load(tmp_path / "voice.json").echo is False


def test_deepgram_hears_spoken_punctuation_only_while_dictating(tmp_path):
    controls = Controls(tmp_path / "voice.json")
    assert controls.deepgram("wss://dg?x=1") == "wss://dg?x=1"
    controls.handle("start dictating")
    assert controls.deepgram("wss://dg?x=1").endswith("&dictation=true")


@pytest.mark.parametrize("words", ["I'll type it.", "I'll type my password", "I will type the password myself",
                                   "Let me type it", "Private mode.", "Stop looking", "please stop looking"])
def test_private_mode_commands(words):
    assert parse(words) == Private()


@pytest.mark.parametrize("words", ["type my password reset email", "I'll type it later in the doc",
                                   "open private mode in firefox", "stop looking for flights",
                                   "type I'll type it in the chat"])
def test_private_words_inside_a_request_go_to_the_agent(words):
    assert parse(words) is None


@pytest.mark.parametrize("words, leave", [("Done.", "done"), ("I'm done", "done"), ("finished", "done"),
                                          ("Cancel.", "cancel"), ("done my password is hunter2", None),
                                          ("hunter2", None), ("open firefox", None)])
def test_only_done_or_cancel_leave_private_mode(words, leave):
    assert leave_private(words) == leave


def test_private_mode_hides_every_utterance_until_done(tmp_path):
    controls = Controls(tmp_path / "voice.json")
    assert controls.handle("I'll type my password") == Reply(say=PRIVATE_LINE, stop=True)
    for words in ("hunter2", "my password is hunter2", "talk faster", "open firefox", "start dictating"):
        assert controls.shown(words) == "(private)"
        reply = controls.handle(words)
        assert reply.goal is None and "hunter2" not in (reply.say or "")
    assert controls.speed == 1.0 and not controls.dictating  # box commands wait too
    assert controls.shown("I'm done.") == "I'm done."
    assert controls.handle("I'm done.") == Reply(goal=RESUME_GOAL)
    assert controls.shown("open firefox") == "open firefox"
    assert controls.handle("open firefox") == Reply(goal="open firefox")


def test_cancel_leaves_private_mode_and_sends_nothing(tmp_path):
    controls = Controls(tmp_path / "voice.json")
    controls.handle("private mode")
    reply = controls.handle("cancel")
    assert reply.goal is None and reply.say
    assert not controls.private
    assert controls.handle("done") == Reply(goal="done")  # outside private mode "done" is an ordinary request


def test_the_voice_page_stops_the_agent_and_hides_the_screen_in_private_mode(tmp_path):
    agent = Agent("http://127.0.0.1:9", controls=Controls(tmp_path / "voice.json"))
    sent = []
    agent.post = lambda path, body: sent.append((path, body))
    lines: queue.Queue = queue.Queue()
    server = page_server(agent, 0, 0, lines)
    port = server.server_address[1]

    def segment(text):
        request = urllib.request.Request(f"http://127.0.0.1:{port}/segment", json.dumps({"text": text}).encode(),
                                         {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{port}"})
        urllib.request.urlopen(request, timeout=5).close()

    try:
        segment("I'll type my password")
        assert sent == [("/stop", {})]
        assert lines.get_nowait()["say"] == PRIVATE_LINE
        with patch("urllib.request.urlopen", side_effect=AssertionError("no frame is fetched while private")):
            assert agent.screen() is None
        segment("hunter2")
        assert sent == [("/stop", {})]
        segment("done")
        assert sent[-1] == ("/goal", {"text": RESUME_GOAL, "continues": False})
    finally:
        server.shutdown()


def test_the_relay_hands_the_page_no_private_words(tmp_path):
    controls = Controls(tmp_path / "voice.json")

    def result(words):
        return json.dumps({"type": "Results", "is_final": True, "channel": {"alternatives": [
            {"transcript": words, "words": [{"word": w} for w in words.split()]}]}})

    assert private_transcript(result("hunter2"), controls) == result("hunter2")  # not private: unchanged
    controls.handle("private mode")
    hidden = private_transcript(result("my password is hunter2"), controls)
    assert "hunter2" not in hidden
    assert json.loads(hidden)["channel"]["alternatives"][0]["transcript"] == ""
    assert private_transcript(result("I'm done."), controls) == result("I'm done.")
    metadata = json.dumps({"type": "Metadata"})
    assert private_transcript(metadata, controls) == metadata
