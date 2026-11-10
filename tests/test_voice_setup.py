"""The spoken setup: short steps by voice, with repeat, go back, skip and stop; saved as it goes; never the agent's."""

import pytest

from brain.voice import settings
from brain.voice.commands import STEP, SetUp, parse
from brain.voice.controls import Controls, Reply
from brain.voice.setup import CHANGE_HINT, QUESTIONS, STOPPED, Step, computer_name, talk_level


@pytest.fixture
def path(tmp_path):
    return tmp_path / "voice.json"


def started(path):
    controls = Controls(path)
    assert controls.handle("Set up.") == Reply(say=QUESTIONS[Step.WELCOME])
    return controls


@pytest.mark.parametrize("words", ["Set up.", "setup", "Start setup.", "run set up", "Please run setup."])
def test_setup_commands(words):
    assert parse(words) == SetUp()


def test_setup_words_inside_a_request_go_to_the_agent():
    assert parse("set up a meeting with Janet") is None


def test_the_whole_setup_by_voice_saves_each_choice(path):
    controls = started(path)
    reply = controls.handle("hello?")
    assert reply.goal is None and reply.say.endswith(QUESTIONS[Step.SPEED])
    faster = controls.handle("Faster.")
    assert faster == Reply(say=f"Now I talk at {STEP:g} times speed. Faster, slower, or that's good?")
    assert controls.speed == STEP  # the front end speaks that line at the new speed
    assert settings.load(path).speed == STEP
    controls.handle("talk at 2x")
    assert controls.speed == 2.0
    assert controls.handle("That's good.").say == QUESTIONS[Step.NAME]
    assert controls.handle("Call it work laptop.").say == f"I'll call this computer Work laptop. {QUESTIONS[Step.TALK]}"
    done = controls.handle("Just results.")
    assert done.goal is None
    assert done.say == ("I'll tell you just the results. All set. I talk at 2 times speed, this computer is Work "
                        f"laptop, and I'll tell you just the results. {CHANGE_HINT}")
    assert settings.load(path) == settings.Settings(speed=2.0, computer="Work laptop", talk="results")
    assert controls.handle("open firefox") == Reply(goal="open firefox")  # setup is over


def test_repeat_go_back_and_skip(path):
    controls = started(path)
    assert controls.handle("Repeat that.").say == QUESTIONS[Step.WELCOME]
    assert controls.handle("go back").say == QUESTIONS[Step.WELCOME]  # nothing before the first step
    assert controls.handle("Skip.").say == QUESTIONS[Step.SPEED]
    assert controls.handle("make it purple").say == "Say faster, slower, or that's good."
    assert controls.handle("skip").say == QUESTIONS[Step.NAME]
    assert controls.handle("Go back.").say == QUESTIONS[Step.SPEED]
    controls.handle("slower")
    assert controls.handle("Say that again.").say == QUESTIONS[Step.SPEED]
    controls.handle("that's fine")
    controls.handle("Home PC")
    assert controls.handle("go back").say == QUESTIONS[Step.NAME]
    controls.handle("Kitchen computer")
    assert controls.handle("whatever").say == "Say every step, just results, or only when done."
    assert controls.handle("skip").say.endswith(CHANGE_HINT)
    assert settings.load(path) == settings.Settings(speed=round(1 / STEP, 2), computer="Kitchen computer")


def test_stop_setup_keeps_what_was_chosen(path):
    controls = started(path)
    controls.handle("ok")
    controls.handle("faster")
    assert controls.handle("Cancel setup.") == Reply(say=STOPPED)
    assert settings.load(path).speed == STEP
    assert controls.handle("talk faster").say  # box commands work again
    assert controls.handle("open firefox") == Reply(goal="open firefox")


def test_nothing_in_setup_reaches_the_agent(path):
    controls = started(path)
    for words in ["open firefox", "start dictating", "delete all my email", "Every step."]:
        assert controls.handle(words).goal is None
    assert controls.dictating is False


def test_a_first_run_starts_setup_by_itself_and_only_once(path):
    controls = Controls(path, first_run_setup=True)
    assert controls.greeting() == QUESTIONS[Step.WELCOME]
    assert controls.handle("open firefox").goal is None  # the welcome's "say anything"
    controls.handle("stop setup")
    assert path.exists()  # even stopped at once, setup doesn't start again by itself
    again = Controls(path, first_run_setup=True)
    assert again.greeting() is None
    assert again.handle("open firefox") == Reply(goal="open firefox")


def test_first_run_setup_is_off_unless_asked(path):
    controls = Controls(path)
    assert controls.greeting() is None
    assert controls.handle("open firefox") == Reply(goal="open firefox")


def test_set_up_stops_dictation(path):
    controls = Controls(path)
    controls.handle("start dictating")
    controls.handle("set up")
    assert controls.dictating is False


@pytest.mark.parametrize("words, name", [("Work laptop.", "Work laptop"), ("call it home PC", "Home PC"),
                                         ("  It's the kitchen one!  ", "The kitchen one"), ("...", "")])
def test_computer_names_are_tidied(words, name):
    assert computer_name(words) == name


@pytest.mark.parametrize("text, level", [("every step", "steps"), ("just results", "results"),
                                         ("only when done", "done"), ("only done", "done"), ("purple", None)])
def test_talk_levels(text, level):
    assert talk_level(text) == level


def test_settings_keep_the_new_fields_and_drop_bad_ones(path):
    settings.save(settings.Settings(computer="Work laptop", talk="done"), path)
    assert settings.load(path) == settings.Settings(computer="Work laptop", talk="done")
    path.write_text('{"speed": 1, "computer": 3, "talk": "loud"}')
    assert settings.load(path) == settings.Settings()
