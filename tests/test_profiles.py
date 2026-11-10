"""Computer profiles: a screen fingerprint that survives small changes, matching without a model call, the spoken
"which computer is this?", and the device text the agent is told."""

import cv2
import numpy as np
import pytest

from brain.voice import profiles, settings
from brain.voice.controls import Controls, Reply
from brain.voice.profiles import Match, Profile, Store

KEY = "Never press: CMD+Q."


def screen(seed: int, size: tuple[int, int] = (1280, 800)) -> np.ndarray:
    """A made-up desktop: blurred blocks of colour, different for each seed."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (6, 10, 3), dtype=np.uint8)
    return cv2.GaussianBlur(cv2.resize(small, size, interpolation=cv2.INTER_NEAREST), (0, 0), 40)


def touched(frame: np.ndarray) -> np.ndarray:
    """The same screen a moment later: a new clock, a moved pointer, JPEG noise."""
    frame = frame.copy()
    cv2.putText(frame, "10:42", (1180, 790), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(frame, (600, 400), (612, 420), (0, 0, 0), -1)
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return profiles.from_jpeg(jpeg.tobytes())


def known(name: str, frame: np.ndarray, **hints) -> Profile:
    return Profile(name, fingerprint=profiles.fingerprint(frame), resolution=profiles.resolution(frame), **hints)


def test_the_fingerprint_survives_small_changes_and_tells_screens_apart():
    a = screen(1)
    assert len(profiles.fingerprint(a)) == 64  # 256 bits
    assert profiles.distance(profiles.fingerprint(a), profiles.fingerprint(touched(a))) <= profiles.SAME // 4
    assert profiles.distance(profiles.fingerprint(a), profiles.fingerprint(screen(2))) > profiles.NEAR


def test_matching_finds_the_computer_asks_when_unsure_and_knows_a_new_one():
    work, home = screen(1), screen(2)
    stored = (known("Work laptop", work), known("Home PC", home), Profile("Unseen"))
    found = profiles.match(stored, touched(work))
    assert found.kind == "match" and found.profile.name == "Work laptop"
    assert profiles.match(stored, screen(3)) == Match("new")
    twin = known("Work laptop two", touched(work))  # two computers that look alike: ask
    unsure = profiles.match((*stored, twin), work)
    assert unsure.kind == "ambiguous"
    assert {p.name for p in unsure.candidates} == {"Work laptop", "Work laptop two"}


def test_another_resolution_counts_against_a_match():
    work = screen(1)
    found = profiles.match((known("Work laptop", work),), cv2.resize(work, (1920, 1200)))
    assert found.kind == "ambiguous"


def test_the_store_round_trips_and_survives_a_broken_file(tmp_path):
    path = tmp_path / "profiles.json"
    assert profiles.load(path) == Store()
    store = profiles.remember(Store((Profile("Work laptop", os="macos", launcher="CMD+SPACE"),)), "work laptop",
                              screen(1))
    profiles.save(store, path)
    loaded = profiles.load(path)
    assert loaded == store and loaded.active == "Work laptop"
    assert loaded.profiles[0].launcher == "CMD+SPACE" and loaded.profiles[0].resolution == (1280, 800)
    path.write_text("[1, 2]")
    assert profiles.load(path) == Store()


def test_a_profile_renders_into_the_agents_device_text():
    profile = Profile("Work laptop", os="macos", launcher="CMD+SPACE", prefer=("CMD+L focuses the address bar",),
                      avoid=("CMD+Q",))
    assert profiles.device_text(profile) == ("Work laptop, a macos computer. CMD+SPACE opens the application "
                                             "launcher. Prefer these shortcuts: CMD+L focuses the address bar. "
                                             f"{KEY}")


def test_exo_device_overrides_the_active_profile(tmp_path, monkeypatch):
    path = tmp_path / "profiles.json"
    monkeypatch.delenv("EXO_DEVICE", raising=False)
    assert profiles.device("KDE laptop.", path) == "KDE laptop."  # no profiles
    profiles.save(Store((Profile("Home PC"),), active="Home PC"), path)
    assert profiles.device("KDE laptop.", path) == "KDE laptop."  # a name alone tells the agent nothing
    profiles.save(Store((Profile("Home PC", os="windows", avoid=("CMD+Q",)),), active="Home PC"), path)
    assert profiles.device("KDE laptop.", path) == f"Home PC, a windows computer. {KEY}"
    monkeypatch.setenv("EXO_DEVICE", "Chromebook. The search key opens the launcher.")
    assert profiles.device("KDE laptop.", path) == "Chromebook. The search key opens the launcher."


class Rig:
    """Controls over a settings folder, with a screen the test can change."""

    def __init__(self, folder, frame=None):
        self.frame = frame
        self.path = folder / "voice.json"
        self.store = folder / "profiles.json"

    def controls(self, **options) -> Controls:
        return Controls(self.path, frame=lambda: self.frame, **options)

    def saved(self) -> Store:
        return profiles.load(self.store)


def test_a_known_computer_is_chosen_quietly(tmp_path):
    rig = Rig(tmp_path, screen(1))
    profiles.save(Store((known("Work laptop", screen(1)), known("Home PC", screen(2)))), rig.store)
    rig.frame = touched(screen(1))
    controls = rig.controls()
    assert controls.greeting() is None
    assert rig.saved().active == "Work laptop" and controls.settings.computer == "Work laptop"
    assert controls.handle("open firefox") == Reply(goal="open firefox")


def test_a_new_computer_is_asked_about_and_named(tmp_path):
    rig = Rig(tmp_path, screen(3))
    profiles.save(Store((known("Work laptop", screen(1)),)), rig.store)
    controls = rig.controls()
    assert controls.greeting() == "Is this a new computer?"
    assert controls.handle("open firefox") == Reply(say="Say yes or no. Is this a new computer?")
    assert controls.handle("Repeat that.") == Reply(say="Is this a new computer?")
    assert controls.handle("Yes.") == Reply(say="What should I call it?")
    assert controls.handle("Kitchen computer.") == Reply(say="I'll call this computer Kitchen computer.")
    kitchen = profiles.find(rig.saved(), "Kitchen computer")
    assert rig.saved().active == "Kitchen computer" and kitchen.fingerprint == profiles.fingerprint(screen(3))
    assert settings.load(rig.path).computer == "Kitchen computer"
    assert controls.handle("open firefox") == Reply(goal="open firefox")


def test_an_unsure_screen_asks_which_computer(tmp_path):
    rig = Rig(tmp_path, screen(1))
    profiles.save(Store((known("Work laptop", screen(1)), known("Home PC", touched(screen(1))))), rig.store)
    controls = rig.controls()
    assert controls.greeting() == "Is this Work laptop or Home PC?"
    assert controls.handle("the blue one") == Reply(say="Say Work laptop, Home PC or new.")
    assert controls.handle("It's the home PC.") == Reply(say="OK, this is Home PC.")
    assert rig.saved().active == "Home PC"
    assert profiles.find(rig.saved(), "Home PC").fingerprint == profiles.fingerprint(screen(1))  # refreshed


def test_not_new_asks_which_and_never_mind_changes_nothing(tmp_path):
    rig = Rig(tmp_path, screen(3))
    before = Store((known("Work laptop", screen(1)), known("Home PC", screen(2))), active="Home PC")
    profiles.save(before, rig.store)
    controls = rig.controls()
    controls.greeting()
    assert controls.handle("No.") == Reply(say="Is this Work laptop or Home PC?")
    assert controls.handle("Never mind.") == Reply(say="OK, I'll leave it.")
    assert rig.saved() == before
    assert controls.handle("open firefox") == Reply(goal="open firefox")


def test_neither_names_a_new_one(tmp_path):
    rig = Rig(tmp_path, screen(1))
    profiles.save(Store((known("Work laptop", screen(1)), known("Home PC", touched(screen(1))))), rig.store)
    controls = rig.controls()
    controls.greeting()
    assert controls.handle("Neither.").say == "Then it's a new one. What should I call it?"
    controls.handle("Studio Mac")
    assert [p.name for p in rig.saved().profiles] == ["Work laptop", "Home PC", "Studio Mac"]


@pytest.mark.parametrize("frame", [None, screen(4)])
def test_setup_names_the_computers_profile_with_or_without_a_frame(tmp_path, frame):
    rig = Rig(tmp_path, frame)
    controls = rig.controls(first_run_setup=True)
    for words in ["hi", "that's good", "Work laptop"]:
        controls.handle(words)
    profile = profiles.find(rig.saved(), "Work laptop")
    assert rig.saved().active == "Work laptop"
    assert profile.fingerprint == ("" if frame is None else profiles.fingerprint(frame))


def test_no_profiles_or_no_frame_asks_nothing(tmp_path):
    assert Rig(tmp_path, screen(1)).controls().greeting() is None  # no profiles yet
    rig = Rig(tmp_path)
    profiles.save(Store((known("Work laptop", screen(1)),)), rig.store)
    assert rig.controls().greeting() is None
