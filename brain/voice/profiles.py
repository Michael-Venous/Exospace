"""Named computer profiles (issue #29): what the user calls each computer the box plugs into, what it is, and a small
fingerprint of its screen, so the box knows which one it is on without a model call.

A profile has a name ("Work laptop"), an OS, a launcher hint ("CMD+SPACE"), shortcuts to prefer and keys to avoid
(free text lines), and a fingerprint: a 256-bit difference hash of the screen shrunk to 17x16 grey, with the capture
resolution. The name and fingerprint come from speech (brain.voice.setup, brain.voice.plug_in); the hints are typed
into the file by whoever helps set the box up. Profiles live in profiles.json next to the voice settings, with the
active one's name, so the agent's process reads the same file (`device`).

Profiles live with the voice layer because the voice front end names, matches and asks about them; the agent only
reads `device`'s text, as it read $EXO_DEVICE before. They are separate from the device profile of PLAN.md §5.4,
which is still detected from the first frames on every connect.

Telling computers apart by their USB enumeration, before any frame arrives, needs Pico firmware work (Hardware
role): not done here; see issue #29.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from brain.types import Frame
from brain.voice import settings

PATH = settings.PATH.with_name("profiles.json")
HASH_SIDE = 16  # the hash compares each of 16x16 cells with its right neighbour: 256 bits
SAME = 40  # bits apart: at most this, the same computer
NEAR = 80  # at most this, maybe that computer: the user is asked
MARGIN = 16  # a match must beat the next closest profile by this many bits, or it is ambiguous
# Bits added for a different capture resolution: past SAME, so the same picture at another size is asked about,
# never matched for sure (the answer stores the new size).
OTHER_RESOLUTION = SAME + 1


@dataclass(frozen=True)
class Profile:
    name: str
    os: str = "unknown"
    launcher: str = ""  # how the launcher opens, e.g. "CMD+SPACE"
    prefer: tuple[str, ...] = ()  # shortcuts to prefer, one per line
    avoid: tuple[str, ...] = ()  # keys never to press, one per line
    fingerprint: str = ""  # hex difference hash of the screen; empty until a frame is seen
    resolution: tuple[int, int] | None = None  # (width, height) of that frame


@dataclass(frozen=True)
class Store:
    profiles: tuple[Profile, ...] = ()
    active: str = ""  # the name of the computer plugged in now


@dataclass(frozen=True)
class Match:
    """Which profile a screen is: `kind` "match" (that `profile`), "ambiguous" (one of `candidates`) or "new"."""

    kind: Literal["match", "ambiguous", "new"]
    profile: Profile | None = None
    candidates: tuple[Profile, ...] = ()


def fingerprint(frame: Frame) -> str:
    """A difference hash: does each cell of the shrunk grey screen get brighter to its right? Small changes (a
    clock, a cursor, JPEG noise) flip few of its bits."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    small = cv2.resize(grey, (HASH_SIDE + 1, HASH_SIDE), interpolation=cv2.INTER_AREA).astype(np.int16)
    return np.packbits(small[:, 1:] > small[:, :-1]).tobytes().hex()


def from_jpeg(jpeg: bytes | None) -> Frame | None:
    """A frame from the agent's GET /screen.jpg; None when there is none."""
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) if jpeg else None


def resolution(frame: Frame) -> tuple[int, int]:
    return frame.shape[1], frame.shape[0]


def distance(a: str, b: str) -> int:
    """How many bits two fingerprints differ in."""
    return (int(a, 16) ^ int(b, 16)).bit_count()


def match(profiles: tuple[Profile, ...], frame: Frame) -> Match:
    """The profile this screen belongs to, without a model call. Profiles without a fingerprint never match."""
    seen, size = fingerprint(frame), resolution(frame)
    scored = sorted(((distance(seen, p.fingerprint) + (0 if p.resolution == size else OTHER_RESOLUTION), p)
                     for p in profiles if p.fingerprint), key=lambda pair: pair[0])
    near = tuple(p for score, p in scored if score <= NEAR)
    if not near:
        return Match("new")
    best = scored[0][0]
    if best <= SAME and (len(scored) == 1 or scored[1][0] - best >= MARGIN):
        return Match("match", profile=scored[0][1])
    return Match("ambiguous", candidates=near)


def find(store: Store, name: str) -> Profile | None:
    return next((p for p in store.profiles if p.name.lower() == name.lower()), None)


def remember(store: Store, name: str, frame: Frame | None) -> Store:
    """The store with `name` active, created if new, and its fingerprint taken from `frame` when there is one (no
    frame: the name only). Its hints are kept."""
    known = find(store, name)
    profile = known or Profile(name)
    if frame is not None:
        profile = replace(profile, fingerprint=fingerprint(frame), resolution=resolution(frame))
    kept = tuple(profile if p is known else p for p in store.profiles)
    return Store(profiles=kept if known else (*kept, profile), active=profile.name)


def device_text(profile: Profile) -> str:
    """The computer as the agent is told it, in the words $EXO_DEVICE uses."""
    kind = f", a {profile.os} computer" if profile.os != "unknown" else ""
    lines = [f"{profile.name}{kind}."]
    if profile.launcher:
        lines.append(f"{profile.launcher} opens the application launcher.")
    if profile.prefer:
        lines.append("Prefer these shortcuts: " + "; ".join(profile.prefer) + ".")
    if profile.avoid:
        lines.append("Never press: " + "; ".join(profile.avoid) + ".")
    return " ".join(lines)


def device(fallback: str, path: Path = PATH) -> str:
    """The device text for the agent: $EXO_DEVICE when set, else the active profile's when it says more than a name,
    else `fallback`."""
    if override := os.environ.get("EXO_DEVICE"):
        return override
    store = load(path)
    profile = find(store, store.active)
    hints = profile and (profile.os != "unknown" or profile.launcher or profile.prefer or profile.avoid)
    return device_text(profile) if profile and hints else fallback


def load(path: Path = PATH) -> Store:
    """The saved profiles; a missing or broken file is none."""
    try:
        saved = json.loads(path.read_text())
        profiles = tuple(Profile(name=str(p["name"]), os=str(p.get("os", "unknown")),
                                 launcher=str(p.get("launcher", "")),
                                 prefer=tuple(map(str, p.get("prefer", ()))), avoid=tuple(map(str, p.get("avoid", ()))),
                                 fingerprint=str(p.get("fingerprint", "")),
                                 resolution=(int(p["resolution"][0]), int(p["resolution"][1]))
                                 if p.get("resolution") else None)
                         for p in saved.get("profiles", ()))
        return Store(profiles=profiles, active=str(saved.get("active", "")))
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return Store()


def save(store: Store, path: Path = PATH) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(store), indent=2) + "\n")
    except OSError as exc:
        print(f"  (computer profiles not saved to {path}: {exc})", flush=True)
