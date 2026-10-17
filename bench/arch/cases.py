"""Labeled spoken segments for the triage study.

kind:  act  = one visible pointer action satisfies it (act_ids are acceptable targets)
       plan = needs typing/keys/multiple steps/content/answers/confirmation/disambiguation
       none = remark or context; no action requested
ok:    other decisions that are also acceptable (e.g. a correction that may be planned)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Case:
    screen: str
    utterance: str
    kind: str
    act_ids: tuple[str, ...] = ()
    ok: frozenset[str] = field(default_factory=frozenset)


def act(screen: str, text: str, *ids: str, ok: tuple[str, ...] = ()) -> Case:
    return Case(screen, text, "act", ids, frozenset(ok))


def plan(screen: str, text: str, *alt_ids: str) -> Case:
    # alt_ids: acting on these is acceptable as a first step (not a false execute)
    return Case(screen, text, "plan", alt_ids, frozenset({"act"} if alt_ids else ()))


def none(screen: str, text: str) -> Case:
    return Case(screen, text, "none", (), frozenset({"plan"}))


CASES: list[Case] = [
    # desktop
    act("desktop", "open up chrome", "chrome"),
    act("desktop", "open blender", "blender"),
    act("desktop", "open that q3 report draft", "report_doc"),
    plan("desktop", "open up blender and make me a house", "blender"),
    plan("desktop", "empty the recycle bin"),
    none("desktop", "ok cool, thanks"),
    # inbox
    act("inbox", "click on janet's email that has info on the report", "janet"),
    none("inbox", "oh john is asking for a report"),
    act("inbox", "actually go back", "back", ok=("plan",)),
    plan("inbox", "draft up the report real quick and send it", "reply"),
    act("inbox", "open the email about benefits", "hr"),
    act("inbox", "go to my sent folder", "sent"),
    plan("inbox", "delete all of these"),
    plan("inbox", "no wait, the other one"),
    plan("inbox", "reply to john saying I'll have it by five", "reply"),
    # youtube
    act("youtube", "select the first video", "v_lofi"),
    act("youtube", "play the one about pasta", "v_pasta"),
    none("youtube", "hmm these all look kind of boring"),
    plan("youtube", "search for lofi hip hop", "search"),
    act("youtube", "the rocket launch one", "v_rocket"),
    # bios (keyboard-driven firmware)
    plan("bios", "go to the boot tab", "boot_tab"),
    plan("bios", "make the usb boot first then save and exit"),
    plan("bios", "what's my cpu temp"),
    # phone
    act("phone", "open whatsapp", "whatsapp"),
    act("phone", "open the camera", "camera"),
    plan("phone", "text mom that I'm running late", "messages"),
    none("phone", "ugh my battery is dying"),
    # blender
    plan("blender", "make me a house"),
    act("blender", "click the render menu", "render_menu"),
    act("blender", "switch to the scripting workspace", "scripting_ws"),
    plan("blender", "delete the cube", "cube", "outliner_cube"),
    none("blender", "nice, that looks great"),
]
