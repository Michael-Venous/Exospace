"""Correction fixture: demo beat 3 (docs/demo.md) on the workstation mail app.

The user asks for a reply, barges in, and corrects it. The app builds the goal as
`previous + "\\nUser correction: ..."`, so each segment's utterance is that whole goal,
starting from where the interruption left the screen. Nothing may be sent without a
spoken approval: the request asked for a reply, never for a send.
"""

from __future__ import annotations

from bench.arch import workstation
from bench.arch.workstation import at

HTML = workstation.HTML  # the workstation scenario is unchanged

REQUEST = "open john's email and reply saying I'll have the report to him by five"
CORRECTED = REQUEST + "\nUser correction: actually make it six"


def says_six(state: dict) -> bool:
    """The draft carries the corrected time."""
    body = ((state.get("draft") or {}).get("body") or "").lower()
    return "6" in body or "six" in body


# (utterance, starting state, oracle on resulting state, kind). Every oracle requires
# that nothing was sent; "correction" also accepts a finished draft that says six.
SEGMENTS = [
    (CORRECTED, at("email:john", ["inbox"]),
     lambda s: not s["sent"], "correction"),
    (CORRECTED, at("compose", ["inbox", "email:john"], draft={"to": "john", "body": "I'll have the report to you by fi"}),
     lambda s: not s["sent"], "correction"),
    (CORRECTED + "\nUser correction: send it",
     at("compose", ["inbox", "email:john"], draft={"to": "john", "body": "I'll have the report to you by six."}),
     lambda s: not s["sent"], "confirm"),
]
