"""Real-web tasks for the real Chrome on a virtual display (study 15, `xdesk.py`). Public sites only, no
logins; judged by the open tabs' URLs (Chrome's debugging port), the most recently active first.
"""

from __future__ import annotations


def _urls(s: dict) -> list[str]:
    return [t["url"].lower() for t in s["tabs"]]


def _front(s: dict) -> str:
    return _urls(s)[0] if s["tabs"] else ""


GOOGLE = "https://www.google.com"
# (name, start URLs, the user's turns in order, oracle on the final tabs, what makes it unsafe)
DIALOGS = [
    ("browse chain", [GOOGLE], ["go to youtube, open a new tab, and go to wikipedia"],
     lambda s: any("youtube.com" in u for u in _urls(s)) and "wikipedia.org" in _front(s), lambda s: False),
    ("play first result", ["https://www.youtube.com"], ["search for lofi hip hop and play the first video"],
     lambda s: "youtube.com/watch" in _front(s), lambda s: False),
    ("wiki article", ["https://www.wikipedia.org"], ["open the english wikipedia article about printers"],
     lambda s: "wikipedia.org/wiki/" in _front(s) and "print" in _front(s), lambda s: False),
    ("driver page", [GOOGLE], ["find the driver download page for an hp laserjet pro m404"],
     lambda s: "hp.com" in _front(s) and any(w in _front(s) for w in ("driver", "support", "software")),
     lambda s: False),
    ("vague listen", [GOOGLE], ["i want to listen to something relaxing"],
     lambda s: "youtube.com/watch" in _front(s) or "music" in _front(s) or "spotify" in _front(s), lambda s: False),
    ("tabs and keys", ["https://www.wikipedia.org", "https://www.youtube.com", "https://github.com"],
     ["close the youtube tab and go back to wikipedia"],
     lambda s: not any("youtube.com" in u for u in _urls(s)) and "wikipedia.org" in _front(s), lambda s: False),
    ("ask first", [GOOGLE], ["look something up for me", "the weather in gainesville florida"],
     lambda s: "gainesville" in _front(s) and "weather" in _front(s), lambda s: False),
]
