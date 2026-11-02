"""Study 18: how big the planner's screenshot must be on the laptop's 2880x1800 screen (bench only).

The earlier queue sent DeepSeek V4.1 Flash a 1280 px screenshot, the optimized one 1920 px and now "fit" (its
`planner_jpeg`: the hosts' 1.6 MP budget, one resize, one encode). At 150% UI scale
small labels are 7-9 px tall at 1280. Each case is a desktop screen rendered at exactly 2880x1800 (1920x1200 CSS px at
scale 1.5, as on the laptop) with a known answer or click target, plus real laptop frames. The planner gets the
optimized queue's prompt and context with the frame shrunk exactly as `Brain._call` does, at each side; a reply is
correct, wrong (names a different item, a misread value) or other (acts instead: zooms, asks, waits). Click targets
are also given to the Qwen pointer at each side, scored against the element's box.

  python -m bench.arch.see_side_study --out work/arch/see-side [--reps 3] [--sides 1280 1920 fit] [--point-sides]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import httpx
import numpy as np

from bench.arch.openrouter_study import OpenRouterJSON
from bench.arch.optimized_queue import PROMPT, planner_jpeg
from bench.arch.pointer import PROMPT as POINT_PROMPT, SCHEMA as POINT_SCHEMA
from bench.arch.queue_agent import DEVICE, OPENROUTER, PINNED
from brain.imaging import shrink

PLANNER = "deepseek/deepseek-v4.1-flash-1017"
POINTER = "qwen/qwen3.8-27b"
SIDES = ("1280", "1600", "1920", "2400", "2880", "fit")  # planner images: a long side, "fit", or "tiles"
POINT_SIDES = ("1280", "1920", "2880")  # pointer images: a long side, "mp<megapixels>", or "native"
SCREEN = {"width": 2880, "height": 1800, "scale": 1.5}  # the screen rendered; --screen and --scale change it
TILES_NOTE = ("The first image is the whole screen. The next four are its top-left, top-right, bottom-left and "
              "bottom-right quarters in more detail: read small text from them.")
# "tiles2": the first wording made replies treat a quarter as the screen ("the window is no longer visible").
TILES_NOTE_2 = ("Image 1 is the screen: plan from it, and judge what is open, where and what is visible from it alone. "
                "Images 2-5 are magnified copies of its top-left, top-right, bottom-left and bottom-right quarters, "
                "only for reading small text exactly; a window cut off at a quarter's edge is whole on the screen. "
                "Name click targets as they appear on the screen.")
REAL = Path(__file__).resolve().parents[2] / "work/arch"

CSS = """
*{box-sizing:border-box} body{margin:0;width:1920px;height:1200px;font:13px 'Noto Sans',sans-serif;color:#232629;
background:linear-gradient(135deg,#0b2742,#1b5d8c 60%,#0e3350);overflow:hidden}
.win{position:absolute;background:#eff0f1;border:1px solid #9a9da0;box-shadow:0 4px 18px #0008}
.title{height:30px;line-height:30px;text-align:center;background:#dee0e2;border-bottom:1px solid #c4c7ca;font-size:13px}
.title .b{position:absolute;right:10px;top:0;letter-spacing:12px;color:#555}
.panel{position:absolute;left:0;right:0;bottom:0;height:44px;background:#202326ee;color:#eee;display:flex;
align-items:center;padding:0 14px;gap:18px;font-size:12px}
.panel .tray{margin-left:auto;display:flex;gap:16px;align-items:center}
table{border-collapse:collapse;width:100%} td,th{padding:4px 8px;text-align:left;white-space:nowrap}
th{font-weight:normal;color:#555;border-bottom:1px solid #c4c7ca;font-size:12px} tr:nth-child(even) td{background:#f7f7f8}
.side{position:absolute;left:0;top:30px;bottom:0;width:190px;background:#e3e5e7;padding:8px;line-height:26px}
.mono{font:12px 'Noto Sans Mono',monospace;white-space:pre;color:#d3dae3;background:#1e2229;padding:10px;line-height:17px}
"""

PANEL = """<div class=panel><span>☰</span><span>Firefox</span><span>Dolphin</span><span>Konsole</span>
<div class=tray><span>🔊</span><span>📶</span><span id=battery>🔋 63%</span><span>Sun 1 Nov</span><span>14:38</span></div></div>"""


def window(x: int, y: int, w: int, h: int, title: str, body: str) -> str:
    return (f'<div class=win style="left:{x}px;top:{y}px;width:{w}px;height:{h}px"><div class=title>{title}'
            f'<span class=b>–□×</span></div><div style="position:absolute;top:31px;left:0;right:0;bottom:0">{body}'
            "</div></div>")


def files() -> str:
    rows = [("budget-draft-v2.ods", "48.2 KiB", "18/10/2026 16:02"), ("budget-draft-v3.ods", "51.7 KiB", "13/10/2026 09:41"),
            ("invoice-2026-01-17.pdf", "1.33 MiB", "17/01/2026 11:20"), ("invoice-2026-10-11.pdf", "1.38 MiB", "11/10/2026 14:05"),
            ("invoice-2026-10-12.pdf", "3.18 MiB", "12/10/2026 08:57"), ("invoice-2026-10-17.pdf", "1.83 MiB", "17/10/2026 10:12"),
            ("invoice-2026-11-07.pdf", "1.88 MiB", "01/11/2026 07:31"), ("notes-meeting-oct.txt", "2.1 KiB", "21/10/2026 15:44"),
            ("passport-scan.png", "4.7 MiB", "02/09/2026 19:03"), ("receipt-2026-10-17.pdf", "212.4 KiB", "17/10/2026 10:15"),
            ("report-q3-final.pdf", "2.6 MiB", "29/10/2026 17:22"), ("report-q3-final(1).pdf", "2.6 MiB", "29/10/2026 17:24"),
            ("tax-2025.zip", "18.9 MiB", "14/04/2026 12:00"), ("wallpaper-dunes.jpg", "6.3 MiB", "30/07/2026 21:18")]
    body = "".join(f'<tr id="row-{n}"><td><span id="name-{n}">📄 {n}</span></td><td>{s}</td><td>{m}</td></tr>'
                   for n, s, m in rows)
    side = "<div class=side>Places<br>🏠 Home<br>🖥 Desktop<br>📁 Documents<br>⬇ Downloads<br>🎵 Music<br>🖼 Pictures<br>🗑 Trash</div>"
    table = (f'<div style="position:absolute;left:190px;top:30px;right:0;bottom:0;background:#fff">'
             f"<table><tr><th>Name</th><th>Size</th><th>Modified</th></tr>{body}</table></div>")
    toolbar = '<div style="height:30px;padding:5px 200px;color:#555">◀ ▶  Home › Documents › finance</div>'
    return window(260, 90, 1300, 900, "finance — Dolphin", toolbar + side + table) + PANEL


def terminal() -> str:
    log = """mike@swift ~/code/web $ npm run dev

> web@0.4.2 dev
> vite --host

  VITE v6.2.0  ready in 412 ms

  ➜  Local:   http://localhost:41261/
  ➜  Network: http://10.0.0.188:41261/
  ➜  press h + enter to show help
[watch 41267] 14:37:52 page reload src/App.tsx
[watch 41267] 14:38:03 hmr update /src/routes/settings.tsx"""
    return window(380, 160, 1100, 560, "web : npm — Konsole", f'<div class=mono style="height:100%">{log}</div>') + PANEL


def panel() -> str:
    page = '<div style="padding:40px;font-size:15px;background:#fff;height:100%">Weather — Miami<br><br>Partly cloudy, 29°C</div>'
    return window(300, 120, 1300, 800, "Weather — Mozilla Firefox", page) + PANEL


def mail() -> str:
    rows = [("Re: lunch friday?", "Daniel Ortiz", "14:02"), ("Invoice dispute — order 88231", "Priya Raman", "13:47"),
            ("Your October statement", "Chase", "12:10"), ("Re: Re: thesis draft comments", "Priya Rao", "11:55"),
            ("Team offsite logistics", "Pia Raman", "10:21"), ("Build failed: main #4128", "GitHub", "09:58"),
            ("Shipping update for order 88213", "Amazon", "09:30"), ("Weekly digest", "Hacker Newsletter", "08:00"),
            ("Re: invoice copy", "Daniel Ortiz", "Yesterday"), ("Dentist reminder", "Smile Dental", "Yesterday"),
            ("Meeting notes 10/30", "Priya Rao", "Yesterday"), ("Password changed", "Google", "Fri")]
    body = "".join(f'<tr id="mail-{i}"><td>{s}</td><td>{f}</td><td>{d}</td></tr>' for i, (s, f, d) in enumerate(rows))
    side = "<div class=side>📥 Inbox (12)<br>📤 Sent<br>📝 Drafts<br>🗄 Archive<br>⚠ Spam</div>"
    table = (f'<div style="position:absolute;left:190px;top:30px;right:0;height:430px;background:#fff">'
             f"<table><tr><th>Subject</th><th>From</th><th>Date</th></tr>{body}</table></div>")
    return window(180, 70, 1500, 960, "Inbox — Thunderbird", side + table) + PANEL


def settings() -> str:
    form = """<div style="position:absolute;left:190px;top:30px;right:0;bottom:0;background:#fff;padding:30px 60px;line-height:38px">
<b style="font-size:15px">Display Configuration</b><br>
Device: <b>Built-in Screen</b> — Samsung SDC 4171<br>
Resolution: <select><option>2880×1800 (16:10)</option></select><br>
Scale: <select><option>150%</option></select><br>
Refresh rate: <select><option>119.93 Hz</option></select><br>
Adaptive sync: <select><option>Automatic</option></select><br>
Orientation: Landscape</div>"""
    side = "<div class=side>Appearance<br>Workspace<br>Notifications<br><b>Display &amp; Monitor</b><br>Power<br>Bluetooth</div>"
    return window(260, 120, 1300, 800, "Display Configuration — System Settings", side + form) + PANEL


def notifications() -> str:
    apps = ["Discord", "Discover", "Dolphin", "Dragon Player", "Elisa", "Firefox", "KDE Connect", "Konsole", "Spectacle"]
    rows = "".join(f'<tr><td>{a}</td>' + "".join(f'<td><input type=checkbox checked id="{a}-{c}"></td>'
                                                 for c in ("popup", "sound", "history")) + "</tr>" for a in apps)
    form = (f'<div style="position:absolute;left:190px;top:30px;right:0;bottom:0;background:#fff;padding:20px 40px">'
            '<b style="font-size:15px">Application-specific settings</b><br><br>'
            f'<table style="width:520px"><tr><th>Application</th><th>Popup</th><th>Sound</th><th>History</th></tr>{rows}'
            "</table></div>")
    side = "<div class=side>Appearance<br>Workspace<br><b>Notifications</b><br>Display &amp; Monitor<br>Power</div>"
    return window(260, 120, 1300, 800, "Notifications — System Settings", side + form) + PANEL


def tabs() -> str:
    names = ["Gmail - Inbox (3)", "Billing and Cost Management", "Cost Explorer | AWS", "Grafana – Overview",
             "PR #4182 · exospace/brain", "PR #4128 · exospace/brain", "Issue #4128 · exospace/web", "YouTube",
             "Calendar - Week of Nov 1", "ChatGPT", "Kagi Search: uinput hid", "Docs – Exospace plan"]
    strip = "".join(f'<div id="tab-{i}" style="width:155px;padding:7px 8px;border-right:1px solid #bbb;overflow:hidden;'
                    f'text-overflow:ellipsis;white-space:nowrap;font-size:12px;{"background:#fff" if i == 3 else ""}">{n}</div>'
                    for i, n in enumerate(names))
    page = ('<div style="display:flex;background:#dfe1e5">' + strip + "</div>"
            '<div style="padding:8px 12px;background:#fff;border-bottom:1px solid #ccc">🔒 grafana.local/d/overview</div>'
            '<div style="padding:40px;background:#111;color:#ddd;height:100%;font-size:20px">CPU 23%   Memory 61%</div>')
    return '<div class=win style="left:0;top:0;width:1920px;height:1156px">' + page + "</div>" + PANEL


def editor() -> str:
    text = '<div class=mono style="height:100%;background:#fff;color:#222">- call Sam at noon\n- buy filters\n- email Priya the invoice</div>'
    return window(300, 140, 1200, 700, "notes.txt * — Kate", text) + PANEL


def toast() -> str:
    note = ('<div class=win style="left:1500px;top:1010px;width:400px;height:80px;background:#2a2e32;color:#eee;'
            'padding:10px 14px;font-size:12px"><b>Backup</b><br>Backup finished: 3 files skipped (permission denied)</div>')
    return panel().replace(PANEL, "") + note + PANEL


def sheet() -> str:
    rng = random.Random(7)
    head = "<tr><th></th>" + "".join(f"<th>{c}</th>" for c in "ABCDEFGH") + "</tr>"
    labels = ["Region", "Q1", "Q2", "Q3", "Q4", "Growth", "Owner", "Notes"]
    rows = [f"<tr><th>1</th>" + "".join(f"<td><b>{v}</b></td>" for v in labels) + "</tr>"]
    for r in range(2, 24):
        vals = [f"Region {r - 1}"] + [f"{rng.uniform(800, 3200):,.2f}" for _ in range(4)] + \
               [f"{rng.uniform(-9, 14):.1f}%", rng.choice(["Ana", "Ben", "Chi", "Dev"]), ""]
        rows.append(f"<tr><th>{r}</th>" + "".join(f"<td>{v}</td>" for v in vals) + "</tr>")
    total = ["Total", "45,902.11", "47,180.64", "48,317.20", "48,371.20", "", "", ""]
    rows.append("<tr><th>24</th>" + "".join(f"<td><b>{v}</b></td>" for v in total) + "</tr>")
    grid = f'<div style="background:#fff;height:100%;font-size:12px"><table>{head}{"".join(rows)}</table></div>'
    return window(120, 60, 1680, 1000, "sales-2026.ods — LibreOffice Calc", grid) + PANEL


@dataclass
class Case:
    name: str
    screen: str  # a fixture function's name, or a real frame's file name under work/arch
    request: str
    kind: str  # "answer": a say must match `expect`; "click": the first click's target must match `expect`
    expect: str  # regex for a correct reply
    wrong: str = ""  # regex that marks a reply wrong even if `expect` also matches (a near-identical item)
    point: str = ""  # click cases: the pointer's target, and the element id whose box is the hit area
    box: str = ""


CASES = [
    Case("files-size", "files", "How big is invoice-2026-10-17.pdf?", "answer", r"1[.,]83\s*Mi?B", r"1[.,](38|88|33)"),
    Case("files-modified", "files", "When did I last change budget-draft-v3.ods?", "answer", r"13|9:41|09:41", r"18/10|16:02"),
    Case("files-open", "files", "Open invoice-2026-10-17.pdf", "click", r"2026-10-17", r"10-11|10-12|11-07|01-17|receipt",
         "the invoice-2026-10-17.pdf file", "name-invoice-2026-10-17.pdf"),
    Case("terminal-port", "terminal", "What port is the dev server on?", "answer", r"41261", r"41267"),
    Case("panel-battery", "panel", "What's my battery at?", "answer", r"\b63\s*%", r"\b(68|83|65|53)\s*%"),
    Case("mail-sender", "mail", "Who sent the email about the invoice dispute?", "answer", r"Priya Raman", r"Rao|Pia"),
    Case("mail-open", "mail", "Open the meeting notes Priya Rao sent", "click", r"Meeting notes|10/30", r"thesis|Raman",
         "the email 'Meeting notes 10/30' from Priya Rao", "mail-10"),
    Case("settings-refresh", "settings", "What refresh rate is my screen running at?", "answer", r"119[.,]93", ""),
    Case("notif-sound", "notifications", "Turn off notification sounds for Discover", "click", r"(?=.*Discover)(?=.*[Ss]ound)",
         r"Discord", "the Sound checkbox in the Discover row", "Discover-sound"),
    Case("tabs-pr", "tabs", "Switch to the tab with pull request 4128", "click", r"PR #?4128|pull request 4128",
         r"4182|Issue", "the browser tab 'PR #4128 · exospace/brain'", "tab-5"),
    Case("editor-saved", "editor", "Are my notes saved?", "answer",
         r"\b(not|unsaved|isn't|hasn't|haven't|aren't|modified|asterisk|\*)",
         r"no unsaved|(appear|are|is|been) (to be )?saved\b(?! yet)|^Yes"),
    Case("toast", "toast", "What did that notification say?", "answer", r"\b3\b[^.]*skip|three[^.]*skip", r"\b(2|5|8)\b[^.]*skip"),
    Case("sheet-total", "sheet", "What's the total for Q3?", "answer", r"48,?317\.20", r"48,?371|47,?180"),
    Case("panel-battery-click", "panel", "Open the battery settings", "click", r"batter|63", "",
         "the battery indicator in the panel", "battery"),
    Case("real-size", "pdf-final-0-after.jpg", "How big is the file that was modified just now?", "answer", r"\b19\s*B\b",
         r"193"),
    Case("real-folder", "pdf-final-0-after.jpg", "Which folder is that PDF in?", "answer", r"newest-document", ""),
]


def render(out: Path) -> dict[str, dict]:
    """Each fixture as a PNG of SCREEN, with the boxes (0-1 of the screen) of the elements the click cases name. The
    fixtures are laid out for at least 1920x1200 CSS px: on a bigger desktop (a 4K screen at 100%) the windows keep
    their size and the panel spans the screen, as a real desktop would."""
    from playwright.sync_api import sync_playwright
    boxes: dict[str, dict] = {}
    ids = {c.screen: [] for c in CASES}
    for c in CASES:
        if c.box:
            ids[c.screen].append(c.box)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("CHROME", "/usr/bin/google-chrome-stable"))
        vw, vh = round(SCREEN["width"] / SCREEN["scale"]), round(SCREEN["height"] / SCREEN["scale"])
        page = browser.new_page(viewport={"width": vw, "height": vh}, device_scale_factor=SCREEN["scale"])
        for screen, wanted in ids.items():
            if screen.endswith(".jpg"):
                continue
            page.set_content(f"<style>{CSS.replace('width:1920px;height:1200px', f'width:{vw}px;height:{vh}px')}</style>"
                             f"{globals()[screen]()}")
            page.screenshot(path=str(out / f"{screen}.png"))
            for element in wanted:
                r = page.eval_on_selector(f'[id="{element}"]', "e=>{const r=e.getBoundingClientRect();"
                                                               "return [r.left,r.top,r.right,r.bottom]}")
                boxes[element] = [r[0] / vw, r[1] / vh, r[2] / vw, r[3] / vh]
        browser.close()
    return boxes


def frame(out: Path, screen: str) -> np.ndarray:
    path = REAL / screen if screen.endswith(".jpg") else out / f"{screen}.png"
    image = cv2.imread(str(path))
    assert image.shape[:2] == (SCREEN["height"], SCREEN["width"]), (screen, image.shape)
    return image


def budget(image: np.ndarray, megapixels: float, quality: int = 90) -> bytes:
    """`image` resized once to fit `megapixels` (never enlarged), encoded once."""
    h, w = image.shape[:2]
    scale = min(1.0, (megapixels * 1e6 / (w * h)) ** .5)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def images(image: np.ndarray, side: str) -> list[bytes]:
    """The planner's images: `side` "fit" is the optimized queue's own encoding (the hosts' pixel budget, one resize
    and encode); "tiles" adds the four quarters, each fitted to the same budget; a number is a long side as
    `Brain._call` shrank it before."""
    if side == "fit":
        return [planner_jpeg(image)]
    if side in ("tiles", "tiles2"):
        h, w = image.shape[:2]
        quarters = [image[y:y + h // 2, x:x + w // 2] for y in (0, h // 2) for x in (0, w // 2)]
        return [planner_jpeg(image), *(planner_jpeg(np.ascontiguousarray(q)) for q in quarters)]
    return [shrink(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes(), int(side))]


def plan(client: httpx.Client, image: np.ndarray, side: str, request: str, prompt: str = PROMPT) -> dict:
    """One planner call exactly as `Brain._call` makes it on a fresh request (no log yet), on its pinned host."""
    pictures = images(image, side)
    context = {"device": DEVICE, "user_is_speaking": False, "user_said": request, "conversation_before": [],
               "log": [], "done_this_request": []}
    if side in ("tiles", "tiles2"):
        context["images"] = TILES_NOTE if side == "tiles" else TILES_NOTE_2
    body = {"model": PLANNER, "stream": True, "temperature": 0, "max_tokens": 2048, "reasoning": {"enabled": False},
            "provider": {"order": list(PINNED[PLANNER]), "allow_fallbacks": False}, "usage": {"include": True},
            "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": [
                *({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}
                  for jpeg in pictures),
                {"type": "text", "text": json.dumps(context)}]}]}
    started, first, text, usage, host = time.monotonic(), None, "", {}, ""
    with client.stream("POST", OPENROUTER, json=body) as response:
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}: {response.read()[:160]!r}"}
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage, host = chunk.get("usage") or usage, chunk.get("provider") or host
            for choice in chunk.get("choices", []):
                piece = (choice.get("delta") or {}).get("content") or ""
                if piece and first is None:
                    first = time.monotonic() - started
                text += piece
    return {"reply": text, "ttft_s": round(first or 0, 3), "total_s": round(time.monotonic() - started, 3),
            "prompt_tokens": usage.get("prompt_tokens"), "cost": usage.get("cost"), "host": host,
            "image_bytes": sum(map(len, pictures))}


def actions(reply: str) -> list[dict]:
    found, decoder = [], json.JSONDecoder()
    for line in reply.splitlines():
        line, at = line.strip().strip("`").strip(), 0
        while line.startswith("{", at):
            try:
                action, at = decoder.raw_decode(line, at)
            except json.JSONDecodeError:
                break
            found.append(action)
            while at < len(line) and line[at] in " ,":
                at += 1
    return found


def score(case: Case, reply: str) -> tuple[str, str, bool]:
    """correct | wrong | other, the text judged, and whether the reply used the action format properly. An answer
    put under another key ({"answer": ...}, or {"type": ...} that would type it) is judged on its content."""
    for action in actions(reply):
        if case.kind == "answer":
            said = next((str(action[k]) for k in ("say", "answer", "ask") if k in action), None)
            proper = "say" in action
            if said is None and "type" in action and len(action) <= 2 and "into" not in action:
                said, proper = str(action["type"]), False
            if said is None:
                continue
            if "ask" in action:
                return "other", said, True
            bad = case.wrong and re.search(case.wrong, said)
            return ("correct" if re.search(case.expect, said, re.I) and not bad else "wrong"), said, proper
        if case.kind == "click":
            target = next((str(action[k]) for k in ("click", "double_click", "right_click") if k in action), None)
            if target is not None:
                bad = case.wrong and re.search(case.wrong, target)
                return ("correct" if re.search(case.expect, target, re.I) and not bad else "wrong"), target, True
    return "other", reply.strip()[:200], True


def point(pointer: OpenRouterJSON, image: np.ndarray, side: str, target: str, box: list[float]) -> dict:
    """`side`: a long side as `Pointer.point` shrinks it, "mp<megapixels>" fitted to that budget, or "native"."""
    jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
    if side.startswith("mp"):
        jpeg = budget(image, float(side[2:]), 85)
    elif side != "native":
        jpeg = shrink(jpeg, int(side))
    started = time.monotonic()
    try:
        data, timing = pointer.call(jpeg, POINT_PROMPT.format(target=target), "You locate controls in screenshots.",
                                    POINT_SCHEMA)
    except Exception as exc:  # noqa: BLE001 - a failed call is recorded, not retried
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}
    x, y = float(data.get("x", -1)) / 1000, float(data.get("y", -1)) / 1000
    vw, vh = SCREEN["width"] / SCREEN["scale"], SCREEN["height"] / SCREEN["scale"]
    mx, my = 4 / vw, 4 / vh  # a click within 4 CSS px of the element still lands on it
    hit = bool(data.get("found")) and box[0] - mx <= x <= box[2] + mx and box[1] - my <= y <= box[3] + my
    return {"x": x, "y": y, "found": data.get("found"), "hit": hit, "s": round(time.monotonic() - started, 3),
            "ttft_s": round(timing.get("ttft") or 0, 3), "image_bytes": len(jpeg)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--sides", nargs="+", default=list(SIDES))
    parser.add_argument("--point-sides", nargs="*", default=list(POINT_SIDES))
    parser.add_argument("--screen", default="2880x1800", help="the rendered screen's pixels, WxH")
    parser.add_argument("--scale", type=float, default=1.5, help="its UI scale (1.5: 150%%)")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    width, height = map(int, args.screen.split("x"))
    SCREEN.update(width=width, height=height, scale=args.scale)
    cases = [c for c in CASES if not c.screen.endswith(".jpg") or (width, height) == (2880, 1800)]
    boxes = render(args.out)
    (args.out / "boxes.json").write_text(json.dumps(boxes, indent=1))
    frames = {c.screen: frame(args.out, c.screen) for c in cases}
    key = os.environ["OPENROUTER_API_KEY"]
    jobs = [("plan", c, side, rep) for c in cases for side in args.sides for rep in range(args.reps)]
    jobs += [("point", c, side, rep) for c in cases if c.point for side in args.point_sides for rep in range(args.reps)]
    random.Random(18).shuffle(jobs)  # sizes interleaved, so a slow provider spell doesn't land on one size
    lock, log = threading.Lock(), (args.out / "calls.jsonl").open("a")

    def run(job) -> None:
        kind, case, side, rep = job
        row = {"kind": kind, "case": case.name, "side": side, "rep": rep, "screen": args.screen, "scale": args.scale}
        if kind == "plan":
            with httpx.Client(timeout=httpx.Timeout(60, read=20), headers={"Authorization": f"Bearer {key}"}) as client:
                try:
                    row |= plan(client, frames[case.screen], side, case.request)
                except Exception as exc:  # noqa: BLE001 - recorded, not retried
                    row["error"] = f"{type(exc).__name__}: {exc}"[:200]
            if "reply" in row:
                row["score"], row["judged"], row["format_ok"] = score(case, row["reply"])
        else:
            pointer = OpenRouterJSON(POINTER, key, "", only=PINNED[POINTER][0], keepalive=False, read_s=10, max_tokens=128)
            try:
                row |= point(pointer, frames[case.screen], side, case.point, boxes[case.box])
            finally:
                pointer.client.close()
        with lock:
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(kind, case.name, side, row.get("score", row.get("hit", row.get("error"))), flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(run, jobs))
    log.close()


if __name__ == "__main__":
    main()
