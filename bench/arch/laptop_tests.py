"""Study 16: the shared-queue architecture (`queue_agent.py`) on the real test laptop, 11 spoken tasks.

The agent runs here and sees only the laptop's screenshots; it acts only through the uinput keyboard and
mouse (`laptop.py`). Each turn is spoken word by word at SPEECH_S a word, the agent hearing it grow; the next
turn comes once the agent has gone quiet (nothing running or queued for QUIET_S). Setup, cleanup and the
oracles use SSH commands only: KWin's window list, files, the display mode, the wallpaper, a test Chrome's
tab list (its debugging port), and a Blender startup script that writes the scene to a file.

  PYTHONPATH=. .venv/bin/python ~/.cache/exo-bench/withkeys.py bench.arch.laptop_tests [--tests 1 2] [--repeats 1]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2

from bench.arch.laptop import HOST, LaptopTarget, shell
from bench.arch.queue_agent import CLICKERS, QueueAgent

OUT = Path("work/arch/laptop")
SPEECH_S = 0.35  # a word, about 170 words a minute
QUIET_S = 2.0  # the agent counts as finished with a turn after this long with nothing running or queued
# Fast vision models, the instruction-tuned (chat) variants OpenRouter serves; effort "" = reasoning off.
MODELS = {"deepseek": ("deepseek/deepseek-v4.1-flash-1017", ""), "gemma31b": ("google/gemma-4-31b-it", ""),
          "flashlite": ("google/gemini-3.5-flash-lite", "minimal"), "flash38": ("google/gemini-3.8-flash", "minimal"),
          "luna": ("openai/gpt-6-luna", ""), "haiku": ("anthropic/claude-haiku-5.5", ""),
          "qwen27b": ("qwen/qwen3.8-27b", ""),
          "qwen27b-think": ("qwen/qwen3.8-27b", "low"), "glm": ("z-ai/glm-5.3-flashx", "minimal"),
          "nemotron": ("nvidia/nemotron-3.5-super-vl", ""), "mimo": ("xiaomi/mimo-v2.6-flash", ""),
          "step5": ("stepfun/step-5-preview", "minimal"), "minimax": ("minimax/minimax-m3.1-flash:free", "minimal")}
CHROME = ("setsid -f google-chrome-stable --user-data-dir=/tmp/exo-chrome --no-first-run --no-default-browser-check "
          "--password-store=basic --ozone-platform=wayland --remote-debugging-port=9333 {urls} >/dev/null 2>&1; sleep 4")
WINDOWS_JS = """var ws = workspace.windowList ? workspace.windowList() : workspace.clientList();
for (var i = 0; i < ws.length; i++) { var w = ws[i]; if (w.normalWindow || w.dialog)
  print("EXOWIN " + JSON.stringify({caption: w.caption, cls: w.resourceClass, active: w.active})); }"""
CLOSE_JS = """var ws = workspace.windowList ? workspace.windowList() : workspace.clientList();
for (var i = 0; i < ws.length; i++) { var w = ws[i]; if ((w.normalWindow || w.dialog) && w.resourceClass != "plasmashell") w.closeWindow(); }"""
BLENDER_ORACLE = r'''
import bpy, json
def dump():
    objs = []
    for o in bpy.data.objects:
        mats = []
        for s in getattr(o, "material_slots", []):
            m = s.material
            if m is None:
                continue
            color = list(m.diffuse_color)[:3]
            if m.use_nodes and m.node_tree:
                for n in m.node_tree.nodes:
                    if n.type == "BSDF_PRINCIPLED":
                        color = list(n.inputs["Base Color"].default_value)[:3]
            mats.append(color)
        objs.append({"name": o.name, "type": o.type, "dims": list(o.dimensions), "loc": list(o.location), "mats": mats})
    with open("/tmp/exo_blender.json", "w") as f:
        json.dump(objs, f)
    return 1.0
bpy.app.timers.register(dump, first_interval=1.0, persistent=True)
'''
BLENDER_STARTUP = "~/.config/blender/5.2/scripts/startup/exo_oracle.py"


def kwin(js: str) -> str:
    """Run a KWin script; its prints come back through the journal, tagged EXOWIN<n> for this run only."""
    tag = f"EXOWIN{random.randrange(10**9)} "
    return shell(f"""cat > /tmp/exo_kwin.js <<'JS'
{js.replace("EXOWIN ", tag)}
JS
name=exo_$RANDOM; id=$(qdbus6 org.kde.KWin /Scripting org.kde.kwin.Scripting.loadScript /tmp/exo_kwin.js $name)
qdbus6 org.kde.KWin /Scripting/Script$id org.kde.kwin.Script.run >/dev/null; sleep 0.6
journalctl --user -b --since -10s -o cat | grep '{tag.strip()} '
qdbus6 org.kde.KWin /Scripting org.kde.kwin.Scripting.unloadScript $name >/dev/null""").replace(tag, "EXOWIN ")


def windows() -> list[dict]:
    seen = [json.loads(line.split("EXOWIN ", 1)[1]) for line in kwin(WINDOWS_JS).splitlines() if "EXOWIN " in line]
    return list({(w["caption"], w["cls"]): w for w in seen}.values())


def active_caption() -> str:
    return next((w["caption"] for w in windows() if w["active"]), "")


def chrome_targets() -> list[dict]:
    """Every page and embedded frame of the test Chrome (a video can play in an embedded player)."""
    out = shell("curl -s --max-time 3 http://127.0.0.1:9333/json/list")
    try:
        return [t for t in json.loads(out) if t.get("type") in ("page", "iframe")]
    except json.JSONDecodeError:
        return []


def chrome_tabs() -> list[dict]:
    return [t for t in chrome_targets() if t["type"] == "page"]


def video_playing() -> bool:
    """Whether any test-Chrome tab has a <video> playing: its debugging port, tunnelled over SSH."""
    from websockets.sync.client import connect

    tunnel = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-N", "-L", "19333:127.0.0.1:9333", HOST])
    try:
        time.sleep(1.0)
        for tab in chrome_targets():
            url = tab.get("webSocketDebuggerUrl", "").replace("127.0.0.1:9333", "127.0.0.1:19333")
            try:
                with connect(url, open_timeout=3, max_size=None) as ws:
                    ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {"expression":
                        "[...document.querySelectorAll('video')].some(v => !v.paused && v.currentTime > 0)"}}))
                    if json.loads(ws.recv(timeout=3)).get("result", {}).get("result", {}).get("value"):
                        return True
            except Exception:  # noqa: BLE001 - a tab that doesn't answer has no video playing
                continue
        return False
    finally:
        tunnel.terminate()


def refresh_hz() -> float:
    out = re.sub(r"\x1b\[[0-9;]*m", "", shell("kscreen-doctor -o"))
    current = re.search(r"\d+:\d+x\d+@([\d.]+)\*", out)
    return float(current.group(1)) if current else 0.0


def wallpaper() -> str:
    """The wallpaper of the desktop on the laptop's screen: plugin, image, colour."""
    return shell("""qdbus6 org.kde.plasmashell /PlasmaShell org.kde.PlasmaShell.evaluateScript '
var d = desktopForScreen(0); d.currentConfigGroup = ["Wallpaper", d.wallpaperPlugin, "General"];
print(d.wallpaperPlugin + "|" + d.readConfig("Image") + "|" + d.readConfig("Color"));'""").strip()


def restore_wallpaper(saved: str) -> None:
    plugin, image, color = (saved.split("|") + ["", "", ""])[:3]
    key, value = ("Image", image) if image else ("Color", color)
    shell(f"""qdbus6 org.kde.plasmashell /PlasmaShell org.kde.PlasmaShell.evaluateScript '
var d = desktopForScreen(0); d.wallpaperPlugin = "{plugin}";
d.currentConfigGroup = ["Wallpaper", "{plugin}", "General"]; d.writeConfig("{key}", "{value}");'""")


def blender_scene() -> list[dict]:
    try:
        return json.loads(shell("cat /tmp/exo_blender.json 2>/dev/null") or "[]")
    except json.JSONDecodeError:
        return []


def pdf(name: str, when: str) -> str:
    """A one-page PDF in Downloads with the given modification time."""
    return (f"printf '%%PDF-1.1\\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>"
            f"endobj 3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 144]>>endobj\\ntrailer<</Root 1 0 R>>\\n%%%%EOF\\n'"
            f" > ~/Downloads/{name}; touch -d '{when}' ~/Downloads/{name}")


@dataclass
class Test:
    name: str
    turns: list[str]
    setup: str  # bash, before the agent starts
    check: object  # () -> (passed, detail)
    cleanup: str = ""
    turn_s: float = 90.0  # how long a turn may run
    data: dict = field(default_factory=dict)


def _youtube() -> tuple[bool, str]:
    """YouTube's own page in front: not the decoy image, nor a video tab the browser restored from its last session."""
    caption = active_caption()
    return bool(re.match(r"(\(\d+\) )?YouTube [—-] ", caption)), caption


def _tabs_check(want) -> object:
    def check() -> tuple[bool, str]:
        tabs = chrome_tabs()
        return want(tabs, active_caption().lower()), json.dumps([t["url"][:70] for t in tabs])
    return check


def _newest_pdf() -> tuple[bool, str]:
    caps = [w["caption"] for w in windows()]
    return any("boarding_pass" in c for c in caps), json.dumps(caps)


def _taxes() -> tuple[bool, str]:
    out = shell("ls ~/Documents/taxes 2>/dev/null; echo ---; ls ~/Downloads | grep exo_invoice")
    inside, _, left = out.partition("---")
    ok = "exo_invoice_0412.pdf" in inside and "exo_invoice_0519.pdf" in inside and "exo_invoice" not in left
    return ok, out.replace("\n", " ")


def _groceries() -> tuple[bool, str]:
    out = shell("ls ~/Desktop; cat ~/Desktop/groceries* 2>/dev/null")
    return "groceries" in out and "milk" in out.lower() and "egg" in out.lower(), out.replace("\n", " | ")


def _display_and_wallpaper(test: Test) -> tuple[bool, str]:
    """The panel only runs at 120 Hz: the agent must find that and say so, leave it there, and change the wallpaper
    (or ask what to change it to)."""
    hz, paper = refresh_hz(), wallpaper()
    said = " ".join(test.data.get("said_turn1", [])).lower()
    # 120 named, and 60 not claimed as the current rate (xrandr on Wayland reads XWayland's 60 and lists 120 as a mode)
    reported = bool(re.search(r"\b(120|119\.9\d*)(\.\d+)?\s*hz", said)) and not re.search(
        r"\b(at|running|currently|set to|is)\b[^.?]{0,25}?\b(60|59\.9\d*)(\.\d+)?\s*hz", said)
    changed = paper != test.data.get("wallpaper")
    asked = test.data.get("asked_wallpaper", False)
    return (hz >= 119 and reported and (changed or asked),
            f"refresh {hz} Hz; reported 120 in turn 1 {reported}; wallpaper {paper!r} (changed {changed}, asked {asked})")


def _printer_pdf() -> tuple[bool, str]:
    out = shell("find ~/Downloads -maxdepth 1 -name '*.pdf' -newer /tmp/exo_test_start -printf '%f\\n'")
    return any("rint" in f.lower() for f in out.split()), out.strip()


def footprint(meshes: list[dict]) -> float:
    """The house's width: the larger horizontal extent of all its meshes together (positions and sizes)."""
    if not meshes:
        return 0.0
    spans = []
    for axis in (0, 1):
        low = min(o["loc"][axis] - o["dims"][axis] / 2 for o in meshes)
        high = max(o["loc"][axis] + o["dims"][axis] / 2 for o in meshes)
        spans.append(high - low)
    return round(max(spans), 2)


def _house(test: Test) -> tuple[bool, str]:
    scene = [o for o in blender_scene() if o["type"] == "MESH"]
    widths = test.data.get("widths", [])
    brown = any(m for o in scene for m in o["mats"] if m[0] > m[1] > m[2] and m[0] > 0.15 and m[0] < 0.85)
    wider = len(widths) >= 2 and widths[-1] > widths[0] * 1.05
    detail = f"{len(scene)} meshes {[o['name'] for o in scene]}; widths after turns {widths}; brown {brown}"
    return len(scene) >= 2 and wider and brown, detail


DOWNLOADS = " && ".join([pdf("exo_report_q3.pdf", "-3 days"), pdf("exo_lease.pdf", "-40 days"),
                         pdf("exo_boarding_pass.pdf", "now"), pdf("exo_invoice_0412.pdf", "-9 days"),
                         pdf("exo_invoice_0519.pdf", "-5 days")])
CLEAN_FILES = ("rm -f ~/Downloads/exo_* ~/Downloads/youtube_thumbnail.jpg; rm -rf ~/Documents/taxes; "
               # Whatever a test saved on the Desktop (empty before the tests), under any name it chose.
               "[ -f /tmp/exo_test_start ] && find ~/Desktop -mindepth 1 -maxdepth 1 -newer /tmp/exo_test_start -exec rm -rf {} +")
TESTS = [
    Test("open youtube", ["open youtube"],
         "cp /usr/share/wallpapers/Next/contents/images/1920x1080.png ~/Downloads/youtube_thumbnail.jpg 2>/dev/null || "
         "head -c 2000 /dev/urandom > ~/Downloads/youtube_thumbnail.jpg", lambda t: _youtube()),
    Test("weather tab", ["open a new tab and search for the weather in gainesville"],
         CHROME.format(urls="https://en.wikipedia.org"),
         lambda t: _tabs_check(lambda tabs, cap: any("gainesville" in x["url"].lower() and "weather" in x["url"].lower()
                                                     for x in tabs) and len(tabs) >= 2)()),
    Test("close named tab", ["close the youtube tab"],
         CHROME.format(urls="https://en.wikipedia.org https://www.youtube.com https://github.com"),
         lambda t: _tabs_check(lambda tabs, cap: len(tabs) == 2 and not any("youtube" in x["url"] for x in tabs))()),
    Test("space video", ["find me an interesting video about space and play it"], CHROME.format(urls="https://www.google.com"),
         lambda t: (video_playing(), active_caption()), turn_s=120),
    Test("newest pdf", ["open my downloads and open the newest pdf"], DOWNLOADS, lambda t: _newest_pdf()),
    Test("taxes folder", ["make a folder called taxes in documents and move the invoice pdfs from downloads into it"],
         DOWNLOADS, lambda t: _taxes(), turn_s=150),
    Test("groceries note", ["open a text editor, write buy milk and eggs, and save it on the desktop as groceries"],
         "true", lambda t: _groceries(), turn_s=120),
    Test("120hz and wallpaper", ["my monitor feels like its on 60hz, can you check for me",
                                 "yes change it to 120hz", "now change my wallpaper"],
         "true", _display_and_wallpaper, turn_s=120, cleanup="kscreen-doctor output.eDP-1.mode.1 >/dev/null 2>&1"),
    Test("printer article pdf", ["save the wikipedia article on printers as a pdf"], CHROME.format(urls="https://www.google.com"),
         lambda t: _printer_pdf(), turn_s=150),
    Test("correction", ["open youtube actually no open wikipedia and look up the eiffel tower"],
         CHROME.format(urls="https://www.google.com"),
         lambda t: _tabs_check(lambda tabs, cap: "eiffel" in cap)()),
    # the stress test: off the default run until everything above passes (`--tests 11`)
    Test("blender house", ["open up blender and make a house", "actually make it wider", "make it brown"],
         f"mkdir -p ~/.config/blender/5.2/scripts/startup && cat > {BLENDER_STARTUP} <<'PY'\n{BLENDER_ORACLE}\nPY\n"
         "rm -f /tmp/exo_blender.json", _house, turn_s=300, cleanup=f"rm -f {BLENDER_STARTUP} /tmp/exo_blender.json"),
]
# The default run, ~3.5 min: 2 (weather) and 9 (printer pdf) passed every run of the baseline and repeat others'
# skills; 11 (Blender) waits until everything else passes.
DEFAULT = [1, 3, 4, 5, 6, 7, 8, 10]


def reset() -> None:
    """Close every app window, the test Chrome and leftover files: each test starts on the empty desktop."""
    kwin(CLOSE_JS)
    time.sleep(1.5)
    shell("pkill -f exo-chrome; for app in blender okular kate kwrite systemsettings dolphin konsole gwenview firefox "
          f"brave; do pkill -x $app; done; {CLEAN_FILES}; sleep 1")


def speak(agent: QueueAgent, words: str) -> float:
    """Word by word, as speech-to-text would deliver it; returns when the last word was heard."""
    said = []
    for word in words.split():
        said.append(word)
        agent.brain.hear(" ".join(said), speaking=True)
        time.sleep(SPEECH_S)
    agent.brain.hear(" ".join(said), speaking=False)
    return time.monotonic()


STALL_S = 8.0  # no new action and no model output for this long while busy: stalled
REPEATS = 3  # the same action this many times in a row: stalled


def progress(agent: QueueAgent) -> int:
    return len(agent.shared.log) + sum(len(c["lines"]) for c in agent.brain.calls)


def wait_quiet(agent: QueueAgent, limit_s: float) -> tuple[float, str]:
    """Until the agent goes quiet; or why it stalled."""
    started = quiet_since = moved_at = time.monotonic()
    seen = progress(agent)
    while time.monotonic() - started < limit_s:
        now = time.monotonic()
        if progress(agent) != seen:
            seen, moved_at = progress(agent), now
        recent = [json.dumps(r.action, sort_keys=True) for r in agent.shared.log[-6 * REPEATS:]]
        for size in range(1, 7):  # one action, or a cycle of up to six, over and over
            tail = recent[-size * REPEATS:]
            if len(tail) == size * REPEATS and all(tail[i] == tail[i % size] for i in range(len(tail))):
                return now - started, f"repeating {tail[:size]}"
        if agent.brain.busy():
            quiet_since = now
            if now - moved_at >= STALL_S:
                return now - started, (f"no progress for {STALL_S:.0f} s: brain {agent.brain.phase}, calling "
                                       f"{agent.brain.calling}, executor {agent.shared.state}, queue {len(agent.shared.queue)}, "
                                       f"dirty {agent.brain.dirty}, alive {agent.brain.is_alive()}/{agent.executor.is_alive()}")
        elif now - quiet_since >= QUIET_S:
            return now - started, ""
        time.sleep(0.2)
    return time.monotonic() - started, f"over the {limit_s:.0f} s turn limit"


def run(test: Test, index: int, repeat: int, model: str, effort: str, target: LaptopTarget, label: str,
        clicker: str, plan: str) -> dict:
    folder = OUT / f"{label}-{index}-{repeat}"
    folder.mkdir(parents=True, exist_ok=True)
    reset()
    test.data = {}
    shell(f"touch /tmp/exo_test_start; {test.setup}", timeout=60)
    if test.name.startswith("120hz"):
        test.data["wallpaper"] = wallpaper()
    grabs_before = len(target.grabs)
    agent = QueueAgent(target, model, effort, os.environ["TYPESAFE_API_KEY"], folder, clicker, plan)
    timeline: list[tuple[float, str]] = []  # the active window over the run, for debugging (oracle side only)
    watching = threading.Event()

    def watch() -> None:
        while not watching.is_set():
            timeline.append((round(time.monotonic() - agent.started, 1), active_caption()))
            watching.wait(0.5)

    threading.Thread(target=watch, daemon=True).start()
    turns = []
    try:
        for k, words in enumerate(test.turns):
            if k:
                agent.brain.end_turn()
            inputs_before = len(agent.executor.inputs)
            heard = speak(agent, words)
            took, stalled = wait_quiet(agent, test.turn_s)
            firsts = agent.executor.inputs[inputs_before:]
            turns.append({"turn": words, "after_speech_s": round(took, 1), "stalled": stalled,
                          "first_input_vs_speech_end_s": round(firsts[0] - heard, 2) if firsts else None})
            if stalled:
                break
            if test.name.startswith("blender"):
                test.data.setdefault("widths", []).append(footprint([o for o in blender_scene() if o["type"] == "MESH"]))
            if test.name.startswith("120hz") and k == 0:
                test.data["said_turn1"] = list(agent.brain.said)
            if test.name.startswith("120hz") and k == 2:
                test.data["asked_wallpaper"] = any("?" in s for s in agent.brain.said[-2:])
        passed, detail = test.check(test)
        passed = passed and not any(t["stalled"] for t in turns)
    finally:
        watching.set()
        cv2.imwrite(str(folder / "after.jpg"), target.grab())
        agent.close()
        if test.cleanup:
            shell(test.cleanup)
        if test.name.startswith("120hz"):
            restore_wallpaper(test.data["wallpaper"])
    calls = agent.brain.calls
    row = {"test": test.name, "index": index, "repeat": repeat, "model": label, "passed": passed, "detail": detail,
           "turns": turns, "said": agent.brain.said, "calls": len(calls),
           "cancelled": sum(c["cancelled"] for c in calls),
           "ttft_p50_s": statistics.median(ttfts) if (ttfts := [c["ttft_s"] for c in calls if "ttft_s" in c]) else None,
           "grounds": sum(t.get("label") == "ground" for t in agent.trace),
           "screenshots": len(target.grabs) - grabs_before,
           "screenshot_s": round(sum(target.grabs[grabs_before:]), 1),
           "not_found": sum(r.result == "not_found" for r in agent.shared.log),
           "errors": [asdict(r) for r in agent.shared.log if r.result == "error"]}
    (folder / "calls.json").write_text(json.dumps(calls, indent=1))
    (folder / "log.json").write_text(json.dumps([asdict(r) for r in agent.shared.log], indent=1))
    (folder / "trace.json").write_text(json.dumps(agent.trace, indent=1, default=str))
    (folder / "windows.json").write_text(json.dumps(timeline, indent=0))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", nargs="*", type=int, help="1-based test numbers (default: DEFAULT)")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--model", default="deepseek", choices=list(MODELS))
    parser.add_argument("--clicker", default="jev", choices=["jev", *CLICKERS])
    parser.add_argument("--plan", default="screen", choices=["screen", "ahead"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "report.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    model, effort = MODELS[args.model]
    target = LaptopTarget()
    try:
        for repeat in range(args.repeats):
            for n in args.tests or DEFAULT:
                label = "+".join([args.model, *[x for x in (args.clicker, args.plan) if x not in ("jev", "screen")]])
                row = run(TESTS[n - 1], n, repeat, model, effort, target, label, args.clicker, args.plan)
                rows.append(row)
                path.write_text(json.dumps(rows, indent=1))
                print(json.dumps({k: row[k] for k in ("test", "model", "passed", "detail", "turns", "calls",
                                                      "cancelled", "not_found")})[:600], flush=True)
    finally:
        reset()
        target.close()


if __name__ == "__main__":
    main()
