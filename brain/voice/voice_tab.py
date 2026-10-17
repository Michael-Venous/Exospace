#!/usr/bin/env python3
"""Voice demo at http://127.0.0.1:8766: speak, Deepgram transcribes, Jev pulls out the app name,
and the OS launcher's own search opens it, checked on screen by Jev. No installed-app list."""
import asyncio
import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import gi
import websockets
from Xlib import X, XK, display
from Xlib.ext import xtest

gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, GdkPixbuf  # noqa: E402

ENV = Path(__file__).resolve().parents[2] / ".env"
PORT = 8766
WS_PORT = 8767
DG_URL = "wss://api.deepgram.com/v1/listen?model=nova-3&interim_results=true&endpointing=300&smart_format=true"
JEV_URL = "https://api.typesafe.ai/v1/systemone"
# Cinnamon ignores a faked Super press (the Pico is a real USB keyboard, so the box is fine). To test on this laptop,
# bind a second menu shortcut such as Ctrl+Esc and put it here; until then the launcher check stops before typing.
LAUNCHER = ("Super_L",)

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Exospace Voice</title>
<style>
body{margin:0;min-height:100vh;display:grid;place-content:center;justify-items:center;gap:100px;padding:16px;box-sizing:border-box;
  background:#fff;color:#1d1d1f;font:500 22px/1.4 -apple-system,"Ubuntu Sans",system-ui,sans-serif;text-align:center}
#orb{width:150px;height:150px;border:0;border-radius:50%;cursor:pointer;
  background:radial-gradient(circle at 34% 28%,#fff 0,#cfe0ff 18%,#7d93ff 52%,#b77bff 80%,#ff9ad5 100%);
  box-shadow:0 18px 50px rgb(125 147 255/.35),0 0 0 calc(var(--lvl,0)*36px) rgb(125 147 255/.12);
  transform:scale(calc(1 + var(--lvl,0)*.55));transition:transform 90ms ease-out,box-shadow 90ms ease-out;animation:breathe 4s ease-in-out infinite}
#orb:focus-visible{outline:2px solid #7d93ff;outline-offset:8px}
.on #orb{animation:breathe 4s ease-in-out infinite,drift 6s linear infinite}
#heard{max-width:min(640px,90vw);min-height:2.8em;margin:0}
#live{color:#86868b}
#said:empty~#live:empty::before{content:"Tap to talk";color:#a1a1a6}
.err{color:#d70015}
@keyframes breathe{50%{scale:1.04}}
@keyframes drift{to{filter:hue-rotate(360deg)}}
@media (prefers-reduced-motion:reduce){*{animation:none!important}}
</style>
<button id=orb aria-label="Start listening" aria-pressed=false></button>
<p id=heard aria-live=polite><span id=said></span> <span id=live></span></p>
<script>
let stopListening = null;
function listening(on) {
  document.body.classList.toggle("on", on);
  orb.setAttribute("aria-pressed", on);
  orb.ariaLabel = on ? "Stop listening" : "Start listening";
}
async function run(text) {
  let r;
  // A server crash returns HTML or drops the connection; show it instead of failing silently.
  try { r = await fetch("/command", {method: "POST", headers: {"Content-Type": "application/json"},
                                     body: JSON.stringify({text})}).then(r => r.json()); }
  catch (err) { r = {error: "No answer from the server: " + err.message}; }
  if (r.error) { said.className = "err"; said.textContent = r.error; }
}
orb.onclick = async () => {
  if (stopListening) return stopListening();
  let mic;
  try { mic = await navigator.mediaDevices.getUserMedia({audio: true}); }
  catch (err) { said.className = "err"; said.textContent = "Microphone blocked: " + err.message; return; }
  const rec = new MediaRecorder(mic), sock = new WebSocket(`ws://${location.hostname}:""" + str(WS_PORT) + """`);
  let done = "";  // finalized words of the utterance in progress
  // Mic level (RMS) drives the orb's size through --lvl.
  const ctx = new AudioContext(), an = ctx.createAnalyser(), buf = new Uint8Array(512);
  an.fftSize = buf.length;
  ctx.createMediaStreamSource(mic).connect(an);
  (function tick() {
    if (ctx.state === "closed") return orb.style.removeProperty("--lvl");
    an.getByteTimeDomainData(buf);
    let sum = 0; for (const v of buf) sum += (v - 128) ** 2;
    orb.style.setProperty("--lvl", Math.min(1, Math.sqrt(sum / buf.length) / 30));
    requestAnimationFrame(tick);
  })();
  stopListening = () => rec.state === "recording" ? rec.stop() : sock.close();
  rec.ondataavailable = e => { if (e.data.size && sock.readyState === 1) sock.send(e.data); };
  // CloseStream makes Deepgram send its last words, then close the socket.
  rec.onstop = () => { if (sock.readyState === 1) sock.send(JSON.stringify({type: "CloseStream"})); };
  sock.onopen = () => {
    rec.start(100);
    listening(true);
    said.className = ""; said.textContent = ""; live.textContent = "Listening…";
  };
  sock.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.type !== "Results") return;
    const words = m.channel.alternatives[0].transcript;
    if (m.is_final && words) done = (done + " " + words).trim();
    // Silence sends empty results; skip them so the last thing said stays on screen.
    if (words) { said.className = ""; said.textContent = done; live.textContent = m.is_final ? "" : words; }
    if (m.speech_final && done) { run(done); done = ""; }
  };
  sock.onclose = e => {
    stopListening = null;
    if (rec.state !== "inactive") rec.stop();
    mic.getTracks().forEach(t => t.stop());
    ctx.close();
    listening(false); live.textContent = "";
    if (done) run(done);
    if (e.code !== 1000) { said.className = "err"; said.textContent = `Deepgram connection closed (${e.code}). Check the server log.`; }
  };
};
</script>"""


def env(name: str) -> str:
    for line in ENV.read_text().splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"{name} missing from .env")


KEY = env("TYPESAFE_API_KEY")
DG_KEY = env("DEEPGRAM_API_KEY")
DPY = display.Display()
ACTIONS = {"open_app": "Open or launch an app or program.",
           "open_tab": "Open a new tab in the Chrome browser that is already open.",
           "none_here": "Anything else, or the request is unclear."}
SPAN = ("If `spoken_command` asks to open an app, choose the {} word of the app's name or kind of app, "
        "leaving out request words like open, launch, my or please.")


def jev(questions: dict, state: dict, frame: bytes | None = None) -> tuple[dict, int]:
    """Ask Jev; with a frame, `@screen` in the state is that PNG. Returns the answers and the time in ms."""
    req = {"model": "jev-1.14.0", "state": state, "questions": questions}
    headers = {"Authorization": f"Bearer {KEY}"}
    if frame is None:
        body, headers["Content-Type"] = json.dumps(req).encode(), "application/json"
    else:
        req["assets"] = {"@screen": {"type": "image", "part": "frame"}}
        b = uuid.uuid4().hex
        body = (f'--{b}\r\nContent-Disposition: form-data; name="request"\r\nContent-Type: application/json\r\n\r\n'
                f'{json.dumps(req)}\r\n--{b}\r\nContent-Disposition: form-data; name="frame"; filename="frame.png"\r\n'
                f'Content-Type: image/png\r\n\r\n').encode() + frame + f"\r\n--{b}--\r\n".encode()
        headers |= {"Content-Type": f"multipart/form-data; boundary={b}", "TypeSafe-Danger-Zone-Images": "true"}
    start = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(JEV_URL, data=body, headers=headers), timeout=60) as r:
        answers = json.load(r)["answers"]
    return answers, round((time.perf_counter() - start) * 1000)


def screenshot() -> bytes:
    """The primary monitor, where Cinnamon's menu opens, as PNG at its logical size."""
    g = Gdk.Display.get_default().get_primary_monitor().get_geometry()
    shot = Gdk.pixbuf_get_from_window(Gdk.get_default_root_window(), g.x, g.y, g.width, g.height)
    return shot.scale_simple(g.width, g.height, GdkPixbuf.InterpType.BILINEAR).save_to_bufferv("png", [], [])[1]


def press(*keys: str) -> None:
    """Press keysyms together, then release, like the Pico's `key` op; XTest stands in for its USB keyboard."""
    codes = [DPY.keysym_to_keycode(XK.string_to_keysym(k)) for k in keys]
    for c in codes:
        xtest.fake_input(DPY, X.KeyPress, c)
    DPY.sync()
    time.sleep(0.01)
    for c in reversed(codes):
        xtest.fake_input(DPY, X.KeyRelease, c)
    DPY.sync()


def open_app(name: str) -> dict:
    """PLAN §6 keyboard skill: open the OS launcher, type the name, Jev checks the top result, Enter, Jev verifies."""
    # Only letters, digits and spaces are typed, so spoken text can't become a shortcut or a command.
    query = re.sub(r"[^a-z0-9 ]", "", name.lower())[:40].strip()
    steps = []

    def check(label: str, question: str, wait: float) -> bool:
        time.sleep(wait)  # shortcut: fixed waits; the box should use settle detection (PLAN §6).
        answers, ms = jev({"q": {"type": "noul", "instructions": question}}, {"screen": "@screen"}, screenshot())
        steps.append(f"{label} {answers['q']['noul']:.2f} ({ms} ms)")
        return answers["q"]["noul"] > 0.5

    if not query:
        return {"steps": ["Nothing typeable in the name"]}
    press(*LAUNCHER)
    # Typing before the launcher is confirmed open would type into whatever window has focus.
    if not check("Launcher open?", "Is an app launcher or start menu with a search box open in `screen`?", 0.4):
        return {"steps": steps}
    for ch in query:
        press("space" if ch == " " else ch)
    if not check(f"Typed '{query}', top result matches?",
                 f"In `screen`, does the highlighted top result of the app menu search match what the user asked "
                 f"to open: {name!r}?", 0.4):
        press("Escape")
        return {"steps": steps}
    press("Return")
    check("Opened?", f"Is the app the user asked for with {name!r} open in `screen`?", 1.5)
    return {"steps": steps, "opened": True}


def handle(text: str) -> dict:
    # 500 characters hold at most 250 words, inside Jev's 254-option limit.
    words = {f"w{i}": m for i, m in enumerate(re.finditer(r"\w+", text))}
    spans = {k: f"Word {k[1:]}: {m[0]}" for k, m in words.items()} | {"none_here": "No app is named."}
    answers, ms = jev({"action": {"type": "choice", "instructions": "What does `spoken_command` ask the computer to do?",
                                  "criteria": ACTIONS},
                       "first": {"type": "choice", "instructions": SPAN.format("FIRST"), "criteria": spans},
                       "last": {"type": "choice", "instructions": SPAN.format("LAST"), "criteria": spans}},
                      {"spoken_command": text})
    choice, first, last = (answers[k]["choice"] for k in ("action", "first", "last"))
    result = {"ms": ms, "choice": choice, "confidence": answers["action"]["confidence"], "opened": False}
    if choice == "open_tab":
        # Fixed URL: spoken text never reaches the command line.
        subprocess.Popen(["google-chrome", "--new-tab", "chrome://newtab/"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        result["opened"] = True
    elif choice == "open_app" and first in words and last in words and words[first].start() <= words[last].start():
        # Jev picks word positions; the name is copied from the transcript, never written by a model.
        result["label"] = text[words[first].start():words[last].end()]
        result |= open_app(result["label"])
    elif choice == "open_app":
        result["steps"] = ["Couldn't find the app name"]
    return result


async def relay(browser) -> None:
    """Pipe mic audio from the page to Deepgram and transcripts back; the key never reaches the browser."""
    async with websockets.connect(DG_URL, extra_headers={"Authorization": f"Token {DG_KEY}"}) as dg:
        async def upload() -> None:
            try:
                async for chunk in browser:
                    await dg.send(chunk)
            finally:
                await dg.close()

        uploader = asyncio.create_task(upload())
        try:
            async for msg in dg:
                await browser.send(msg)
        except websockets.ConnectionClosed:
            pass
        finally:
            uploader.cancel()


async def serve_ws() -> None:
    # Origin check stops other sites from streaming through this relay (WebSockets skip CORS).
    origins = [f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"]
    async with websockets.serve(relay, "127.0.0.1", WS_PORT, origins=origins):
        await asyncio.Future()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.reply(200, "text/html; charset=utf-8", PAGE.encode())
        else:
            self.reply(404, "text/plain", b"not found")

    def do_POST(self):
        # Host + JSON content-type checks stop other sites from spending the key or pressing keys via this server.
        if self.path != "/command" or self.headers.get("Host") not in (f"127.0.0.1:{PORT}", f"localhost:{PORT}") \
                or self.headers.get("Content-Type") != "application/json":
            return self.reply(403, "application/json", b'{"error":"forbidden"}')
        text = str(json.loads(self.rfile.read(int(self.headers["Content-Length"]))).get("text", "")).strip()
        if not 0 < len(text) <= 500:
            return self.reply(400, "application/json", b'{"error":"text must be 1-500 characters"}')
        try:
            result = handle(text)
        except urllib.error.HTTPError as e:
            result = {"error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"}
        except OSError as e:
            result = {"error": str(e)}
        self.reply(200, "application/json", json.dumps(result).encode())

    def reply(self, code: int, ctype: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"Voice-to-tab demo on http://127.0.0.1:{PORT} (Deepgram relay on ws://127.0.0.1:{WS_PORT})", flush=True)
    threading.Thread(target=HTTPServer(("127.0.0.1", PORT), Handler).serve_forever, daemon=True).start()
    asyncio.run(serve_ws())
