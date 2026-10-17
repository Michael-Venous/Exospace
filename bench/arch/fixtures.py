"""Synthetic but realistic screens for architecture studies.

Each screen is plain HTML rendered by Playwright. Elements of interest carry
`data-gt` ids; their boxes are ground truth for scoring only. Models see pixels.
"""

from __future__ import annotations

import json
from pathlib import Path

BROWSER_FRAME = """
<div style="height:72px;background:#dee1e6;font:13px system-ui;display:flex;flex-direction:column">
 <div style="height:34px;display:flex;align-items:flex-end;padding-left:8px">
  <div style="background:#fff;border-radius:8px 8px 0 0;padding:7px 14px;width:200px">{tab}</div></div>
 <div style="height:38px;background:#fff;display:flex;align-items:center;gap:6px;padding:0 8px">
  <div data-gt="back" style="width:30px;height:30px;border-radius:50%;display:grid;place-items:center;font-size:20px">&#8592;</div>
  <div style="width:30px;height:30px;display:grid;place-items:center;font-size:20px;color:#999">&#8594;</div>
  <div style="width:30px;height:30px;display:grid;place-items:center;font-size:18px">&#8635;</div>
  <div style="flex:1;background:#f1f3f4;border-radius:16px;padding:7px 14px;color:#333">{url}</div>
 </div></div>"""


def _page(body: str, style: str = "") -> str:
    return f"<!doctype html><html><body style='margin:0;font-family:system-ui,Segoe UI,sans-serif;{style}'>{body}</body></html>"


def _web(tab: str, url: str, content: str) -> str:
    return _page(BROWSER_FRAME.format(tab=tab, url=url) + content)


DESKTOP = _page("""
<div style="position:absolute;inset:0 0 48px 0;background:linear-gradient(135deg,#0b3d91,#2a7bd1)">
 <div style="display:flex;flex-direction:column;gap:22px;padding:24px;width:96px;color:#fff;font-size:12px;text-align:center">
  <div data-gt="recycle"><div style="font-size:40px">&#128465;</div>Recycle Bin</div>
  <div data-gt="chrome"><div style="width:44px;height:44px;margin:auto;border-radius:50%;background:conic-gradient(#db4437 0 33%,#f4b400 0 66%,#0f9d58 0);display:grid;place-items:center"><div style="width:18px;height:18px;border-radius:50%;background:#4285f4;border:3px solid #fff"></div></div>Google Chrome</div>
  <div data-gt="blender"><div style="width:44px;height:44px;margin:auto;border-radius:50%;background:#ea7600;display:grid;place-items:center"><div style="width:16px;height:16px;border-radius:50%;background:#265787;border:3px solid #fff"></div></div>Blender 4.2</div>
  <div data-gt="spotify"><div style="width:44px;height:44px;margin:auto;border-radius:50%;background:#1db954"></div>Spotify</div>
  <div data-gt="report_doc"><div style="font-size:40px">&#128196;</div>Q3 report draft.docx</div>
 </div></div>
<div style="position:absolute;left:0;right:0;bottom:0;height:48px;background:#1f1f1f;display:flex;align-items:center;justify-content:space-between;padding:0 12px;color:#fff;font-size:12px">
 <div data-gt="start" style="width:36px;height:36px;display:grid;grid-template-columns:1fr 1fr;gap:2px;padding:8px;box-sizing:border-box"><i style="background:#4cc2ff"></i><i style="background:#4cc2ff"></i><i style="background:#4cc2ff"></i><i style="background:#4cc2ff"></i></div>
 <div>10:42 AM<br>10/17/2026</div></div>""")

INBOX = _web("Mail - Outlook", "outlook.office.com/mail/inbox", """
<div style="display:flex;height:728px;font-size:14px">
 <div style="width:200px;background:#f3f2f1;padding:12px;display:flex;flex-direction:column;gap:6px">
  <div style="background:#0f6cbd;color:#fff;border-radius:4px;padding:8px;text-align:center">New mail</div>
  <div data-gt="inbox_folder" style="padding:6px;background:#e1dfdd">Inbox <b>3</b></div>
  <div data-gt="sent" style="padding:6px">Sent Items</div><div style="padding:6px">Drafts</div>
  <div style="padding:6px">Deleted Items</div><div style="padding:6px">Archive</div></div>
 <div style="width:420px;border-right:1px solid #ddd">
  <div data-gt="john" style="padding:12px;border-bottom:1px solid #eee;background:#deecf9"><b>John Miller</b><div><b>Need the Q3 report today</b></div><div style="color:#666">Hey, can you send me the Q3 report before 5? Thanks</div></div>
  <div data-gt="janet" style="padding:12px;border-bottom:1px solid #eee"><b>Janet Park</b><div><b>Q3 report numbers + notes</b></div><div style="color:#666">Here are the figures and my notes for the report...</div></div>
  <div data-gt="hr" style="padding:12px;border-bottom:1px solid #eee"><b>HR Team</b><div>Benefits enrollment closes Friday</div><div style="color:#666">Reminder to complete your enrollment</div></div>
  <div data-gt="alex" style="padding:12px;border-bottom:1px solid #eee"><b>Alex Chen</b><div>Lunch tomorrow?</div><div style="color:#666">Thinking tacos at noon</div></div></div>
 <div style="flex:1;padding:20px">
  <div style="font-size:20px">Need the Q3 report today</div><div style="color:#666;margin:6px 0 16px">John Miller &lt;john.miller@contoso.com&gt;</div>
  <div>Hey, can you send me the Q3 report before 5? Leadership wants it for tomorrow's meeting. Thanks, John</div>
  <div style="margin-top:20px;display:flex;gap:8px"><span data-gt="reply" style="border:1px solid #ccc;padding:6px 12px">Reply</span><span style="border:1px solid #ccc;padding:6px 12px">Forward</span></div></div></div>""")

_VIDEOS = [("v_lofi", "#6b4f9e", "lofi hip hop radio - beats to relax/study to", "Lofi Girl"),
           ("v_pasta", "#c0392b", "Fresh pasta from scratch in 10 minutes", "Joshua Weissman"),
           ("v_rocket", "#2c3e50", "Starship flight 9 full launch replay", "SpaceX"),
           ("v_cat", "#e67e22", "Cats that are bad at their jobs", "Daily Dose"),
           ("v_python", "#2980b9", "Python in 100 seconds", "Fireship"),
           ("v_guitar", "#16a085", "Learn 4 chords to play 100 songs", "Andy Guitar")]
YOUTUBE = _web("YouTube", "youtube.com", """
<div style="padding:10px 20px;display:flex;align-items:center;gap:20px;font-size:14px">
 <b style="font-size:20px;color:#f00">&#9654; YouTube</b>
 <div data-gt="search" style="flex:1;max-width:520px;border:1px solid #ccc;border-radius:20px;padding:8px 16px;color:#888">Search</div></div>
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:18px;padding:10px 20px;font-size:14px">"""
                  + "".join(f"<div data-gt='{gid}'><div style='height:190px;border-radius:12px;background:{c}'></div><b>{t}</b><div style='color:#666'>{ch}</div></div>"
                            for gid, c, t, ch in _VIDEOS) + "</div>")

BIOS = _page("""
<div style="background:#0000aa;color:#fff;font:16px 'Courier New',monospace;height:768px;padding:0">
 <div style="background:#00aaaa;color:#000;text-align:center;padding:4px">Aptio Setup Utility - Copyright (C) 2026 American Megatrends</div>
 <div style="display:flex;gap:0;background:#00aaaa;color:#000">
  <span data-gt="main_tab" style="padding:4px 16px;background:#fff">Main</span><span data-gt="advanced_tab" style="padding:4px 16px">Advanced</span>
  <span data-gt="boot_tab" style="padding:4px 16px">Boot</span><span data-gt="security_tab" style="padding:4px 16px">Security</span>
  <span data-gt="exit_tab" style="padding:4px 16px">Save &amp; Exit</span></div>
 <div style="display:flex;padding:20px;gap:40px"><div style="flex:2;line-height:2">
  BIOS Vendor &nbsp;&nbsp;&nbsp;&nbsp;American Megatrends<br>Version &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.21.1278<br>
  Processor &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AMD Ryzen 7 7840U<br>CPU Temperature &nbsp;54 C<br>Total Memory &nbsp;&nbsp;&nbsp;32768 MB<br>
  System Date &nbsp;&nbsp;&nbsp;&nbsp;[Sat 10/17/2026]<br>System Time &nbsp;&nbsp;&nbsp;&nbsp;[10:42:13]<br><br>
  Boot Option #1 &nbsp;[Windows Boot Manager]<br>Boot Option #2 &nbsp;[UEFI: SanDisk USB]</div>
 <div style="flex:1;border-left:1px solid #fff;padding-left:16px;line-height:1.8">&#8594;&#8592;: Select Screen<br>&#8593;&#8595;: Select Item<br>Enter: Select<br>+/-: Change Opt.<br>F9: Optimized Defaults<br>F10: Save &amp; Exit<br>ESC: Exit</div></div></div>""")

_APPS = [("Phone", "#34a853"), ("Messages", "#1a73e8"), ("WhatsApp", "#25d366"), ("Camera", "#555"),
         ("Chrome", "#db4437"), ("YouTube", "#ff0000"), ("Settings", "#777"), ("Calendar", "#4285f4"),
         ("Gmail", "#ea4335"), ("Maps", "#0f9d58"), ("Spotify", "#1db954"), ("Photos", "#fbbc04")]
PHONE = _page("""
<div style="height:915px;background:linear-gradient(#3a1c71,#d76d77,#ffaf7b);color:#fff;font-size:13px">
 <div style="display:flex;justify-content:space-between;padding:8px 16px"><b>10:42</b><span>&#9650; 5G &nbsp;12% &#128267;</span></div>
 <div style="font-size:56px;text-align:center;margin:40px 0 8px">10:42</div><div style="text-align:center">Saturday, October 17</div>
 <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:28px 0;padding:60px 16px;text-align:center">"""
                + "".join(f"<div data-gt='{n.lower()}'><div style='width:58px;height:58px;margin:auto;border-radius:16px;background:{c}'></div>{n}</div>"
                          for n, c in _APPS) + "</div></div>", "width:412px")

BLENDER = _page("""
<div style="background:#303030;color:#ddd;font-size:13px;height:800px">
 <div style="display:flex;gap:16px;padding:4px 10px;background:#232323">
  <span style="color:#ea7600">&#9673;</span><span>File</span><span>Edit</span><span data-gt="render_menu">Render</span><span>Window</span><span>Help</span>
  <span style="margin-left:30px;background:#4a4a4a;padding:0 10px">Layout</span><span data-gt="modeling_ws">Modeling</span><span>Sculpting</span><span>Shading</span><span data-gt="scripting_ws">Scripting</span></div>
 <div style="display:flex;height:770px">
  <div style="flex:1;background:#3d3d3d;position:relative">
   <div style="position:absolute;left:8px;top:8px;display:flex;flex-direction:column;gap:4px">"""
                  + "".join(f"<div style='width:28px;height:28px;background:#555;border-radius:4px'></div>" for _ in range(6)) + """</div>
   <div data-gt="cube" style="position:absolute;left:40%;top:38%;width:140px;height:140px;background:#9a9a9a;border:2px solid #ffa000;transform:skewY(-12deg)"></div>
   <div style="position:absolute;left:0;right:0;top:55%;height:1px;background:#a33"></div></div>
  <div style="width:260px;background:#282828;padding:8px;line-height:2">Scene Collection<br>&nbsp;&#9656; Collection<br>
   &nbsp;&nbsp;&nbsp;<span data-gt="outliner_camera">Camera</span><br>&nbsp;&nbsp;&nbsp;<span data-gt="outliner_cube" style="color:#ffa000">Cube</span><br>&nbsp;&nbsp;&nbsp;Light</div></div></div>""")

SCREENS: dict[str, tuple[str, dict[str, int]]] = {
    "desktop": (DESKTOP, {"width": 1280, "height": 800}),
    "inbox": (INBOX, {"width": 1280, "height": 800}),
    "youtube": (YOUTUBE, {"width": 1280, "height": 800}),
    "bios": (BIOS, {"width": 1024, "height": 768}),
    "phone": (PHONE, {"width": 412, "height": 915}),
    "blender": (BLENDER, {"width": 1280, "height": 800}),
}


def render_all(out: Path) -> dict[str, dict[str, list[float]]]:
    """Render every screen to `out/<name>.jpg`; return normalized gt boxes."""
    from playwright.sync_api import sync_playwright

    out.mkdir(parents=True, exist_ok=True)
    boxes: dict[str, dict[str, list[float]]] = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for name, (html, viewport) in SCREENS.items():
            page = browser.new_page(viewport=viewport, device_scale_factor=1)
            page.set_content(html)
            page.screenshot(path=str(out / f"{name}.jpg"), type="jpeg", quality=90)
            w, h = viewport["width"], viewport["height"]
            boxes[name] = {
                el.get_attribute("data-gt"): [b["x"] / w, b["y"] / h, (b["x"] + b["width"]) / w, (b["y"] + b["height"]) / h]
                for el in page.query_selector_all("[data-gt]")
                if (b := el.bounding_box())
            }
            page.close()
        browser.close()
    (out / "boxes.json").write_text(json.dumps(boxes, indent=1))
    return boxes


if __name__ == "__main__":
    render_all(Path("work/arch/screens"))
