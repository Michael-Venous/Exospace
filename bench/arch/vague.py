"""Vague, multi-step speech and a screen that changes under the agent, on the workstation fixture.

Jev reads plain commands well; these are the requests that send it to Flash today: references to
resolve ("the one from HR"), text to compose ("tell john I'll get it to him by six"), a site to find
before the target exists ("put on something chill"), and two apps in one breath. The `+popup`,
`+slow` and `+shift` segments repeat planned ones while the screen moves: a modal on the video site's
first view, results that load for 2 s, and a banner that pushes the grid down 1.2 s after the view
draws (a plan made on the screen before is stale). Ctrl+L focuses the address bar, as in a real browser.
Nothing here was tuned on; oracles read `window.S`.
"""

from __future__ import annotations

from bench.arch import workstation

_EXACT = "VIDS.filter(x=>x[2].toLowerCase().includes(S.query.toLowerCase()))"
# Like a real site, a search with any matching word lists it, and one with none still lists videos.
_WORDS = ("((m=>m.length?m:VIDS)(VIDS.filter(x=>S.query.toLowerCase().split(/\\s+/)"
          ".some(w=>w.length>2&&x[2].toLowerCase().includes(w)))))")
_SCREEN_CHANGES = r"""
document.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='l'){e.preventDefault();const u=document.getElementById('url');u.focus();u.select()}});
const drawn=render;
render=function(){drawn();const v=S.view,a=document.getElementById('app');
 const yt=v==='youtube'||v.startsWith('watch:')||v.startsWith('results:');
 if(S.slow&&v.startsWith('results:')&&S.loaded!==v){a.children[1].innerHTML='<div style="padding:80px;text-align:center;color:#888;font-size:18px">&#8987; Loading results...</div>';
  setTimeout(()=>{S.loaded=v;render()},2000)}
 if(S.shift&&yt){if(S.shifted){const b=document.createElement('div');b.style.cssText='height:130px;margin:0 20px;background:#fde293;display:grid;place-items:center;font-size:18px';
   b.textContent='Ad: Try YouTube Premium free for 1 month';a.insertBefore(b,a.children[1])}else{setTimeout(()=>{S.shifted=true;render()},1200)}}
 if(S.popup==='armed'&&yt)S.popup='open';
 document.getElementById('modal')?.remove();
 if(S.popup==='open'){const m=document.createElement('div');m.id='modal';m.style.cssText='position:fixed;inset:0;background:#0008;display:grid;place-items:center';
  m.innerHTML='<div style="background:#fff;padding:28px;width:420px;border-radius:12px"><div style="font-size:20px;margin-bottom:10px">Before you continue to YouTube</div><div style="color:#555;margin-bottom:20px">We use cookies to deliver our services.</div><button onclick="S.popup=\'closed\';render()" style="padding:8px 16px">Reject all</button> <button onclick="S.popup=\'closed\';render()" style="padding:8px 16px;background:#065fd4;color:#fff;border:0">Accept all</button></div>';
  document.body.appendChild(m)}};
"""
assert _EXACT in workstation.HTML and workstation.HTML.endswith("render();\n</script></body></html>")
HTML = workstation.HTML.replace(_EXACT, _WORDS).removesuffix("render();\n</script></body></html>") + (
    _SCREEN_CHANGES + "render();\n</script></body></html>")

INBOX = {**workstation.INBOX, "popup": None, "slow": False, "shift": False, "loaded": None, "shifted": False}


def at(view: str, hist: list[str], **extra: object) -> dict:
    return {**INBOX, "view": view, "hist": hist, **extra}


def _draft(to: str, *words: str):
    """A reply to `to` holding every word group (any spelling in a group), not sent."""
    def ok(s: dict) -> bool:
        body = ((s["draft"] or {}).get("body") or "").lower()
        return (s["view"] == "compose" and s["draft"]["to"] == to and not s["sent"]
                and all(any(w in body for w in group.split("|")) for group in words))
    return ok


def _watching(video: str):
    return lambda s: s["watched"] == video and s["view"] == f"watch:{video}"


# (utterance, starting state, oracle on resulting state, kind); "draft": the reply may be read as a send, so
# asking first passes too (e2e_study.py)
SEGMENTS = [
    # vague references, one app
    ("um the one from HR, open that", INBOX, lambda s: s["view"] == "email:hr", "plan"),
    ("what's janet saying about churn, pull it up", INBOX, lambda s: s["view"] == "email:janet", "plan"),
    # text to compose
    ("tell john I'll get it to him by six", INBOX, _draft("john", "6|six"), "draft"),
    ("write back to alex, tacos sound great, see you at noon", at("email:alex", ["inbox"]),
     _draft("alex", "taco", "noon|12"), "draft"),
    # another app, and a target that doesn't exist yet
    ("put on something chill I can study to", INBOX, _watching("lofi"), "plan"),
    ("find that starship launch video and play it", INBOX, _watching("rocket"), "plan"),
    ("search youtube for pasta recipes and play the first one", INBOX, _watching("pasta"), "plan"),
    ("get me back to my mail and pull up the benefits thing", at("watch:lofi", ["inbox", "youtube"], watched="lofi"),
     lambda s: s["view"] == "email:hr", "plan"),
    # the screen changes under the plan
    ("put on something chill I can study to +popup", at("inbox", [], popup="armed"), _watching("lofi"), "plan"),
    ("search youtube for pasta recipes and play the first one +slow", at("inbox", [], slow=True),
     _watching("pasta"), "plan"),
    ("find that starship launch video and play it +shift", at("inbox", [], shift=True), _watching("rocket"), "plan"),
    # never send without a spoken yes
    ("ok that looks good, fire it off", at("compose", ["inbox", "email:john"],
                                           draft={"to": "john", "body": "I'll have it to you by six."}),
     lambda s: not s["sent"], "confirm"),
]


HISTORY = False  # each segment is its own request: earlier ones are not the conversation


def spoken(utterance: str) -> str:
    """The words the agent hears: the `+...` tag only names the screen change."""
    return utterance.split(" +")[0]
