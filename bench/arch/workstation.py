"""Interactive workstation fixture: mail + video site with real navigation.

State lives in `window.S`; `window.load(state)` restores any state so each
spoken segment can be evaluated independently from a known starting point.
Models only ever see screenshots.
"""

from __future__ import annotations

HTML = r"""<!doctype html><html><body style="margin:0;font-family:system-ui,sans-serif;font-size:14px">
<div style="height:72px;background:#dee1e6;display:flex;flex-direction:column">
 <div style="height:34px;display:flex;align-items:flex-end;padding-left:8px">
  <div id="tab" style="background:#fff;border-radius:8px 8px 0 0;padding:7px 14px;width:200px"></div></div>
 <div style="height:38px;background:#fff;display:flex;align-items:center;gap:6px;padding:0 8px">
  <button id="back" onclick="back()" style="width:30px;height:30px;border:0;border-radius:50%;background:none;font-size:20px">&#8592;</button>
  <span style="width:30px;text-align:center;color:#999;font-size:20px">&#8594;</span>
  <input id="url" style="flex:1;background:#f1f3f4;border:0;border-radius:16px;padding:7px 14px"
    onkeydown="if(event.key==='Enter'){navUrl(this.value)}"></div></div>
<div id="app"></div>
<script>
const MAIL={john:{from:'John Miller',subj:'Need the Q3 report today',body:"Hey, can you send me the Q3 report before 5? Leadership wants it for tomorrow's meeting. Thanks, John"},
 janet:{from:'Janet Park',subj:'Q3 report numbers + notes',body:'Revenue was $4.2M (up 12%). Churn fell to 3.1%. Notes: lead with growth, mention the new EU customers.'},
 hr:{from:'HR Team',subj:'Benefits enrollment closes Friday',body:'Reminder to complete your enrollment.'},
 alex:{from:'Alex Chen',subj:'Lunch tomorrow?',body:'Thinking tacos at noon.'}};
const VIDS=[['lofi','#6b4f9e','lofi hip hop radio - beats to relax/study to'],['pasta','#c0392b','Fresh pasta from scratch in 10 minutes'],
 ['rocket','#2c3e50','Starship flight 9 full launch replay'],['cat','#e67e22','Cats that are bad at their jobs'],
 ['python','#2980b9','Python in 100 seconds'],['guitar','#16a085','Learn 4 chords to play 100 songs']];
window.S={view:'inbox',hist:[],sent:[],draft:null,query:'',watched:null};
function go(v){S.hist.push(S.view);S.view=v;render()}
function back(){if(S.hist.length){S.view=S.hist.pop();render()}}
function navUrl(u){u=u.toLowerCase();if(u.includes('youtube'))go('youtube');else if(u.includes('outlook')||u.includes('mail'))go('inbox')}
function load(s){window.S=JSON.parse(JSON.stringify(s));render()}
function render(){const a=document.getElementById('app'),v=S.view;
 const yt=v==='youtube'||v.startsWith('watch:')||v.startsWith('results:');
 document.getElementById('tab').textContent=yt?'YouTube':'Mail - Outlook';
 document.getElementById('url').value=yt?'youtube.com':'outlook.office.com/mail/inbox';
 if(!yt){const list=Object.entries(MAIL).map(([k,m])=>`<div onclick="go('email:${k}')" style="padding:12px;border-bottom:1px solid #eee;cursor:pointer;${v==='email:'+k?'background:#deecf9':''}"><b>${m.from}</b><div><b>${m.subj}</b></div><div style="color:#666">${m.body.slice(0,48)}...</div></div>`).join('');
  let pane='<div style="color:#888;padding:40px">Select an item to read</div>';
  if(v.startsWith('email:')){const k=v.slice(6),m=MAIL[k];pane=`<div style="font-size:20px">${m.subj}</div><div style="color:#666;margin:6px 0 16px">${m.from}</div><div>${m.body}</div><div style="margin-top:20px"><button onclick="S.draft={to:'${k}',body:''};go('compose')">Reply</button> <button>Forward</button></div>`}
  if(v==='compose'){pane=`<div style="font-size:18px">Reply to ${MAIL[S.draft.to].from}</div><textarea id="body" oninput="S.draft.body=this.value" style="width:90%;height:260px;margin:10px 0">${S.draft.body}</textarea><div><button onclick="S.sent.push({...S.draft});S.draft=null;go('sent')" style="background:#0f6cbd;color:#fff;border:0;padding:8px 18px">Send</button> <button onclick="S.draft=null;back()">Discard</button></div>`}
  if(v==='sent'){pane='<div style="font-size:18px;padding:40px">Message sent</div>'}
  a.innerHTML=`<div style="display:flex;height:728px"><div style="width:200px;background:#f3f2f1;padding:12px"><div style="background:#0f6cbd;color:#fff;padding:8px;text-align:center">New mail</div><div style="padding:6px">Inbox</div><div style="padding:6px">Sent Items</div></div><div style="width:420px;border-right:1px solid #ddd">${list}</div><div style="flex:1;padding:20px">${pane}</div></div>`;return}
 const vids=(v.startsWith('results:')?VIDS.filter(x=>x[2].toLowerCase().includes(S.query.toLowerCase())):VIDS);
 let main=`<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:18px;padding:10px 20px">${vids.map(([k,c,t])=>`<div onclick="S.watched='${k}';go('watch:${k}')" style="cursor:pointer"><div style="height:170px;border-radius:12px;background:${c}"></div><b>${t}</b></div>`).join('')}</div>`;
 if(v.startsWith('watch:')){const x=VIDS.find(y=>y[0]===v.slice(6));main=`<div style="padding:20px"><div style="height:420px;width:750px;background:${x[1]};position:relative;color:#fff"><div style="position:absolute;left:12px;right:12px;bottom:34px;height:4px;background:#ffffff55"><div style="width:9%;height:4px;background:#f00"></div></div><div style="position:absolute;left:12px;bottom:8px;font-size:15px">&#10074;&#10074; &nbsp; 0:21 / 3:58:12 &nbsp; LIVE</div></div><div style="font-size:20px;margin-top:10px">${x[2]}</div><div style="color:#666">Now playing</div></div>`}
 a.innerHTML=`<div style="padding:10px 20px;display:flex;align-items:center;gap:20px"><b style="font-size:20px;color:#f00">&#9654; YouTube</b><input id="q" placeholder="Search" value="${S.query}" style="flex:1;max-width:520px;border:1px solid #ccc;border-radius:20px;padding:8px 16px" onkeydown="if(event.key==='Enter'){S.query=this.value;go('results:'+this.value)}"></div>${main}`}
render();
</script></body></html>"""

INBOX = {"view": "inbox", "hist": [], "sent": [], "draft": None, "query": "", "watched": None}


def at(view: str, hist: list[str], **extra: object) -> dict:
    return {**INBOX, "view": view, "hist": hist, **extra}


# (utterance, starting state, oracle on resulting state, kind)
SEGMENTS = [
    ("open janet's email about the report numbers", INBOX,
     lambda s: s["view"] == "email:janet", "act"),
    ("oh john is asking for a report", at("email:janet", ["inbox"]),
     lambda s: s["view"] == "email:janet" and not s["sent"], "none"),
    ("actually go back", at("email:janet", ["inbox"]),
     lambda s: s["view"] == "inbox", "act"),
    ("open john's email", INBOX,
     lambda s: s["view"] == "email:john", "act"),
    ("reply saying I'll have the report to him by 5pm", at("email:john", ["inbox"]),
     lambda s: s["view"] == "compose" and "5" in (s["draft"] or {}).get("body", "") and not s["sent"], "plan"),
    ("send it", at("compose", ["inbox", "email:john"], draft={"to": "john", "body": "I'll have the report to you by 5pm."}),
     lambda s: len(s["sent"]) == 1 and s["sent"][0]["to"] == "john", "act"),
    ("go to youtube", INBOX,
     lambda s: s["view"] == "youtube", "plan"),
    ("search for lofi", at("youtube", ["inbox"]),
     lambda s: s["view"].startswith("results:") and "lofi" in s["query"].lower(), "plan"),
    ("play the first video", at("youtube", ["inbox"]),
     lambda s: s["watched"] == "lofi", "act"),
    ("nice, thanks", at("watch:lofi", ["inbox", "youtube"], watched="lofi"),
     lambda s: s["view"] == "watch:lofi", "none"),
]
