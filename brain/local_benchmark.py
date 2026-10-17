"""Live screenshot-only browser task with a separate DOM outcome oracle.

Run: python -m brain.local_benchmark --runs 3
The models see only screenshots and goal/history, never fixture HTML or DOM.
All event data is synthetic and saved only in the isolated browser's local state.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics
import time

from brain.agent.factory import build_executor
from brain.agent.loop import Agent
from brain.agent.memory import Memory
from brain.browser import BrowserTarget
from brain.cancel import CancelToken
from brain.capture import active_rect
from brain.config import Config
from brain.controller import Controller
from brain.telemetry import Telemetry
from brain.types import DeviceProfile

HTML = '''<!doctype html><html><head><meta charset="utf-8"><title>Exospace local test</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#e9eef5;color:#17263a;font:18px Arial}
header{background:#152d48;color:white;padding:22px 40px}h1{font-size:26px;margin:0}h2{font-size:23px}
main{display:flex;gap:28px;margin:26px 40px}section{background:white;border-radius:12px;padding:25px;flex:1}
label{display:block;margin:13px 0 5px;font-weight:bold}input,textarea{width:100%;padding:10px;font:18px Arial;border:2px solid #899aaf;border-radius:5px}textarea{height:74px}
button{background:#235cab;color:white;border:0;border-radius:6px;padding:13px 22px;font:18px Arial;cursor:pointer;margin-top:16px}
button.secondary{background:#53677d}.line{margin:15px 0;line-height:1.45}.pill{background:#d9e9fe;padding:8px;border-radius:5px}
#modal{display:none;position:fixed;inset:0;background:#0008;align-items:center;justify-content:center}
#modal>div{background:white;border-radius:12px;padding:32px;width:460px;box-shadow:0 10px 40px #0005}
#saved{display:none;background:#deefdf;border:2px solid #358447;padding:20px;margin-top:20px}#saved h2{margin-top:0}
</style></head><body><header><h1>Exospace • Local Calendar Test</h1></header>
<main><section id="source"><h2>Event request</h2><p class="pill">Synthetic data • local save only</p>
<p class="line"><b>Title:</b> Robotics planning session</p><p class="line"><b>Date:</b> October 21, 2026</p>
<p class="line"><b>Original time:</b> 6:00 PM</p><p class="line"><b>Location:</b> Innovation Lab, Room 204</p>
<p class="line"><b>Agenda:</b> Review the prototype, assign testing roles, and plan the demo.</p>
<p class="line"><b>Update:</b> Move the start to 7:30 PM. Allow 45 minutes.</p>
<button id="create" onclick="openForm()">Create event</button></section>
<section id="workspace"><h2>Calendar</h2><p id="empty">No event has been created yet.</p>
<form id="form" style="display:none" onsubmit="saveEvent(event)">
<label for="title">Event title</label><input id="title" required>
<label for="date">Date (YYYY-MM-DD)</label><input id="date" required>
<div style="display:flex;gap:16px"><div style="flex:1"><label for="start">Start (HH:MM, 24-hour)</label><input id="start" required></div>
<div style="flex:1"><label for="end">End (HH:MM, 24-hour)</label><input id="end" required></div></div>
<label for="location">Location</label><input id="location" required>
<label for="notes">Agenda</label><textarea id="notes" required></textarea>
<button id="save" type="submit">Save locally</button></form>
<div id="saved"></div></section></main>
<div id="modal"><div><h2>Quick tip</h2><p>You can edit an event before saving. This calendar stores test data locally.</p>
<button id="dismiss" onclick="document.getElementById('modal').style.display='none'">Got it</button></div></div>
<script>
const variant=Number(new URLSearchParams(location.search).get('variant')||0);
function openForm(){document.getElementById('form').style.display='block';document.getElementById('empty').style.display='none';
 document.getElementById('create').disabled=true;
 if(variant%2===0){document.getElementById('modal').style.display='flex';}
 if(variant%3===2){document.querySelector('main').style.flexDirection='row-reverse';}}
function saveEvent(e){e.preventDefault();window.savedEvent={};for(const id of ['title','date','start','end','location','notes'])window.savedEvent[id]=document.getElementById(id).value;
 const box=document.getElementById('saved');box.replaceChildren();const heading=document.createElement('h2');heading.textContent='Event saved locally';box.appendChild(heading);
 for(const [k,v] of Object.entries(window.savedEvent)){const p=document.createElement('p');p.textContent=k+': '+v;box.appendChild(p)}
 document.getElementById('form').style.display='none';box.style.display='block';}
</script></body></html>'''

GOAL = ('Read the event request on this page. Create and save the event locally, using '
        'the updated start time and calculating the end time from the duration. Copy '
        'the title, date, location and full agenda accurately. Dismiss any informational '
        'popup if needed. Verify the saved event summary before finishing. Do not send '
        'invitations or navigate away from this local test.')
EXPECTED = {'title':'Robotics planning session','date':'2026-10-21','start':'19:30',
            'end':'20:15','location':'Innovation Lab, Room 204',
            'notes':'Review the prototype, assign testing roles, and plan the demo.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=int,default=1)
    parser.add_argument('--headed',action='store_true')
    parser.add_argument('--output',default='work/local-benchmark')
    parser.add_argument('--max-steps',type=int,default=25)
    args=parser.parse_args()
    if not 1 <= args.runs <= 10: parser.error('--runs must be 1-10')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    fixture=out/'calendar.html';fixture.write_text(HTML)
    cfg=replace(Config.load(),model_log=out/'model_calls.jsonl',bench_csv=out/'results.csv')
    reports=[]
    for index in range(args.runs):
        token=CancelToken()
        with ExitStack() as cleanup:
            target=BrowserTarget(fixture.as_uri()+f'?variant={index}',headed=args.headed)
            cleanup.callback(target.close)
            executor=build_executor(cfg,token,cleanup,lambda s:print(s,flush=True))
            original_next = executor.next_actions
            def traced_next(image, ctx):
                plan = original_next(image, ctx)
                (out/f'run-{index}-step-{ctx.step}.jpg').write_bytes(image)
                with (out/f'trace-{index}.jsonl').open('a') as trace:
                    trace.write(json.dumps({'step':ctx.step,'note':ctx.note,
                        'actions':[asdict(a) for a in plan.actions],
                        'rejected':plan.rejected})+'\n')
                return plan
            executor.next_actions = traced_next
            ctl=Controller(target,target,token)
            frame=target.grab()
            profile=DeviceProfile('linux',active_rect(frame),'mouse',time.time())
            memory=Memory.load(out/f'session-{index}-{time.time_ns()}.json')
            agent=Agent(executor,ctl,memory,Telemetry(cfg.bench_csv,cfg.executor),
                        args.max_steps,lambda s:print(s,flush=True))
            target.page.screenshot(path=str(out/f'before-{index}.png'))
            print(f'RUN {index+1}: popup={index%2==0}, relocated={index%3==2}',flush=True)
            started=time.monotonic()
            result=agent.run(GOAL,profile,'calendar-complex')
            elapsed=time.monotonic()-started
            # Independent outcome oracle, only after the agent has finished.
            actual=target.page.evaluate('window.savedEvent || null')
            target.page.screenshot(path=str(out/f'after-{index}.png'))
            calls=[json.loads(line) for line in cfg.model_log.read_text().splitlines()
                   if json.loads(line).get('run_id')==result.run_id]
            by_role={}
            for role in sorted({c['role'] for c in calls}):
                times=[c['elapsed_s'] for c in calls if c['role']==role]
                by_role[role]={'calls':len(times),'total_s':sum(times),
                               'median_s':statistics.median(times),'max_s':max(times)}
            report={'run':index+1,'result':asdict(result),'wall_s':elapsed,
                    'oracle_success':actual==EXPECTED,'actual':actual,'expected':EXPECTED,
                    'models':by_role,'popup':index%2==0,'relocated':index%3==2}
            reports.append(report)
            (out/'report.json').write_text(json.dumps(reports,indent=2))
            print(json.dumps(report,indent=2),flush=True)
    return 0 if all(r['oracle_success'] and r['result']['status']=='done' for r in reports) else 1


if __name__=='__main__': raise SystemExit(main())
