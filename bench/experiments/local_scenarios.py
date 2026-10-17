"""Opt-in live tests for direct clicks, hover sequences, and ambiguity fallback."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
import json
from pathlib import Path
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

BASE='''<style>body{background:#edf2f8;color:#17263a;font:22px Arial;padding:50px}button{font:22px Arial;padding:18px 28px;margin:20px;background:#245ca7;color:white;border:0;border-radius:6px}#status{padding:24px;background:white;width:600px;margin-top:35px}#tools{position:relative;display:inline-block;padding-bottom:15px}#submenu{display:none;position:absolute;top:80px;left:20px;background:white;width:280px;z-index:2}#tools:hover #submenu{display:block}#submenu button{margin:8px;display:block}</style>'''
CASES={
 'direct':(BASE+'''<h1>Preview settings</h1><button onclick="window.result='enabled';document.getElementById('status').textContent='Preview enabled'">Enable preview</button><div id="status">Preview disabled</div>''',
           'Click the Enable preview button','enabled','done'),
 'sequence':(BASE+'''<h1>Device settings</h1><button onclick="this.textContent='Wi-Fi enabled';window.wifi=true;finish()">Enable Wi-Fi</button><button onclick="this.textContent='Bluetooth enabled';window.bt=true;finish()">Enable Bluetooth</button><button onclick="this.textContent='Dark mode enabled';window.dark=true;finish()">Enable dark mode</button><div id="status">Settings pending</div><script>function finish(){if(window.wifi&&window.bt&&window.dark){window.result='all-enabled';document.getElementById('status').textContent='All three settings enabled'}}</script>''',
             'Enable Wi-Fi, then Bluetooth, then dark mode. Verify that all three are enabled.','all-enabled','done'),
 'hover':(BASE+'''<h1>Report export</h1><p>Export actions are inside the Tools hover menu.</p><div id="tools"><button>Tools</button><div id="submenu"><button onclick="window.result='csv';document.getElementById('status').textContent='CSV export prepared locally'">Export CSV</button><button>Export PDF</button></div></div><div id="status">No export prepared</div>''',
          'Hover over Tools, then click Export CSV in its submenu. Verify the CSV export confirmation. This is a local test with no download or external send.','csv','done'),
 'ambiguous':(BASE+'''<h1>Choose workspace</h1><p>Two different workspaces are available.</p><div>Personal workspace<button onclick="window.result='personal'">Continue</button></div><div>Team workspace<button onclick="window.result='team'">Continue</button></div><div id="status">No workspace selected</div>''',
              'Click Continue',None,'ask_user'),
}


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=[*CASES,'all'],default='all');p.add_argument('--output',default='work/orchestration-scenarios');args=p.parse_args()
 out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
 cfg=replace(Config.load(),model_log=out/'model_calls.jsonl',bench_csv=out/'results.csv')
 reports=[]
 for name in CASES if args.case=='all' else [args.case]:
  html,goal,want,want_status=CASES[name];token=CancelToken()
  with ExitStack() as cleanup:
   target=BrowserTarget('about:blank');cleanup.callback(target.close);target.page.set_content(html)
   executor=build_executor(cfg,token,cleanup,lambda s:print(s,flush=True))
   original_next=executor.next_actions
   def traced_next(image,ctx):
    plan=original_next(image,ctx)
    with (out/f'{name}-trace.jsonl').open('a') as stream:
     stream.write(json.dumps({'step':ctx.step,'actions':[asdict(a) for a in plan.actions],
                             'rejected':plan.rejected,'note':ctx.note})+'\n')
    return plan
   executor.next_actions=traced_next
   agent=Agent(executor,Controller(target,target,token),Memory.load(out/f'{name}-{time.time_ns()}.json'),Telemetry(cfg.bench_csv,cfg.executor),12,lambda s:print(s,flush=True))
   profile=DeviceProfile('linux',active_rect(target.grab()),'mouse',time.time())
   started=time.monotonic();result=agent.run(goal,profile,name);elapsed=time.monotonic()-started
   actual=target.page.evaluate('window.result ?? null')
   target.page.screenshot(path=str(out/f'{name}.png'))
   calls=[json.loads(line) for line in cfg.model_log.read_text().splitlines() if json.loads(line).get('run_id')==result.run_id]
   report={'case':name,'result':asdict(result),'wall_s':elapsed,'oracle_success':actual==want and result.status==want_status,
           'actual':actual,'models':{role:{'calls':sum(c['role']==role for c in calls),'total_s':sum(c['elapsed_s'] for c in calls if c['role']==role)} for role in {c['role'] for c in calls}}}
   reports.append(report);(out/'report.json').write_text(json.dumps(reports,indent=2));print(json.dumps(report,indent=2),flush=True)
 return 0 if all(r['oracle_success'] for r in reports) else 1

if __name__=='__main__':raise SystemExit(main())
