"""Compare Lite and Flash on verified small browser workflows, with real Jev.

All pages and submissions are synthetic and local. The URL task is an in-page
navigation form, not Chrome's native address bar or the real YouTube site.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import statistics
import time

from brain.agent.factory import build_executor
from brain.agent.jev_flash import PROMPT
from brain.agent.loop import Agent
from brain.agent.memory import Memory
from brain.browser import BrowserTarget
from brain.cancel import CancelToken
from brain.capture import active_rect
from brain.config import Config
from brain.controller import Controller
from brain.telemetry import Telemetry
from brain.types import DeviceProfile

STYLE='''<style>body{font:24px Arial;background:#edf2f8;color:#18283b;padding:35px}
h1{font-size:34px}button,a{font:24px Arial;padding:16px 24px;margin:15px 0;display:block}
button{background:#245ca7;color:white;border:0;border-radius:6px}input{display:block;
font:24px Arial;padding:12px;width:650px;margin:8px 0 20px;border:2px solid #899aaf}
label{display:block}#status{padding:25px;background:white;max-width:850px;margin-top:25px;line-height:1.6}
</style>'''
CASES={
    'two_clicks': (
        STYLE+'''<h1>Local document catalog</h1><div id="start"><button onclick="document.getElementById('start').hidden=true;document.getElementById('results').hidden=false">Open catalog</button></div>
        <div id="results" hidden><h2>Catalog results</h2><a href="#" onclick="window.result='alpha';document.getElementById('results').hidden=true;document.getElementById('status').textContent='Alpha guide opened successfully';return false">Alpha guide</a>
        <a href="#" onclick="window.result='beta';return false">Beta guide</a></div><div id="status">No document opened</div>''',
        'Open the catalog, then click the first result. Verify that its guide opened.', 'alpha'),
    'type_url': (
        STYLE+'''<h1>Local website launcher test</h1><p>This form simulates navigation locally. No external website will be opened.</p>
        <form onsubmit="event.preventDefault();window.result=document.getElementById('url').value;document.getElementById('form').hidden=true;document.getElementById('status').textContent='Local destination opened: '+window.result" id="form">
        <label for="url">Website address</label><input id="url" autocomplete="off"><button>Open locally</button></form><div id="status">Awaiting website address</div>''',
        'Enter youtube.com in Website address, then choose Open locally. Verify the displayed destination.', 'youtube.com'),
    'three_fields': (
        STYLE+'''<h1>Local project card</h1><form id="form" onsubmit="event.preventDefault();window.result={};for(const id of ['name','project','note'])window.result[id]=document.getElementById(id).value;document.getElementById('form').hidden=true;document.getElementById('status').textContent='Card saved locally. Name: '+window.result.name+'. Project: '+window.result.project+'. Note: '+window.result.note">
        <label for="name">Name</label><input id="name" autocomplete="off"><label for="project">Project</label><input id="project" autocomplete="off"><label for="note">Note</label><input id="note" autocomplete="off"><button>Save card</button></form><div id="status">No card saved</div>''',
        'Fill Name with Morgan, Project with Exospace, and Note with Test the prototype. Save the card and verify the saved values.',
        {'name':'Morgan','project':'Exospace','note':'Test the prototype'}),
}
MODELS={'lite':'gemini-3.5-flash-lite','flash':'gemini-3.8-flash'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--output',default='work/sequence-comparison-v1')
    parser.add_argument('--case',choices=[*CASES,'all'],default='all')
    parser.add_argument('--models',choices=list(MODELS),nargs='+',default=list(MODELS))
    args=parser.parse_args()
    if not 1<=args.repeats<=5: parser.error('repeats must be 1-5')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    manifest={'models':{m:MODELS[m] for m in args.models},'thinking':'low',
              'repeats':args.repeats,'case':args.case,
              'semantic_prompt_sha256':sha256(PROMPT.encode()).hexdigest(),
              'fixtures_sha256':sha256(json.dumps(CASES,sort_keys=True).encode()).hexdigest(),
              'scope':'Synthetic isolated browser workflows; URL is an in-page local form, not native address bar.',
              'timing':'Agent run through verified completion, including Jev, model calls, retries, input and settle.'}
    manifest_path=out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise ValueError('Benchmark configuration changed; choose a fresh output directory')
    manifest_path.write_text(json.dumps(manifest,indent=2))
    report_path=out/'report.json'
    reports=json.loads(report_path.read_text()) if report_path.exists() else []
    done={(r['case'],r['model'],r['repeat']) for r in reports}
    base=Config.load()
    cases=CASES if args.case=='all' else {args.case:CASES[args.case]}
    for repeat in range(args.repeats):
        for case,(html,goal,expected) in cases.items():
            for model in args.models[::1 if repeat%2==0 else -1]:
                if (case,model,repeat) in done: continue
                folder=out/f'{case}-{model}-{repeat}';folder.mkdir(exist_ok=True)
                cfg=replace(base,gemini_model=MODELS[model],gemini_thinking='low',
                            model_log=folder/'model_calls.jsonl',bench_csv=folder/'results.csv')
                token=CancelToken()
                with ExitStack() as cleanup:
                    target=BrowserTarget('about:blank');cleanup.callback(target.close)
                    target.page.set_content(html)
                    target.page.screenshot(path=str(folder/'before.png'))
                    executor=build_executor(cfg,token,cleanup,lambda s:None)
                    generate=executor.model.generate
                    def recorded_generate(*a,**kw):
                        response=generate(*a,**kw)
                        with (folder/'semantic_plans.jsonl').open('a') as f:
                            f.write(json.dumps({'step':kw.get('step'),'response':response})+'\n')
                        return response
                    executor.model.generate=recorded_generate
                    assess=executor.jev.assess
                    def recorded_assess(image,checks,targets,**kw):
                        values,points=assess(image,checks,targets,**kw)
                        with (folder/'jev_checks.jsonl').open('a') as f:
                            f.write(json.dumps({'step':kw.get('step'),'checks':checks,'targets':targets,
                                                'values':values,'points':points})+'\n')
                        return values,points
                    executor.jev.assess=recorded_assess
                    old_next=executor.next_actions
                    def traced(image,ctx):
                        plan=old_next(image,ctx)
                        (folder/f'step-{ctx.step}.jpg').write_bytes(image)
                        with (folder/'trace.jsonl').open('a') as f:
                            f.write(json.dumps({'step':ctx.step,'actions':[asdict(a) for a in plan.actions],
                                                'rejected':plan.rejected,'note':ctx.note})+'\n')
                        return plan
                    executor.next_actions=traced
                    events=[];old_send=target.send
                    def send(command):
                        if command['op'] not in {'ping','release_all'}:
                            events.append({'elapsed_s':time.monotonic()-started,'op':command['op']})
                        old_send(command)
                    target.send=send
                    agent=Agent(executor,Controller(target,target,token),Memory.load(folder/f'memory-{time.time_ns()}.json'),
                                Telemetry(cfg.bench_csv,cfg.executor),12,lambda s:None)
                    profile=DeviceProfile('linux',active_rect(target.grab()),'mouse',time.time())
                    print(f'START {case} {model} repeat={repeat+1}',flush=True)
                    started=time.monotonic();result=agent.run(goal,profile,case);elapsed=time.monotonic()-started
                    actual=target.page.evaluate('window.result ?? null')
                    target.page.screenshot(path=str(folder/'after.png'))
                    calls=[json.loads(line) for line in cfg.model_log.read_text().splitlines()]
                    row={'case':case,'model':model,'model_id':MODELS[model],'thinking':'low','repeat':repeat,
                         'success':actual==expected and result.status=='done','actual':actual,'expected':expected,
                         'wall_s':elapsed,'first_input_s':events[0]['elapsed_s'] if events else None,
                         'result':asdict(result),'input_events':events,
                         'calls':{role:{'n':sum(c['role']==role for c in calls),
                                        'total_s':sum(c['elapsed_s'] for c in calls if c['role']==role)} for role in {c['role'] for c in calls}}}
                    reports.append(row);report_path.write_text(json.dumps(reports,indent=2))
                    print(json.dumps({k:v for k,v in row.items() if k not in {'input_events','result'}}),flush=True)
    summary={case:{model:{'passed':sum(r['success'] for r in reports if r['case']==case and r['model']==model),
                         'n':sum(r['case']==case and r['model']==model for r in reports),
                         'median_s':statistics.median(r['wall_s'] for r in reports if r['case']==case and r['model']==model)}
                   for model in args.models} for case in cases}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
