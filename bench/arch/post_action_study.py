"""Read-only comparison: separate post-action completion vs batched next-step questions.

No HID and no production change. Uses the current JevOnlyPlanner's exact question bundle.
Run: python -m bench.arch.post_action_study
"""
import json
import statistics
import time
from pathlib import Path
from playwright.sync_api import sync_playwright
from brain.agent.executor import StepContext
from brain.agent.jev import JevVision, probability
from brain.agent.jev_only import JevOnlyPlanner
from brain.agent.memory import Memory
from brain.agent.race import DIRECT_GATE
from brain.cancel import CancelToken
from brain.config import Config
from brain.telemetry import Telemetry
from brain.types import DeviceProfile, Rect
from bench.arch.launch_check_study import window

OUT = Path('work/post-action-study')
CASES = [
 ('calculation_done', 'calculate 12 times 9', ["type_text('12*9=')"], True, False,
  window('Calculator', '<h1>12 × 9 = 108</h1><p>7 8 9 ÷</p><p>4 5 6 ×</p><p>1 2 3 −</p><p>0 . = +</p>')),
 ('calculation_missing', 'calculate 12 times 9', [], False, False,
  window('Calculator', '<h1>0</h1><p>7 8 9 ÷</p><p>4 5 6 ×</p><p>1 2 3 −</p><p>0 . = +</p>')),
 ('search_unsent', 'search for cats', ["type_text('cats')"], False, False,
  window('Firefox — Google', '<h1>Google</h1><input value="cats"><button>Google Search</button>')),
 ('send_unapproved', 'send an email saying hello to Alex', ["type_text('hello')"], False, True,
  window('Mail — New message', '<p>To: Alex</p><p>Subject: Hello</p><textarea>hello</textarea><button>Send</button>')),
]

class Captured(Exception):
    pass

class Questions:
    def evaluate(self, image, questions, **kwargs):
        self.questions, self.kwargs = questions, kwargs
        raise Captured


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg=Config.load(); jev=JevVision(cfg.typesafe_api_key,CancelToken(),OUT/'calls.jsonl',model=cfg.jev_model)
    rows=[]
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(); page=browser.new_page(viewport={'width':1280,'height':800})
            for repeat in range(2):
                for name,goal,history,expected,approval,html in CASES:
                    page.set_content(html); shot=page.screenshot(type='jpeg')
                    ctx=StepContext(goal,DeviceProfile('linux',Rect(0,0,1280,800),'mouse',1),Memory(OUT/'memory.json'),history,2,8,run_id=name)
                    captured=Questions(); planner=JevOnlyPlanner(captured,lambda ctx:False)
                    try: planner.next_actions(shot,ctx)
                    except Captured: pass
                    for mode in (('separate','batched') if repeat==0 else ('batched','separate')):
                        telemetry=Telemetry(Path('bench/results.csv'),'post-action-'+mode); run=telemetry.new_run()
                        start=time.monotonic(); answers={}; calls=0
                        if mode=='separate':
                            score=jev.completion(shot,goal,history,run_id=run,step=2); calls+=1
                        if mode=='batched' or score < DIRECT_GATE:
                            answers,*_=jev.evaluate(shot,captured.questions,state=captured.kwargs['state'],run_id=run,step=2); calls+=1
                            if mode=='batched': score=probability(answers['complete'].get('noul'))
                        done=score>=DIRECT_GATE
                        held=not done and probability(answers.get('irreversible',{}).get('noul',0))>=.5
                        correct=done==expected and held==approval
                        elapsed=time.monotonic()-start
                        telemetry.step(run,'synthetic',name,2,t_capture=0,t_upload=0,t_model=elapsed,t_execute=0,t_settle=0)
                        telemetry.finish(run,'synthetic',name,2,correct,elapsed)
                        row=dict(case=name,repeat=repeat,mode=mode,seconds=elapsed,calls=calls,complete=score,held=held,correct=correct,next=answers.get('next',{}).get('choice'))
                        rows.append(row); (OUT/'results.json').write_text(json.dumps(rows,indent=2)); print(json.dumps(row),flush=True)
            browser.close()
    finally: jev.close()
    for mode in ('separate','batched'):
        selected=[r for r in rows if r['mode']==mode]
        print(mode,'correct',sum(r['correct'] for r in selected),'/',len(selected),'calls',sum(r['calls'] for r in selected),'median',statistics.median(r['seconds'] for r in selected),flush=True)

if __name__=='__main__': main()
