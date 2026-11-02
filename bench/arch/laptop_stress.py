"""Disposable native multi-turn stress episodes; pixel/HID agent, independent oracles.

Uses the same laptop harness and isolated browser profiles as laptop_reliability. Never
opens personal accounts, sends messages, or changes real display settings. Partial-STT
hypotheses and physical talk-button events are replayed; microphone/STT is not measured.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import shlex
import shutil
import signal
import statistics
import threading
import time
from pathlib import Path

import httpx
from websockets.sync.client import connect

from bench.arch import laptop_tests as harness
from bench.arch.laptop import LaptopTarget
from bench.arch.laptop_reliability import output_files
from bench.arch.reliability_oracles import FileOracle


class RecordedTarget:
    """The same HID-dispatch timestamps for both architectures, including typing chunks."""
    def __init__(self, target, meta):
        self.target,self.meta=target,meta
    def __getattr__(self,name):return getattr(self.target,name)
    def grab(self):return self.target.grab()
    def send(self,command):
        self.meta['hid'].append({'t':time.monotonic(),'op':command['op'],
                                 'characters':len(command.get('text',''))})
        self.target.send(command)


def dom_state() -> dict:
    states=[]
    for tab in harness.chrome_tabs():
        try:
            with connect(tab['webSocketDebuggerUrl'],open_timeout=2) as ws:
                ws.send(json.dumps({'id':1,'method':'Runtime.evaluate','params':{'expression':
                    'JSON.stringify({path:location.pathname,visible:document.visibilityState,'
                    'value:document.querySelector("input")?.value,state:window.state||null})'}}))
                while True:
                    reply=json.loads(ws.recv(timeout=3))
                    if reply.get('id')==1:break
                value=reply.get('result',{}).get('result',{}).get('value')
                states.append(json.loads(value) if value else {'error':'unobservable page'})
        except Exception as exc:states.append({'error':type(exc).__name__})
    return {'tabs':states}


def site(root: Path):
    root.mkdir()
    style='<style>body{font:26px sans-serif;margin:60px;max-width:1250px}button,input{font:inherit;margin:18px;padding:14px}input[type=range]{width:650px}label{display:block}aside{padding:24px;background:#eef}a{margin:24px}</style>'
    pages={
      'collections.html':'<h1>Collections</h1><p>Available collections:</p><a href="birds.html">Birds</a><a href="plants.html">Plants</a>',
      'birds.html':'<h1>Birds</h1>', 'plants.html':'<h1>Plants</h1>',
      'brightness.html':'<h1>Reading controls</h1><label>Reading brightness<input type="range" min="0" max="100" value="70" oninput="document.querySelector(\'output\').textContent=this.value+\'%\'"></label><output>70%</output>',
      'address.html':"""<h1>Project address</h1><label>Address<input autocomplete="off" oninput="update(this.value)"></label><div id="choices"></div><script>window.state={selected:null,values:[]};function update(v){state.values.push(v);const c=document.querySelector('#choices');c.replaceChildren();if(v==='main-'||v==='docs/'){const b=document.createElement('button');b.textContent='Studio';b.onclick=()=>{state.selected='Studio';document.querySelector('h1').textContent='Studio';};c.append(b);}}</script>""",
      'profiles.html':'''<h1>Workspace profiles</h1><p>Select an existing profile:</p><button onclick="choose('jordan.reed')"><span style="color:orange">●</span> Jordan Reed — jordan.reed@sample.invalid</button><br><button onclick="choose('j.reed')"><span style="color:green">●</span> Jordan Reed — j.reed@sample.invalid</button><br><button onclick="choose('taylor.blue')">Taylor Blue — taylor.blue@sample.invalid</button><aside id="selected">Current workspace: Taylor Blue</aside><script>window.state={selected:'taylor.blue',choices:[]};function choose(v){state.selected=v;state.choices.push(v);document.querySelector('#selected').textContent='Current workspace: '+v;}</script>'''
    }
    for name,body in pages.items():
        (root/name).write_text('<!doctype html><meta charset="utf-8"><title>'+name[:-5]+'</title>'+style+body)
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(root)))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    return server


def cases(root: Path,url: str,variation:int=0):
    q=shlex.quote
    long_text=' '.join(f'draft item {i:03d};' for i in range(100))
    result=[]
    def browser(name,page,goal,check,script='normal'):
        test=harness.Test(name,[goal],harness.CHROME.format(urls=url+'/'+page),check,turn_s=45)
        test.data['script']=script
        result.append(test)
        return test
    def active():return next((x for x in dom_state()['tabs'] if x.get('visible')=='visible'),{})
    def profile_check(test):
        data=active().get('state') or {}
        return (data.get('selected')=='j.reed' and data.get('choices')==['j.reed'],json.dumps(data))
    test=browser('partial-identity','profiles.html','Switch to the green Jordan Reed profile, j.reed at sample dot invalid.',profile_check,'partial')
    test.data['partial']='Switch to Jordan Reed, jordan.reed at sample dot invalid.'
    for cancelled in (False,True):
        name='stop-without-transcript' if cancelled else 'interrupt-long-typing'
        folder=root/name;folder.mkdir()
        (folder/'draft.txt').write_bytes(b'Original note.\n')
        final='Only keep this final sentence.'
        oracle=FileOracle(folder,{'draft.txt':b'Original note.\n' if cancelled else final})
        def checked(test,oracle=oracle):
            good,detail=oracle(test)
            meta=test.data['meta']
            good &= bool(meta.get('typing_began')) and meta.get('extra_old_characters',999999)<=32
            return good,json.dumps({'files':json.loads(detail),'barge_in_s':meta.get('barge_in_s'),
                                    'extra_old_characters':meta.get('extra_old_characters')})
        t=harness.Test(name,[f'Replace all the text with exactly this literal text: {long_text} Then save the same file.'],
                       f'setsid -f kwrite {q(str(folder/"draft.txt"))} >/dev/null 2>&1; sleep 2',checked,turn_s=45)
        t.data.update(script='cancel' if cancelled else 'interrupt',correction=f'Stop typing. Replace all text with exactly "{final}" and save the same file.')
        result.append(t)
    folder=root/'covered-editor';folder.mkdir();(folder/'keep.txt').write_text('Keep this original.\n')
    def closed(test):
        windows=harness.windows()
        editors=[w['caption'] for w in windows if w['id'] not in harness.KEEP and ('kate' in w['cls'].lower() or 'kwrite' in w['cls'].lower())]
        files=[w for w in windows if w['id'] not in harness.KEEP and 'dolphin' in w['cls'].lower()]
        unchanged=(folder/'keep.txt').read_bytes()==b'Keep this original.\n'
        return not editors and bool(files) and unchanged,json.dumps({'editor_windows':editors,'file_windows':len(files),'original_unchanged':unchanged})
    result.append(harness.Test('covered-editor',['Close the text editor, which is behind the file manager. Keep the file manager open.'],
       f'setsid -f kwrite {q(str(folder/"keep.txt"))} >/dev/null 2>&1; sleep 2; setsid -f dolphin --new-window {q(str(folder))} >/dev/null 2>&1; sleep 2',closed,turn_s=45))
    def unavailable(test):
        said=' '.join(test.data['said']).lower()
        page=active()
        ok=page.get('path')=='/collections.html' and 'coral' in said and any(x in said for x in ("not","can't","cannot","which","where","couldn't","available"))
        return ok,json.dumps({'page':page,'said':said})
    browser('truthful-failure','collections.html','Open the Coral collection.',unavailable)
    level=25 if variation%2==0 else 37
    def brightness(test):
        state=active();return state.get('value')==str(level),json.dumps(state)
    browser('exact-slider','brightness.html',f'Set the reading brightness to exactly {level}%.',brightness)
    def address(test):
        state=active();expected='main-' if variation%2==0 else 'docs/'
        return state.get('value')==expected and (state.get('state') or {}).get('selected')=='Studio',json.dumps(state)
    prefix='main and the character dash' if variation%2==0 else 'docs and a forward slash character'
    browser('spoken-symbols','address.html',f'In the Address field, type {prefix}, then choose Studio from its suggestions.',address)
    def service(test):
        said=' '.join(test.data['said']);meta=test.data['meta']
        return meta.get('refusals')==1 and '402' in said,json.dumps({'refusals':meta.get('refusals'),'said':said})
    test=browser('permanent-service-refusal','collections.html','Click Birds once. If the model service refuses the request, stop and tell me the error without retrying.',service)
    test.data['refuse']=True
    return result


def speaker(agent,words,meta,test):
    def press():
        now=time.monotonic();agent.brain.end_turn();return now,time.monotonic()-now
    meta['episode_started']=time.monotonic()
    press()
    if test.data.get('script')=='partial':
        agent.brain.hear(test.data['partial'],True)
        time.sleep(2.2)
        agent.brain.hear(words,True)
        time.sleep(.25)
        agent.brain.hear(words,False)
    else:
        agent.brain.hear(words,False)
    script=test.data.get('script','normal')
    if script not in ('interrupt','cancel'):
        meta['final_at']=time.monotonic()
        return meta['final_at']
    deadline=time.monotonic()+20
    while time.monotonic()<deadline and not any(r['op']=='type' and r['characters']>=32 for r in meta['hid']):
        time.sleep(.01)
    meta['typing_began']=any(r['op']=='type' and r['characters']>=32 for r in meta['hid'])
    pressed,barge=press()
    meta['interrupt_pressed_at']=pressed
    meta['barge_in_s']=round(barge,4)
    meta['extra_old_characters']=sum(r['characters'] for r in meta['hid'] if r['op']=='type' and r['t']>pressed)
    if script=='cancel':
        if hasattr(agent.brain,'cancel_turn'):agent.brain.cancel_turn()
    else:
        agent.brain.hear(test.data['correction'],False)
    meta['final_at']=time.monotonic()
    return meta['final_at']


def main():
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(SystemExit(1)))
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--variants',nargs='+',choices=['candidate','optimized'],default=['candidate','optimized'])
    parser.add_argument('--tests',nargs='+',type=int,default=list(range(1,9)))
    parser.add_argument('--repeats',type=int,default=1)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    harness.CLEAN_FILES='true'
    harness.KEEP.update(w['id'] for w in harness.windows());harness.kwin(harness.minimize_js(True))
    root=Path.home()/'Desktop'/f'exospace-stress-{time.time_ns()}';root.mkdir()
    server=site(root/'guide');target=LaptopTarget();rows=[]
    try:
        for repeat in range(args.repeats):
            variants={}
            for variant in args.variants:
                folder=root/f'round-{repeat}'/variant;folder.mkdir(parents=True)
                variants[variant]=cases(folder,f'http://127.0.0.1:{server.server_port}',repeat)
            for index in args.tests:
                for variant in args.variants if (index+repeat)%2 else args.variants[::-1]:
                    test=variants[variant][index-1];meta={'hid':[],'refusals':0};test.data['meta']=meta
                    agents=[]
                    settings=dict(test.data)  # harness resets per-run output data before constructing the agent.
                    def factory(target,*other):
                        test.data.update(settings)
                        wrapped=RecordedTarget(target,meta)
                        if variant=='optimized':
                            from bench.arch.optimized_queue import create
                            agent=create(wrapped,*other)
                        else:
                            from bench.arch.queue_agent import QueueAgent
                            from bench.arch.reliable_queue import configure
                            agent=QueueAgent(wrapped,*other);configure(agent)
                        agents.append(agent)
                        if test.data.get('refuse'):
                            def refuse(request):
                                meta['refusals']+=1
                                return httpx.Response(402,text='This request requires more credits or fewer max_tokens (fixture).')
                            client=agent.executor.pointer.client
                            client.client.close();client.client=httpx.Client(transport=httpx.MockTransport(refuse))
                        return agent
                    outside=output_files()
                    row=harness.run(test,index,repeat,*harness.MODELS['deepseek'],target,variant,'qwen','screen',args.out,
                                    agent_factory=factory,speaker=lambda agent,words:speaker(agent,words,meta,test))
                    outside_after=output_files()
                    row['unexpected_output_files']=sorted(p for p in outside.keys()|outside_after.keys() if outside.get(p)!=outside_after.get(p))
                    row['passed'] &= row['finished'] and not row['unexpected_output_files']
                    row['episode_wall_s']=round(time.monotonic()-meta['episode_started'],3)
                    row['barge_in_s']=meta.get('barge_in_s');row['extra_old_characters']=meta.get('extra_old_characters')
                    row['after_interrupt_s'] = round(meta['barge_in_s']+sum(t['after_speech_s'] for t in row['turns']),3) if 'barge_in_s' in meta else None
                    row['hid_packets']=len([x for x in meta['hid'] if x['op']!='release_all'])
                    row['api_refusals']=meta.get('refusals')
                    if hasattr(agents[-1],'control'):
                        states=agents[-1].control.requests
                        row['terminal_status']=states[max(states)]['status']
                    else:
                        said=' '.join(row['said'])
                        row['terminal_status']='blocked' if "going in circles" in said else 'verified' if row['verified'] else 'uncertain' if row['uncertain'] else 'unfinished'
                    (args.out/f'{variant}-{index}-{repeat}'/'dispatch.json').write_text(json.dumps(meta,indent=1))
                    rows.append(row);(args.out/'report.json').write_text(json.dumps(rows,indent=1))
                    print(json.dumps({k:row[k] for k in ('model','test','passed','detail','turns','barge_in_s','extra_old_characters','api_refusals')}),flush=True)
    finally:
        harness.reset();harness.kwin(harness.minimize_js(False));target.close();server.shutdown()
        shutil.copytree(root,args.out/'files');shutil.rmtree(root)
    result={}
    for variant in args.variants:
        mine=[r for r in rows if r['model']==variant]
        result[variant]={'runs':len(mine),'passed':sum(r['passed'] for r in mine),
                         'after_final_speech_p50_s':statistics.median(sum(t['after_speech_s'] for t in r['turns']) for r in mine),
                         'after_final_speech_total_s':sum(t['after_speech_s'] for r in mine for t in r['turns']),
                         'planner_records':sum(r['calls'] for r in mine),'cancelled_records':sum(r['cancelled'] for r in mine),
                         'hid_packets':sum(r['hid_packets'] for r in mine),
                         'terminal_statuses':{s:sum(r['terminal_status']==s for r in mine) for s in sorted({r['terminal_status'] for r in mine})}}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
