"""Separate development ablation: shorter output and status-gated Jev decoding."""
import argparse
import json
from pathlib import Path
import random
import time
from google import genai
from google.genai import types
import httpx
from bench.interpreter_study import CASES, ELEMENTS, SYSTEM, STATUS, context, prompt, normalize, tokens, jev_questions, action, summarize
from brain.config import Config

SCHEMA={'type':'object','properties':{'s':{'type':'string','enum':STATUS},
    'a':{'type':'array','maxItems':3,'items':{'type':'array','minItems':3,'maxItems':3,'items':{'type':'string'}}}},'required':['s','a']}
RULES=SYSTEM+'\nCompact wire format: {"s": status, "a": [[operation, target_id, value], ...]}. Use an empty a for non-act. An empty value is the JSON empty string, not quote characters.'

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('work/interpreter-optimized-v1'));p.add_argument('--repeats',type=int,default=2);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'rows.jsonl').exists():raise RuntimeError('Fresh output required')
    (args.out/'source.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'manifest.json').write_text(json.dumps({'cases':CASES,'elements':ELEMENTS,'system':RULES,'schema':SCHEMA,'scope':'serial randomized; held-out designation supplied by study runner'},indent=2))
    cfg=Config.load();client=genai.Client(api_key=cfg.gemini_api_key,http_options=types.HttpOptions(timeout=30000,retry_options=types.HttpRetryOptions(attempts=1)))
    jc=httpx.Client(timeout=30,headers={'Authorization':f'Bearer {cfg.typesafe_api_key}'})
    modes={'lite_compact':('gemini-3.5-flash-lite','minimal'),'flash36_compact':('gemini-3.6-flash','minimal'),'flash_compact':('gemini-3.8-flash','low')}
    jobs=[(c,rep,m) for c in CASES for rep in range(args.repeats) for m in [*modes,'jev_gated']];random.Random(919).shuffle(jobs);rows=[]
    for i,(c,rep,mode) in enumerate(jobs):
        ids,inventory=context(c);row={'case':c['id'],'request':c['request'],'repeat':rep,'mode':mode,'expected':c['expected']};t=time.perf_counter()
        try:
            if mode=='jev_gated':
                response=jc.post('https://api.typesafe.ai/v1/systemone',json={'model':cfg.jev_model,'state':{'request':c['request'],'elements':inventory},'questions':jev_questions(c,inventory)})
                if response.status_code!=200:raise RuntimeError('HTTP '+str(response.status_code))
                raw=response.json()['answers'];row['answers']=raw
                status=raw['status']['choice'];acts=[];seen_none=False;ms=tokens(c['request'])
                if status=='act':
                    for j in range(3):
                        op=raw[f'op{j}']['choice']
                        if op=='none':seen_none=True;continue
                        if seen_none:raise ValueError('Action slot gap')
                        value=''
                        if op=='fill':
                            a=int(raw[f'start{j}']['choice']);b=int(raw[f'end{j}']['choice'])
                            if not 0<=a<=b<len(ms):raise ValueError('Invalid span')
                            value=c['request'][ms[a].start():ms[b].end()]
                        if op=='drag':value=raw[f'dest{j}']['choice']
                        acts.append(action(op,raw[f'target{j}']['choice'],value))
                data={'status':status,'actions':acts}
            else:
                model,thinking=modes[mode];buf='';first=None;usage=None
                for chunk in client.models.generate_content_stream(model=model,contents=prompt(c,inventory),config=types.GenerateContentConfig(system_instruction=RULES,response_mime_type='application/json',response_json_schema=SCHEMA,thinking_config=types.ThinkingConfig(thinking_level=thinking),temperature=0)):
                    piece=chunk.text or '';buf+=piece
                    if piece and first is None:first=time.perf_counter()-t
                    if chunk.usage_metadata:usage=chunk.usage_metadata.model_dump(exclude_none=True)
                wire=json.loads(buf);row.update(wire=wire,first_token_s=first,usage=usage)
                data={'status':wire['s'],'actions':[action(*a) for a in wire['a']]}
            row['raw_output']=data;row['output']=normalize(data,ids);row['correct']=row['output']==c['expected']
        except Exception as exc:row.update(correct=False,error=type(exc).__name__)
        row['total_s']=time.perf_counter()-t;rows.append(row)
        with (args.out/'rows.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(i+1,len(jobs),mode,c['id'],row['correct'],round(row['total_s'],3),flush=True)
    (args.out/'summary.json').write_text(json.dumps(summarize(rows),indent=2));client.close();jc.close()

if __name__=='__main__':main()
