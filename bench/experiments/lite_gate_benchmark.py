"""Challenge a conservative, experimental Lite gate; never enables routing.

False admission of any nonliteral/ambiguous task rejects this candidate gate.
Thresholds and cases are saved before requests, and are not fitted to outcomes.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
import time

from brain.agent.jev import JevVision, probability
from brain.browser import BrowserTarget
from brain.cancel import CancelToken
from brain.config import Config
from bench.experiments.routing_benchmark import LEVELS, questions
from bench.experiments.routing_cases import screen

GATE={'max_complexity_exclusive':1.5,'min_complexity_confidence':.95,
      'min_basic_probability':.98,'min_basic_confidence':.95}
GOAL='Complete the form using the request shown on screen, then save it.'
REQUESTS=[
    ('time','Start at 11:40 PM; duration 35 minutes. Fill End time.','End time',False),
    ('tax','Price 79.95; sales tax 7.5%. Fill Total including tax, rounded to cents.','Total',False),
    ('quantity','Order 7 packs at 18.60 each. Fill Amount due.','Amount due',False),
    ('unit','Package weighs 3.2 kilograms. Fill Weight in grams.','Weight in grams',False),
    ('translation','The message is: Please arrive before noon. Fill the Spanish translation.','Spanish message',False),
    ('summary','Meeting notes: Prototype passed input tests. Audio is missing. Adapter arrives tomorrow. Write a one-sentence summary.','Summary',False),
    ('draft','Write a polite request for an extension explaining that hardware has not arrived.','Message',False),
    ('date','The appointment is the second Tuesday of November 2026. Fill date as YYYY-MM-DD.','Date',False),
    ('condition','Stock is 4. Target stock is 12. Order only the shortage. Fill Order quantity.','Order quantity',False),
    ('compare','Option A costs 8 per unit plus 14 shipping. Option B costs 10 per unit with free shipping. For 9 units enter the cheaper option.','Selected option',False),
    ('sort','Sizes: Cedar 29 KB, Birch 17 KB, Maple 31 KB. Enter names sorted from smallest to largest.','Ordered names',False),
    ('constraint','Choose a room with at least 6 seats and a projector. A has 8 seats, no projector. B has 5 and projector. C has 7 and projector.','Room',False),
    ('missing','Enter the destination I prefer. Choices: Lab or Office. No preference has been provided.','Destination',False),
    ('conflict','Set the title to Budget. Also, set the title to Schedule. There is only one title field and no priority rule.','Title',False),
    ('derived_literal','Copy Name Morgan and Project Exospace. For Score, use twice the displayed base score of 17.','Name; Project; Score',False),
    ('conditional_text','Copy the code as-is if enabled; otherwise reverse it. Code: EXO42. Enabled: no.','Code',False),
    ('literal_url','Website address: youtube.com. Copy it exactly.','Website address',True),
    ('literal_fields','Name: Morgan. Project: Exospace. Note: Test the prototype. Copy every value exactly.','Name; Project; Note',True),
    ('literal_number','Reference: 035179. Copy exactly, retaining the leading zero.','Reference',True),
    ('literal_long','Name: Riley. Project: Atlas. Team: Hardware. Room: 204. Note: Review the adapter. Copy all five fields exactly as provided.','Name; Project; Team; Room; Note',True),
    ('literal_math_words','Title: Calculating tax with seven constraints. Copy this exact title; do not calculate anything.','Title',True),
    ('literal_quoted','Note: If stock is below five, reorder. Copy this exact note; do not evaluate the condition.','Note',True),
]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='work/lite-gate-v1');args=p.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists(): raise ValueError('Choose a fresh output directory')
    cases=[{'id':id_,'goal':GOAL,'html':screen('Local form request',['Request','Destination fields'],[[request,fields]]),
            'eligible':eligible} for id_,request,fields,eligible in REQUESTS]
    # Use wide wrapped cells so even the long request fits without clipping.
    for c in cases: c['html']+='<style>table{table-layout:fixed;width:1100px}td{white-space:normal;line-height:1.5}</style>'
    manifest={'cases':cases,'gate':GATE,'repeats':2,'reject_rule':'Any false admission rejects automatic Lite routing.',
              'created_unix':time.time(),'cases_sha256':sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    target=BrowserTarget('about:blank')
    try:
        for c in cases:
            target.page.set_content(c['html'])
            if target.page.evaluate('document.documentElement.scrollHeight > innerHeight'): raise ValueError('Clipped fixture')
            target.page.screenshot(path=str(out/f'{c["id"]}.jpg'),type='jpeg',quality=90)
    finally: target.close()
    cfg=Config.load();jev=JevVision(cfg.typesafe_api_key,CancelToken(),out/'api_calls.jsonl',model=cfg.jev_model)
    rows=[]
    try:
        for repeat in range(2):
            order=cases.copy();random.Random(712+repeat).shuffle(order)
            for case in order:
                qs=questions(case['goal'])
                qs['lite_eligible']={'type':'choice','instructions':{
                    'request':case['goal'],
                    'question':'Use the actual content of screen, including the request shown there. Is this task restricted entirely to explicit literal steps? '
                    'Evaluate the whole task, not just its first click. A single derived field disqualifies the entire task. '
                    'Copying a quoted sentence containing math words is still literal if no evaluation is requested.'},
                    'criteria':{
                        'basic_literal':'All steps and values are explicit: copy exact supplied text, follow a specified sequence, or choose a uniquely identified visible item. '
                                        'No calculation, date conversion, comparison, sorting, translation, summarization, original writing, inference, or missing preference.',
                        'reasoning':'Any part needs deriving a value, evaluating a condition, comparing options, calculating, sorting, translating, summarizing, or composing new text.',
                        'unclear':'Essential user intent, values, preference, or disambiguation is missing or contradictory.'}}
                start=time.monotonic()
                row={'id':case['id'],'repeat':repeat,'eligible':case['eligible']}
                try:
                    answers,_,_=jev.evaluate((out/f'{case["id"]}.jpg').read_bytes(),qs,run_id=case['id'],step=repeat)
                    complexity=answers['complexity'];basic=answers['lite_eligible']
                    score=sum(LEVELS[k]*probability(complexity['probabilities'][k]) for k in LEVELS)
                    admitted=(score<GATE['max_complexity_exclusive'] and
                              probability(complexity['confidence'])>=GATE['min_complexity_confidence'] and
                              answers['clarity']['choice']=='clear' and basic['choice']=='basic_literal' and
                              probability(basic['probabilities']['basic_literal'])>=GATE['min_basic_probability'] and
                              probability(basic['confidence'])>=GATE['min_basic_confidence'])
                    row.update(answers=answers,score=score,admitted=admitted,error='')
                except Exception as exc: row.update(admitted=False,error=type(exc).__name__)
                row['elapsed_s']=time.monotonic()-start;rows.append(row)
                with (out/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(f'GATE {repeat+1} {case["id"]} eligible={case["eligible"]} admitted={row["admitted"]}',flush=True)
    finally: jev.close()
    false=[r for r in rows if r['admitted'] and not r['eligible']]
    report={'gate':GATE,'false_admissions':false,'nonliteral_or_ambiguous_trials':sum(not r['eligible'] for r in rows),
            'literal_admitted':sum(r['eligible'] and r['admitted'] for r in rows),'literal_trials':sum(r['eligible'] for r in rows),
            'distinct_tasks':len(cases),'repeats':2,'candidate_rejected':bool(false),'production_enabled':False,
            'limitations':'Synthetic challenge set. Zero observed false admissions would not guarantee safe routing on arbitrary real applications.',
            'rows':rows}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in {'rows','false_admissions'}}),flush=True)


if __name__=='__main__': main()
