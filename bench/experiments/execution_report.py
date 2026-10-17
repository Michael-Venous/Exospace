"""Summarize architecture benchmark evidence without hiding failed runs."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import statistics


def stats(rows):
    if not rows:return {}
    first=[r['first_input_s'] for r in rows if r['first_input_s'] is not None]
    walls=[r['wall_s'] for r in rows]
    return {'passed':sum(r['success'] for r in rows),'n':len(rows),
        'wall_median_s':statistics.median(walls),'wall_min_s':min(walls),'wall_max_s':max(walls),
        'first_input_median_s':statistics.median(first) if first else None,
        'zero_llm_runs':sum(r['success'] and not any(role in r['calls'] for role in ('flash','live','fast')) for r in rows),
        'ambiguous_requests_with_input':sum(r['case']=='ambiguous' and bool(r['events']) for r in rows),
        'false_done':sum(r.get('result') is not None and r['result']['status']=='done' and not r['success'] for r in rows),
        'correct_end_state_but_not_done':sum(r['case']!='ambiguous' and r['actual']==r['expected'] and not r['success'] and (not r.get('result') or r['result']['status']!='done') for r in rows)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    reports=[];allrows=[]
    for name in args.inputs:
        folder=Path(name);manifest=json.loads((folder/'manifest.json').read_text())
        rows=json.loads((folder/'report.json').read_text())
        for row in rows:
            row['study']=folder.name
            # v1's direct interface did not expose hover. The independent end-state
            # oracle missed click-vs-hover compliance; audit the retained action trace.
            if row['case']=='hover_menu':
                trace=folder/f"{row['case']}-{row['mode']}-{row['repeat']}"/'trace.jsonl'
                steps=[json.loads(s) for s in trace.read_text().splitlines()] if trace.exists() else []
                actions=[a for s in steps for a in s['actions'] if a['type'] not in {'ask_user','done','fail','remember'}]
                if actions and not(actions[0]['type']=='skill' and actions[0]['name']=='hover'):
                    row['raw_success']=row['success'];row['success']=False
                    row['adjudication']='First pointer action clicked rather than hovered; instruction compliance failure despite possible correct export end state.'
        report={'study':folder.name,'manifest':manifest,'rows':rows,
            'summary':{mode:stats([r for r in rows if r['mode']==mode]) for mode in dict.fromkeys(r['mode'] for r in rows)},
            'by_task':{case:{mode:stats([r for r in rows if r['case']==case and r['mode']==mode]) for mode in dict.fromkeys(r['mode'] for r in rows if r['case']==case)} for case in dict.fromkeys(r['case'] for r in rows)}}
        reports.append(report);allrows.extend(rows)
    payload={'scope':'Synthetic isolated screenshot-only browser tests; no target DOM given to models. Success requires exact independent final state plus done, or clarification with no input for ambiguous requests. Hover compliance is additionally audited. No physical devices, microphone/audio, native browser chrome, reconnect endurance, or production reliability claims.',
        'interpretation':'Medians include failures. First input is not necessarily useful or correct. Warm Live setup is excluded from wall and separately recorded; cold studies include connection. Two repeats per task are exploratory, not statistical reliability validation. Development and held-out studies are labeled separately.',
        'studies':reports}
    (out/'exospace-execution-results.json').write_text(json.dumps(payload,indent=2))
    fields=['study','case','mode','repeat','success','wall_s','first_input_s','warm_setup_s','recipe_admitted','overlap_used']
    with (out/'exospace-execution-results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(allrows)
    lines=['Exospace execution architecture benchmark — 2026-10-17','',payload['scope'],'',payload['interpretation'],'']
    for report in reports:
        lines.extend([report['study'], 'mode | passed | median total s | median first input s | successful zero LLM runs | false done'])
        for mode,s in report['summary'].items():
            first=f"{s['first_input_median_s']:.2f}" if s['first_input_median_s'] is not None else '—'
            lines.append(f"{mode} | {s['passed']}/{s['n']} | {s['wall_median_s']:.2f} | {first} | {s['zero_llm_runs']} | {s['false_done']}")
        lines.append('')
        for case,modes in report['by_task'].items():
            lines.append(case+': '+ '; '.join(f"{mode} {s['passed']}/{s['n']}, {s['wall_median_s']:.2f}s" for mode,s in modes.items()))
        lines.append('')
    (out/'exospace-execution-summary.txt').write_text('\n'.join(lines)+'\n')
    # Preserve the shared CSV schema. Append only previously unseen run/step pairs.
    destination=Path('bench/results.csv');destination.parent.mkdir(exist_ok=True)
    existing=list(csv.DictReader(destination.open())) if destination.exists() else []
    keys={(r['run_id'],r['step']) for r in existing};new=[];fieldnames=None
    for name in args.inputs:
        for source in Path(name).glob('*/results.csv'):
            with source.open() as f:
                reader=csv.DictReader(f);fieldnames=reader.fieldnames
                for row in reader:
                    key=(row['run_id'],row['step'])
                    if key not in keys:new.append(row);keys.add(key)
    if new:
        with destination.open('a') as f:
            writer=csv.DictWriter(f,fieldnames=fieldnames)
            if not existing:writer.writeheader()
            writer.writerows(new)
    print(json.dumps({r['study']:r['summary'] for r in reports},indent=2))


if __name__=='__main__':main()
