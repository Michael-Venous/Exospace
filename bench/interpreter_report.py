"""Aggregate preserved interpreter studies; no model calls."""
import csv
import json
import math
from pathlib import Path
import statistics
from bench.interpreter_study import summarize

STUDIES=['interpreter-smoke-v1','interpreter-smoke-live-v2','interpreter-main-v1',
    'interpreter-optimized-v1','interpreter-screen-v1','interpreter-screen-v2','interpreter-screen-span-v3','interpreter-screen-v4',
    'interpreter-freeform-v1','interpreter-vision-v1','interpreter-vision-v2','interpreter-heldout-v1','interpreter-screen-heldout-v1']
OUT=Path('/home/mike/Documents/Codex/2026-10-17/con/outputs')

def main():
    OUT.mkdir(parents=True,exist_ok=True);studies={};allrows=[]
    for study in STUDIES:
        path=Path('work')/study
        if not (path/'rows.jsonl').exists():continue
        rows=[json.loads(s) for s in (path/'rows.jsonl').read_text().splitlines()]
        stats=summarize(rows)
        for s in stats:
            rs=[r for r in rows if r['mode']==s['mode']]
            for key in ['interpretation_s','first_ready_s','grounding_s']:
                values=[r[key] for r in rs if isinstance(r.get(key),(int,float))]
                if values:s[key+'_p50']=statistics.median(values)
            if s['mode']=='jev_raw_screen':
                s['semantic_correct']=sum(r.get('semantic_correct',False) for r in rs)
                s['emitted']=sum(r.get('proposed_action',False) for r in rs)
                s['wrong_emitted']=sum(r.get('proposed_action',False) and not r.get('semantic_correct',False) for r in rs)
                s['wrong_action_proposals']=s['wrong_emitted']
            elif any(k in s['mode'] for k in ['_ground','_freeform','_vision']):
                s['wrong_action_proposals']='not scored; failures include safe grounding declines'
                s['first_correct']=sum(r.get('first_correct',False) for r in rs)
                good=[r['first_ready_s'] for r in rs if r.get('first_correct') and r.get('first_ready_s') is not None]
                if good:s['correct_first_decision_s_p50']=statistics.median(good)
        manifest=json.loads((path/'manifest.json').read_text()) if (path/'manifest.json').exists() else {}
        studies[study]={'summary':stats,'manifest':manifest,'rows':rows}
        allrows += [dict(r,study=study) for r in rows]
    notes=[
        'No new runtime defaults, HID execution, physical device, microphone or B200 deployment. These are interpretation/grounding microbenchmarks.',
        'Main: 24 authored requests repeated twice, five modes. All receive the same oracle element inventory; perception cost excluded. Element IDs shuffled per case. Expected answers never enter model prompts.',
        'Live main uses a warm persistent session with previous requests in history; setup 0.370 s excluded from per-request latency. All measured main calls serial in randomized order; no automatic REST retries.',
        'Optimized: development ablation on the SAME 24 requests, one repetition. Compact JSON reduces output size; Jev gates irrelevant fields by status. Not held-out validation.',
        'Held-out: 12 new requests and new labels frozen before optimized outcomes, two repeats. No prompt tuning on held-out results. Interpretation-only, oracle inventory still supplied.',
        'Screen v1 is diagnostic: operation definitions allowed Jev to substitute a preparatory click for fill. Screen v2 adds explicit semantic definitions; no thresholds lowered. Each tests 24 requests twice over three synthetic screen layouts.',
        'The early synthetic fixture drew links/folder/drop-region as ordinary buttons. V3/v4 render links, a folder icon and a dashed drop region appropriately. V3 uses whole contiguous-span choices; V4 uses independent start/end token choices on the same improved fixtures. Compare V3 to V4 to isolate span representation.',
        'Raw-screen Jev receives screenshot and unedited request, no element inventory, DOM, expected targets or LLM preprocessing. Generic action choices and token-index candidates bind literal text. It proposes only the FIRST semantic action, not full multi-step execution.',
        'Screen-v1 *_ground rows combine already-measured main interpretation time with subsequently measured Jev first-target grounding; synthesized sequential totals, not a single end-to-end run. They still exclude oracle inventory perception.',
        'Freeform: raw request only -> natural target descriptions -> Jev screenshots. No inventory or DOM reaches either model. One repeat per request/model. All target proposals grounded on a frozen frame, no actions executed; this does not establish later-screen sequence success.',
        'Vision studies add the screenshot to the interpreter. V2 uses the improved-role fixture. First_ready_s is time to a first grounding DECISION, which can be a rejection; correct_first_decision_s_p50 filters to correct decisions, including correct non-action decisions. Not physical input latency.',
        'Strict scoring distinguishes clarify and planner. Clarifying an underspecified preference is a safe deferral even when planner was the authored label. Report wrong action proposals separately.',
        'Jev token-span binding failures are failures of the tested factorized design, not proof that every possible Jev architecture is incapable. Quotes/empty text require better span representation. Confidence values are not calibrated task-success probabilities.',
        '95 existing unit tests passed; benchmark code compiled; manual parser checks covered partial JSON, quoted braces, and unknown targets. No outside-Gemini generative API credentials were configured.',
    ]
    findings=[
        'Compact interpretation is much cheaper than full planning: 3.5 Flash-Lite minimal took 0.807 s median (44/48 strict); 3.8 Flash low 1.353 s (48/48); 3.6 Flash minimal 1.044 s (46/48); warm Live 0.728 s (42/48). These numbers use an oracle element inventory and exclude perception.',
        'Short-array output brought Lite to 0.751 s on the development sample. On 12 new requests/labels repeated twice: Lite 22/24 at 0.783 s, Flash 24/24 at 1.577 s, Flash 3.6 22/24 at 1.013 s, status-gated Jev 18/24 at 0.204 s. Lite and 3.6 both mishandled translation. Keep parser tasks separate from generative planning.',
        'Screenshot-aware interpreter -> Jev grounding, no oracle inventory: Lite 21/24 at 1.275 s median total (1.059 s interpretation); Flash 23/24 at 2.272 s (2.024 s interpretation). Both declined one hover through the grounder; Lite also wrongly attempted generation/reasoning. Frozen screenshot proposal validation, not task/HID completion.',
        'Direct raw-screen Jev with dynamically enumerated WHOLE literal spans: 0.206 s median, 31/48 strict, 24/36 action requests accepted correctly and zero wrong accepted actions on the development sample. Independent start/end indices on the same fixture: 22/48, seven wrong accepted actions. Whole-span selection removes a major binding failure without phrase/site rules.',
        'Fresh direct-Jev requests: only 11/24 strict at 0.211 s; nine actions accepted, seven correct and two clicks substituted for clearing a field. Quoted text was declined. The zero-error development result did NOT generalize; do not promote direct Jev as a universal interpreter.',
        'Raw request-only interpretation was fast but sometimes invented navigation into a Notes application. Current screen context matters. Element inventory is a viable interface only if its actual extraction latency and accuracy are counted.',
        'Latency is largely waiting for the first output: main Lite first-token median 0.741 s, median time after first token 0.087 s. Shorter JSON alone will not remove most latency. Compare providers on complete valid actions and cold/warm p95, not advertised tokens/second.',
        'No hardcoded task parser was added. Generic action vocabulary, JSON validation, token/span enumeration, and test-only expected answers are deterministic. All language interpretation used models. Existing runtime/default configuration unchanged.',
    ]
    outside={
        'Cerebras GPT-OSS-120B / Qwen 3.8 27B':'https://inference-docs.cerebras.ai/models/overview',
        'Groq GPT-OSS-20B':'https://console.groq.com/docs/model/openai/gpt-oss-20b',
        'OpenAI GPT-6 Luna, none/low reasoning':'https://developers.openai.com/api/docs/models/gpt-6-luna',
    }
    report={'findings':findings,'notes':notes,'outside_gemini_candidates_not_benchmarked':outside,'studies':studies}
    (OUT/'exospace-interpreter-results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    fields=['study','mode','case','repeat','correct','total_s','interpretation_s','first_ready_s','error']
    with (OUT/'exospace-interpreter-results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(allrows)
    lines=['EXOSPACE INTERPRETER BENCHMARK — 2026-10-17','','FINDINGS',*findings,'',
        'Outside Gemini: prioritize a Cerebras/Groq structured-output comparison and GPT-6 Luna with reasoning disabled. Only Gemini/Jev keys are configured; no latency/accuracy claim for these alternatives. No B200 needed for hosted comparisons.',
        *[f'{k}: {v}' for k,v in outside.items()],'','METHOD AND LIMITS',*notes,'']
    for study,report in studies.items():
        lines+=['\n'+study]
        for s in report['summary']:
            lines.append(f"{s['mode']}: {s['correct']}/{s['n']} strict; median {s['p50_s']:.3f}s, p95 {s['p95_s']:.3f}s; action {s['action_correct']}/{s['action_n']}, nonaction {s['nonaction_correct']}/{s['nonaction_n']}; wrong action proposals {s['wrong_action_proposals']}; errors {s['errors']}")
            for k in ['interpretation_s_p50','first_ready_s_p50','grounding_s_p50','semantic_correct','emitted','wrong_emitted','first_correct','correct_first_decision_s_p50']:
                if k in s:lines.append(f'  {k}: {s[k]}')
        lines+=['Failures:']
        for r in report['rows']:
            if not r['correct']:
                lines.append(json.dumps({k:r[k] for k in ['mode','case','repeat','error','output','raw_output','status','op','text','gate','semantic_correct'] if k in r},ensure_ascii=False))
    (OUT/'exospace-interpreter-summary.txt').write_text('\n'.join(lines)+'\n')
    # Shared-schema rows are explicitly microbenchmark records, not task execution.
    target=Path('bench/results.csv');existing=set()
    if target.exists():
        with target.open() as f:existing={r['run_id'] for r in csv.DictReader(f)}
    header=['run_id','config','device','task_id','step','t_capture','t_upload','t_model','t_execute','t_settle','success','total_s']
    with target.open('a') as f:
        w=csv.DictWriter(f,fieldnames=header,lineterminator='\n')
        if not target.stat().st_size:w.writeheader()
        for i,r in enumerate(allrows):
            run=f"interp-{r['study']}-{r['mode']}-{r['case']}-{r.get('repeat',0)}"
            if run in existing:continue
            w.writerow(dict(run_id=run,config=r['mode'],device='laptop-microbenchmark',task_id=r['case'],step=0,
                t_capture=0,t_upload=0,t_model=round(r['total_s'],6),t_execute=0,t_settle=0,success=int(r['correct']),total_s=round(r['total_s'],6)))
            existing.add(run)
    print(json.dumps({k:v['summary'] for k,v in studies.items()},indent=2))

if __name__=='__main__':main()
