"""Compare original sequential launch checks with combined checks on controlled screens.

No HID or real apps are driven. Uses the configured Jev API and logs timings to bench/results.csv.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from brain.agent.jev import COMPLETION_QUESTION, JevVision, probability
from brain.agent.race import DIRECT_GATE, LAUNCHED
from brain.agent.triage import GATE
from brain.cancel import CancelToken
from brain.config import Config
from brain.telemetry import Telemetry

OUT = Path('work/launch-check-study')


def window(title: str, content: str) -> str:
    return ('<body style="margin:0;background:#233747;font:20px sans-serif">'
            '<main style="margin:35px;background:white;height:650px;border:2px solid #aaa">'
            f'<header style="background:#ddd;padding:15px">{title}</header>'
            f'<section style="padding:35px">{content}</section></main></body>')


CASES = [
    ('firefox_open', 'open Firefox', 'Firefox', True,
     window('Mozilla Firefox — New Tab', '<div>← → ⟳ &nbsp; Search or enter address</div><h1>Firefox</h1><input placeholder="Search the web">')),
    ('calculator_open', 'open Calculator', 'Calculator', True,
     window('Calculator', '<h1>0</h1><p>7 &nbsp; 8 &nbsp; 9 &nbsp; ÷</p><p>4 &nbsp; 5 &nbsp; 6 &nbsp; ×</p><p>1 &nbsp; 2 &nbsp; 3 &nbsp; −</p><p>0 &nbsp; . &nbsp; = &nbsp; +</p>')),
    ('calculation_missing', 'open Calculator and work out 12 times 9', 'Calculator', False,
     window('Calculator', '<h1>0</h1><p>7 &nbsp; 8 &nbsp; 9 &nbsp; ÷</p><p>4 &nbsp; 5 &nbsp; 6 &nbsp; ×</p><p>1 &nbsp; 2 &nbsp; 3 &nbsp; −</p><p>0 &nbsp; . &nbsp; = &nbsp; +</p>')),
    ('web_search_trap', 'open Firefox', 'Firefox', False,
     window('Google Chrome — Firefox - Google Search', '<div>google.com/search?q=Firefox</div><h1>Firefox</h1><p>Download Mozilla Firefox for Windows</p><p>A fast, private web browser. Download now.</p>')),
    ('desktop_shortcut', 'open Firefox', 'Firefox', False,
     '<body style="background:#23517a;color:white;font:20px sans-serif"><p>🖥 Home</p><p>Firefox shortcut</p><footer style="position:fixed;bottom:0">Applications &nbsp; Files &nbsp; 08:40</footer></body>'),
    ('login_not_inbox', 'open Firefox and show my Gmail inbox', 'Firefox', False,
     window('Mozilla Firefox — Sign in - Google Accounts', '<div>accounts.google.com</div><h1>Sign in</h1><p>to continue to Gmail</p><input placeholder="Email or phone"><button>Next</button>')),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config.load()
    token = CancelToken()
    jev = JevVision(cfg.typesafe_api_key, token, OUT / 'model_calls.jsonl', model=cfg.jev_model,
                    timeout_s=cfg.jev_timeout_s)
    shots = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={'width': 1280, 'height': 800})
        for name, _, _, _, html in CASES:
            page.set_content(html)
            shots[name] = page.screenshot(type='jpeg', path=str(OUT / f'{name}.jpg'))
        browser.close()
    rows = []
    try:
        for repeat in range(2):
            for name, goal, app, expected, _ in CASES:
                for mode in (('sequential', 'combined') if repeat == 0 else ('combined', 'sequential')):
                    telemetry = Telemetry(Path('bench/results.csv'), 'launch-check-' + mode)
                    run_id = telemetry.new_run()
                    started = time.monotonic()
                    complete = 0.0
                    questions = {'launched': {'type': 'noul', 'instructions': LAUNCHED}}
                    state = {'utterance': goal}
                    if mode == 'combined':
                        questions['complete'] = {'type': 'noul', 'instructions': COMPLETION_QUESTION}
                        state.update(goal=goal, executed_steps=[f'open the app {app}'])
                    answers, _, _ = jev.evaluate(shots[name], questions, run_id=run_id, step=1, state=state)
                    launched = probability(answers['launched'].get('noul'))
                    if mode == 'combined':
                        complete = probability(answers['complete'].get('noul'))
                    elif launched >= GATE:
                        complete = jev.completion(shots[name], goal, [f'open the app {app}'], run_id=run_id, step=1)
                    elapsed = time.monotonic() - started
                    accepted = launched >= GATE and complete >= DIRECT_GATE
                    telemetry.step(run_id, 'synthetic', name, 1, t_capture=0, t_upload=0, t_model=elapsed,
                                   t_execute=0, t_settle=0)
                    telemetry.finish(run_id, 'synthetic', name, 1, accepted == expected, elapsed)
                    row = dict(repeat=repeat, case=name, mode=mode, expected=expected, accepted=accepted,
                               launched=launched, complete=complete, seconds=elapsed)
                    rows.append(row)
                    (OUT / 'results.json').write_text(json.dumps(rows, indent=2))
                    print(json.dumps(row), flush=True)
    finally:
        jev.close()
    for mode in ('sequential', 'combined'):
        selected = [r for r in rows if r['mode'] == mode]
        print(mode, 'correct', sum(r['accepted'] == r['expected'] for r in selected), '/', len(selected),
              'median_s', round(statistics.median(r['seconds'] for r in selected), 3), flush=True)


if __name__ == '__main__':
    main()
