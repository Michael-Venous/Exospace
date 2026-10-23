"""Independent outcome checks on real laptop apps, with both kinds of spoken instruction.

Run ON the laptop with the same key wrapper as laptop_tests. Transcripts arrive word by word;
this measures the control architecture, not microphone/STT latency. Fixture paths are unique
per invocation, and only those paths are removed. No oracle information reaches the agent.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import signal
import sys
import shlex
import statistics
import threading
import time
from pathlib import Path

from bench.arch import laptop_tests as harness
from bench.arch.laptop import LaptopTarget
from bench.arch.reliability_oracles import FileOracle


def output_files() -> dict[str, tuple[int, int]]:
    """Metadata only: catch stray output at common save destinations outside the owned fixture tree."""
    result = {}
    for name in ('Desktop', 'Documents', 'Downloads', 'Pictures'):
        folder = Path.home()/name
        if folder.is_dir():
            for file in folder.iterdir():
                if file.is_file() and file.name != '.directory' and not file.name.endswith('.kate-swp'):
                    try:
                        stat = file.stat()
                    except (FileNotFoundError, PermissionError):
                        continue
                    result[str(file)] = (stat.st_size, stat.st_mtime_ns)
    return result


def page_state() -> dict:
    """Read-only, outside the agent: active tab URL/DOM state and all test-profile tabs."""
    from websockets.sync.client import connect
    tabs = harness.chrome_tabs()
    states = []
    for tab in tabs:
        try:
            with connect(tab['webSocketDebuggerUrl'], open_timeout=2) as ws:
                ws.send(json.dumps({'id': 1, 'method': 'Runtime.evaluate', 'params': {'expression':
                    'JSON.stringify({path:location.pathname,query:location.search,title:document.title,'
                    'visible:document.visibilityState,events:JSON.parse(sessionStorage.events||"[]")})'}}))
                result = json.loads(ws.recv(timeout=3))['result']['result']['value']
                states.append(json.loads(result))
        except Exception as exc:
            states.append({'error': type(exc).__name__})
    return {'tabs': states}


def browser_check(path: str, ordered: list[str], tabs: int = 1, exact_order: bool = False):
    def check(test):
        state = page_state()
        active = next((p for p in state['tabs'] if p.get('visible') == 'visible'), {})
        events = active.get('events', [])
        cursor = iter(events)
        order_ok = events == ordered if exact_order else all(any(e == want for e in cursor) for want in ordered)
        ok = active.get('path') == path and len(state['tabs']) == tabs and order_ok
        return ok, json.dumps(state)
    return check


def close_pdf_tabs(target: LaptopTarget) -> None:
    """Close only test-created PDF tabs, including a default browser outside the test profile."""
    for _ in range(8):
        window = next((w for w in harness.windows() if w['id'] not in harness.KEEP and
                       w['caption'].startswith('receipt-current.pdf')), None)
        if window is None:
            return
        ident = json.dumps(window['id'])
        harness.kwin(f'var ws=workspace.windowList(); for(var i=0;i<ws.length;i++) '
                     f'if(String(ws[i].internalId)=={ident}) {{ ws[i].minimized=false; workspace.activeWindow=ws[i]; }}')
        if not harness.active_caption().startswith('receipt-current.pdf'):
            raise RuntimeError('could not focus test PDF for tab cleanup')
        target.send({'op': 'key', 'keys': ['CTRL', 'W']})
        time.sleep(.4)
    raise RuntimeError('test PDF tabs remain after bounded cleanup')


STYLE = '<style>body{font:24px sans-serif;margin:60px;max-width:1000px}a,button,input{font:inherit;margin:15px;padding:12px}nav{display:flex;gap:25px}aside{background:#eef;padding:20px}</style>'
SCRIPT = '''<script>addEventListener('pageshow',()=>{let events=JSON.parse(sessionStorage.events||'[]');events.push(location.pathname);sessionStorage.events=JSON.stringify(events);});</script>'''


def site(root: Path):
    """Public GUI fixture, served locally; no browser automation in the agent's path."""
    root.mkdir()
    pages = {
        'index.html': '<h1>Field Notes</h1><nav><a href="/birds.html">Birds</a><a href="/plants.html">Plants</a></nav><p>Choose a collection.</p>',
        'birds.html': '<h1>Birds</h1><a href="/heron.html">Great blue heron</a><a href="/robin.html">American robin</a><p>Two species in this collection.</p>',
        'plants.html': '<h1>Plants</h1><a href="/cedar.html">Red cedar</a><a href="/fern.html">Royal fern</a>',
        'heron.html': '<h1>Great blue heron</h1><p>A tall wading bird.</p><a href="/birds.html">Back to birds</a>',
        'robin.html': '<h1>American robin</h1><p>A songbird with an orange breast.</p>',
        'cedar.html': '<h1>Red cedar</h1><p>An evergreen tree.</p>',
        'fern.html': '<h1>Royal fern</h1><p>A fern of wet woodlands.</p>',
        'waiting.html': '<h1>Loading collection…</h1><script>setTimeout(()=>location.href="/birds.html",1800)</script>',
    }
    for name, body in pages.items():
        (root / name).write_text('<!doctype html><meta charset="utf-8"><title>'+body.split('</h1>')[0].replace('<h1>', '')+'</title>'+STYLE+body+SCRIPT)
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def cases(root: Path, url: str, variation: int = 0) -> list[harness.Test]:
    q = shlex.quote
    def start_editor(file: Path | None = None):
        return f'setsid -f kwrite {q(str(file)) if file else ""} >/dev/null 2>&1; sleep 2'
    def start_files(folder: Path):
        return f'setsid -f dolphin --new-window {q(str(folder))} >/dev/null 2>&1; sleep 2'
    def file_case(name, words, setup, expected, suite='open', turns=None):
        folder = root / name
        folder.mkdir()
        for path, data in setup.items():
            dest = folder / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data if isinstance(data, bytes) else data.encode())
        return harness.Test(name, turns or [words.format(folder=folder)],
                            start_editor(folder / 'draft.txt') if 'draft.txt' in setup else start_editor(),
                            FileOracle(folder, expected), suite=suite, turn_s=75)
    tests = [
        file_case('save-note', 'Could you keep a note for me? It should say bring the blue folder, and save it as packing.txt in {folder}.',
                  {}, {'packing.txt': 'bring the blue folder'}),
        file_case('save-steps', 'In the text editor, make a new document, type call Sam at noon, then save it as reminder.txt in {folder}.',
                  {}, {'reminder.txt': 'call Sam at noon'}, 'steps'),
        file_case('append-followup', '', {'draft.txt': 'Monday: check the mail\n'},
                  {'draft.txt': 'Monday: check the mail\nTuesday: water the plants'}, turns=[
                      'Read the note that is open and tell me what it says.',
                      'Keep that line. Add Tuesday: water the plants on a new line, and save the same file.']),
        file_case('save-copy', 'Select all the text, replace it with revised plan, and save a copy as final.txt in {folder}. Keep draft.txt unchanged.',
                  {'draft.txt': 'original plan\n'}, {'draft.txt': b'original plan\n', 'final.txt': 'revised plan'}, 'steps'),
    ]
    # File operations: exact original bytes must survive, with every decoy left alone.
    for name, instruction, initial, expected, suite in [
        ('rename-file', 'This folder has a rough plan. Rename draft.txt to agenda.txt for me.',
         {'draft.txt': 'agenda items\n', 'draft-old.txt': 'old draft\n'}, {'agenda.txt': 'agenda items\n', 'draft-old.txt': 'old draft\n'}, 'open'),
        ('move-files', 'Make a folder called archive here. Select invoice-blue.txt and invoice-green.txt, cut them, open archive, and paste them there.',
         {'invoice-blue.txt': 'blue invoice\n', 'invoice-green.txt': 'green invoice\n', 'notes.txt': 'leave this\n'},
         {'archive/invoice-blue.txt': 'blue invoice\n', 'archive/invoice-green.txt': 'green invoice\n', 'notes.txt': 'leave this\n'}, 'steps'),
    ]:
        folder = root / name
        folder.mkdir()
        for file, content in initial.items():
            (folder / file).write_text(content)
        tests.append(harness.Test(name, [instruction], start_files(folder),
                                 FileOracle(folder, {name: content.encode() for name, content in expected.items()}),
                                 suite=suite, turn_s=90))
    tests.extend([
        harness.Test('browse-semantic', ['I would like to read about the tall wading bird. Find its page in this guide.'],
                     harness.CHROME.format(urls=url+'/index.html'), browser_check('/heron.html', ['/index.html','/birds.html','/heron.html']), turn_s=60),
        harness.Test('browse-back', ['Open Birds, open Great blue heron, go back once, and then open American robin.'],
                     harness.CHROME.format(urls=url+'/index.html'), browser_check('/robin.html', ['/index.html','/birds.html','/heron.html','/birds.html','/robin.html'], exact_order=True), suite='steps', turn_s=60),
        harness.Test('named-tab', ['Close the Birds tab and leave Plants open.'],
                     harness.CHROME.format(urls=url+'/birds.html '+url+'/plants.html'), browser_check('/plants.html', ['/plants.html']), suite='steps', turn_s=45),
        harness.Test('delayed-page', ['Once this collection is ready, show me the robin.'],
                     harness.CHROME.replace('sleep 4', 'sleep 0.2').format(urls=url+'/waiting.html'),
                     browser_check('/robin.html', ['/waiting.html','/birds.html','/robin.html']), turn_s=60),
        harness.Test('calculator-open', ['I need to know how much 73 times 28 comes to. Could you work it out in the calculator?'],
                     'true', lambda t: (bool(__import__('re').search(r'2,?044', ' '.join(t.data['said']))) and
                                       any('kcalc' in c.lower() for _, c in t.data['timeline']), json.dumps(t.data['said'])), turn_s=60),
        harness.Test('calculator-steps', ['Open the calculator, enter 915 minus 267, press equals, and tell me the answer.'],
                     'true', lambda t: ('648' in ' '.join(t.data['said']) and any('kcalc' in c.lower() for _, c in t.data['timeline']),
                                       json.dumps(t.data['said'])), suite='steps', turn_s=60),
    ])
    # Held out from the 12-case development suite: non-repeatable and disappearing effects.
    tests.append(harness.Test('one-photo-close',
        ['I need a quick photo for my records. Take just one, then close the camera app.'],
        'true', harness._camera, turn_s=90,
        cleanup="find ~/Pictures -maxdepth 1 -newer /tmp/exo_test_start -name '*.jpg' -delete"))
    closed = file_case('save-close', 'In this editor, type reserve a quiet room, save it as room.txt in {folder}, and close the editor.',
                       {}, {'room.txt': 'reserve a quiet room'}, 'steps')
    saved = closed.check
    def saved_and_closed(test):
        ok, detail = saved(test)
        left = [w['caption'] for w in harness.windows() if w['id'] not in harness.KEEP and
                ('kwrite' in w['cls'].lower() or 'kate' in w['cls'].lower())]
        return ok and not left, json.dumps({'files': json.loads(detail), 'editor_windows': left})
    closed.check = saved_and_closed
    tests.append(closed)
    pdf_folder = root/'newest-document'
    pdf_folder.mkdir()
    pdf_bytes = (b'%PDF-1.1\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>'
                 b'endobj 3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 144]>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n')
    pdfs = {'summary-old.pdf': pdf_bytes, 'receipt-current.pdf': pdf_bytes, 'manual.pdf': pdf_bytes,
            'newest-note.txt': b'This is not a PDF.\n'}
    for age, (name, content) in enumerate(pdfs.items()):
        file = pdf_folder/name
        file.write_bytes(content)
        modified = time.time() + (100 if name.endswith('.txt') else -10 if name == 'receipt-current.pdf' else -1000-age)
        os.utime(file, (modified, modified))
    def newest_pdf(test):
        ok, detail = FileOracle(pdf_folder, pdfs)(test)
        active = [w for w in harness.windows() if w['active']]
        opened = any(__import__('re').match(r'^receipt-current[.]pdf(?: [—–-] |$)', w['caption']) for w in active)
        return ok and opened, json.dumps({'files': json.loads(detail), 'active': active})
    tests.append(harness.Test('newest-pdf-heldout', ['Open the newest PDF in this folder for me.'],
                              start_files(pdf_folder), newest_pdf, turn_s=90))
    def planets(test):
        from urllib.parse import unquote
        visits = [(t, unquote(url)) for t, url in harness.firefox_visits()]
        mercury = [t for t, url in visits if url.endswith('/wiki/Mercury_(planet)')]
        venus = [t for t, url in visits if url.endswith('/wiki/Venus') and mercury and t > mercury[0]]
        return bool(venus) and harness.active_caption().startswith('Venus - Wikipedia'), json.dumps(visits)
    tests.append(harness.Test('public-web-heldout',
        ['In this Firefox window, open Wikipedia about the planet Mercury, then open the article about Venus.'],
        harness.FIREFOX, planets, turn_s=120, suite='steps',
        cleanup="pkill -f 'profile /tmp/exo-f[f]'; sleep 1; rm -rf /tmp/exo-ff"))
    if variation % 2:
        replacements = {'bring the blue folder': 'charge the spare battery', 'packing.txt': 'travel.txt',
                        'call Sam at noon': 'collect the parcel at four', 'reminder.txt': 'errands.txt',
                        'Monday: check the mail': 'Before lunch: check the mail',
                        'Tuesday: water the plants': 'After lunch: water the plants',
                        'draft.txt': 'outline.txt', 'draft-old.txt': 'outline-old.txt',
                        'agenda.txt': 'itinerary.txt', 'final.txt': 'approved.txt',
                        'original plan': 'earlier schedule', 'revised plan': 'updated schedule',
                        'invoice-blue.txt': 'receipt-oak.txt', 'invoice-green.txt': 'receipt-maple.txt',
                        'archive': 'collected'}
        def changed(text):
            for old, new in replacements.items():
                text = text.replace(old, new)
            return text
        for test in tests:
            test.turns = [changed(words) for words in test.turns]
            test.setup = changed(test.setup)
            if isinstance(test.check, FileOracle):
                oracle = test.check
                for file in list(oracle.root.rglob('*')):
                    if file.is_file():
                        content = changed(file.read_text())
                        dest = file.with_name(changed(file.name))
                        file.unlink()
                        dest.write_text(content)
                oracle.expected = {changed(name): changed(content.decode()).encode() if isinstance(content, bytes)
                                   else changed(content) for name, content in oracle.expected.items()}
    return tests


def summary(rows):
    firsts = [t['first_input_vs_speech_end_s'] for r in rows for t in r['turns'] if t['first_input_vs_speech_end_s'] is not None]
    return {'runs': len(rows), 'passed': sum(r['passed'] for r in rows),
            'first_input_p50_s': statistics.median(firsts) if firsts else None,
            'after_speech_total_s': sum(t['after_speech_s'] for r in rows for t in r['turns']),
            'outcomes_correct': sum(r.get('outcome_passed', r['passed']) for r in rows),
            'verified_runs': sum(r['verified'] for r in rows),
            'uncertain_runs': sum(r.get('uncertain', False) for r in rows),
            'false_verified': sum(r['verified'] and not r.get('outcome_passed', r['passed']) for r in rows),
            'failed_but_finished': sum(r['finished'] and not r['passed'] for r in rows),
            'errors': sum(len(r['errors']) for r in rows)}


def main(case_factory=cases, default_tests=range(1, 13)):
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tests', type=int, nargs='+',
                        help='default: 1-12; held out: 13 camera, 14 save+close, 15 newest PDF, 16 public Firefox')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--candidate', action='store_true')
    parser.add_argument('--variants', nargs='+', choices=['baseline', 'candidate'],
                        help='interleave both variants; alternate which goes first for each case')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)  # never silently mix repetitions with an earlier run
    # Replace legacy broad fixture cleanup for THIS harness only. All fixture files live below one new root.
    harness.CLEAN_FILES = 'true'
    harness.KEEP.update(w['id'] for w in harness.windows())
    harness.kwin(harness.minimize_js(True))
    root = Path.home() / 'Desktop' / f'exospace-eval-{time.time_ns()}'
    root.mkdir()
    server = site(root / 'guide')
    rows = []
    variants = args.variants or (['candidate'] if args.candidate else ['baseline'])
    target = LaptopTarget()
    try:
        for repeat in range(args.repeats):
            tests = {}
            for variant in variants:
                fixture = root / f'round-{repeat}' / variant
                fixture.mkdir(parents=True)
                tests[variant] = case_factory(fixture, f'http://127.0.0.1:{server.server_port}', repeat)
            for index in args.tests or default_tests:
                for variant in variants if (index+repeat) % 2 else variants[::-1]:
                    test = tests[variant][index-1]
                    if index == 15:
                        test.cleanup = lambda: close_pdf_tabs(target)
                    if variant == 'candidate':
                        from bench.arch.reliable_queue import configure
                    else:
                        configure = None
                    outside_before = output_files()
                    row = harness.run(test, index, repeat, *harness.MODELS['deepseek'], target,
                                      variant, 'qwen', 'screen', args.out, configure=configure)
                    outside_after = output_files()
                    row['unexpected_output_files'] = sorted(path for path in outside_before.keys() | outside_after.keys()
                                                           if outside_before.get(path) != outside_after.get(path))
                    if row['unexpected_output_files']:
                        row['outcome_passed'] = row['passed'] = False
                    row['suite'] = test.suite
                    row['passed'] = row['passed'] and all(t['finished'] for t in row['turns'])
                    row['fixture'] = f'round-{repeat}/{variant}'
                    rows.append(row)
                    (args.out/'report.json').write_text(json.dumps(rows, indent=1))
                    print(json.dumps({k: row[k] for k in ('model','test','passed','detail','turns','errors')}), flush=True)
    finally:
        harness.reset()
        harness.kwin(harness.minimize_js(False))
        target.close()
        server.shutdown()
        # Keep independent artifacts, including wrong/duplicate files, alongside the action logs.
        import shutil
        shutil.copytree(root, args.out/'files')
        shutil.rmtree(root)
    summaries = {variant: summary([r for r in rows if r['model'] == variant]) for variant in variants}
    (args.out/'summary.json').write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
