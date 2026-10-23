"""Post-freeze compositions: keep their results separate from the development suite.

The candidate prompt/architecture is frozen before these are run. Same transcript
pacing, native apps, independent oracles and outside-output guard as the main suite.

  ... withkeys.py bench.arch.laptop_compositions --candidate --out work/arch/compositions
"""
from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

from bench.arch import laptop_reliability as suite
from bench.arch import laptop_tests as harness
from bench.arch.reliability_oracles import FileOracle


def words(text: str) -> str:
    return ' '.join(re.findall(r'\w+', text.lower()))


def cases(root: Path, url: str, variation: int = 0) -> list[harness.Test]:
    renamed = root/'spaced-rename'
    nested = root/'nested-folders'
    copied = root/'copy-close'
    for folder in (renamed, nested, copied):
        folder.mkdir()
    original = b'Meet at the north gate.\n'
    decoy = b'This is the old route.\n'
    (renamed/'route map.txt').write_bytes(original)
    (renamed/'route map - old.txt').write_bytes(decoy)
    plan = b'Tools: 18\nTravel: 27\n'
    (copied/'seed.txt').write_bytes(plan)
    rename_files = FileOracle(renamed, {'Walk Notes.txt': original, 'route map - old.txt': decoy})
    copy_files = FileOracle(copied, {'seed.txt': plan, 'Budget Draft.txt': plan})

    def renamed_opened_read(test):
        ok, detail = rename_files(test)
        caption = harness.active_caption()
        spoken = ' '.join(test.data['said'])
        return (ok and caption.startswith('Walk Notes.txt') and words(original.decode()) in words(spoken),
                json.dumps({'files': json.loads(detail), 'active': caption, 'said': spoken}))
    renamed_opened_read.file_oracle = rename_files

    def two_folders(test):
        dirs = {str(p.relative_to(nested)) for p in nested.rglob('*') if p.is_dir()}
        files = [str(p.relative_to(nested)) for p in nested.rglob('*') if p.is_file() and p.name != '.directory']
        caption = harness.active_caption()
        return (dirs == {'Sunday Plan', 'Sunday Plan/receipts'} and not files and caption.startswith('receipts'),
                json.dumps({'dirs': sorted(dirs), 'files': files, 'active': caption}))

    def copy_closed(test):
        ok, detail = copy_files(test)
        left = [w['caption'] for w in harness.windows() if w['id'] not in harness.KEEP and
                ('kwrite' in w['cls'].lower() or 'kate' in w['cls'].lower())]
        return ok and not left, json.dumps({'files': json.loads(detail), 'editor_windows': left})
    copy_closed.file_oracle = copy_files

    def tabs_kept(test):
        state = suite.page_state()
        paths = sorted(p.get('path', '') for p in state['tabs'])
        active = next((p for p in state['tabs'] if p.get('visible') == 'visible'), {})
        return paths == ['/birds.html', '/plants.html'] and active.get('path') == '/birds.html', json.dumps(state)

    def files(folder):
        return f'setsid -f dolphin --new-window {shlex.quote(str(folder))} >/dev/null 2>&1; sleep 2'

    return [
        harness.Test('spaced-rename-open-read',
            ['Rename route map.txt to Walk Notes.txt, open it in a text editor, and read me what it says.'],
            files(renamed), renamed_opened_read, suite='steps', turn_s=90),
        harness.Test('nested-folders',
            ['Make a folder called Sunday Plan here. Inside it make a folder called receipts, and leave receipts open.'],
            files(nested), two_folders, suite='steps', turn_s=90),
        harness.Test('unchanged-copy-close',
            [f'I need a second copy of this note in {copied}, called Budget Draft.txt. Keep the original unchanged, and close the editor when you finish.'],
            f'setsid -f kwrite {shlex.quote(str(copied/"seed.txt"))} >/dev/null 2>&1; sleep 2',
            copy_closed, turn_s=90),
        harness.Test('tab-followup', ['Open the Plants collection.',
            'Actually, keep this page and show me the Birds collection in another tab.'],
            harness.CHROME.format(urls=url+'/index.html'), tabs_kept, turn_s=60),
    ]


if __name__ == '__main__':
    suite.main(cases, range(1, 5))
