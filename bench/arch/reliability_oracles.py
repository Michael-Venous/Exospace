"""Independent file outcomes; no desktop, browser, model or credentials required."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

def exact_files(root: Path, expected: dict[str, str | bytes]) -> tuple[bool, str]:
    """Exact inventory AND contents; catches repeats, copies, wrong names and wrong directories."""
    actual = {}
    for p in sorted(root.rglob('*')):
        if p.is_file() and not (p.name.startswith('.') and p.name.endswith('.kate-swp')):
            actual[str(p.relative_to(root))] = p.read_bytes()
    # Ignore layout whitespace for dictated prose; repeated words/lines, punctuation and case matter.
    # Bytes expectations preserve existing files exactly.
    same = set(actual) == set(expected)
    expected_dirs = {str(parent) for name in expected for parent in Path(name).parents if str(parent) != '.'}
    actual_dirs = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_dir()}
    same &= actual_dirs == expected_dirs
    for name, content in expected.items():
        value = actual.get(name)
        if isinstance(content, str):
            same &= value is not None and value.decode('utf-8', errors='replace').split() == content.split()
        else:
            same &= value == content
    detail = {name: (data.decode('utf-8', errors='replace')[:160] if not data.startswith(b'%PDF')
                     else hashlib.sha256(data).hexdigest()) for name, data in actual.items()}
    return bool(same), json.dumps(detail)


@dataclass
class FileOracle:
    root: Path
    expected: dict[str, str | bytes]

    def __call__(self, test):
        return exact_files(self.root, self.expected)

