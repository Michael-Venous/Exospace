"""Replay labelled, recorded visual outcomes without controlling a desktop or revealing oracle labels.

Manifest entries: name, expected (bool), goal, claim, steps, first/recent/last image paths.
Labels must come from independent file/history checks. This diagnoses the checker; it does
not measure recovery or end-to-end latency. --history adds chronological observations.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from bench.arch.reliable_queue import ScreenChecks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--history', action='store_true')
    parser.add_argument('--side', type=int, default=1280)
    parser.add_argument('--model', choices=['qwen', 'deepseek'], default='qwen')
    args = parser.parse_args()
    rows = []
    checks = ScreenChecks([], args.side, args.model)
    try:
        for index, item in enumerate(json.loads(args.manifest.read_text())):
            if args.history:
                for frame in [item['first'], *item.get('recent', [])]:
                    checks.observe(Path(frame).read_bytes(), index)
            started = time.monotonic()
            row = {'name': item['name'], 'expected': item['expected']}
            try:
                row['accepted'] = checks.completion(Path(item['last']).read_bytes(), item['goal'], item['steps'],
                                                    evidence=item['claim']) == 1.0
                row['reason'] = checks.last_reason
                row['status'] = checks.last_status
                row['observed'] = checks.last_observed
            except Exception as exc:
                row.update(accepted=False, error=f'{type(exc).__name__}: {exc}'[:200])
            row['s'] = round(time.monotonic()-started, 3)
            rows.append(row)
            print(json.dumps(row), flush=True)
            args.out.write_text(json.dumps(rows, indent=2))
    finally:
        checks.close()


if __name__ == '__main__':
    main()
