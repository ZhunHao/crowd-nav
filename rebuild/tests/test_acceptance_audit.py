"""The smoke audit requires every summary group exactly once."""
import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from audit_acceptance import audit


def smoke_directory(directory, expected):
    counts, seeds = ([5], 2) if expected == 8 else ([5, 8, 12, 15], 10)
    rows = []
    for name in ('global_sarl', 'final_goal_sarl', 'global_direct', 'global_orca'):
        for count in counts:
            for seed in range(seeds):
                (directory/f'{name}-n{count}-seed{seed}.json').write_text(json.dumps(
                    {'status': 'error', 'error': 'retained placement failure'}))
            rows.append(dict(variant=name, traffic_count=count, runs=seeds, successes=0,
                             collisions=0, timeouts=0, errors=seeds, mean_success_time_s='',
                             mean_success_distance_m='', mean_step_inference_ms=''))
    return rows


def write_summary(directory, rows):
    with (directory/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize('expected', [8, 160])
@pytest.mark.parametrize('change', ['missing', 'duplicate', 'unexpected'])
def test_audit_rejects_incomplete_or_nonunique_summary_keys(tmp_path, expected, change):
    rows = smoke_directory(tmp_path, expected)
    if change == 'missing':
        rows.pop()
    elif change == 'duplicate':
        rows.append(dict(rows[0]))
    else:
        rows[-1].update(traffic_count=99, runs=0, errors=0)
    write_summary(tmp_path, rows)
    with pytest.raises(AssertionError, match='summary'):
        audit(tmp_path, expected)


@pytest.mark.parametrize('expected,groups', [(8, 4), (160, 16)])
def test_audit_accepts_every_expected_unique_group(tmp_path, expected, groups):
    write_summary(tmp_path, smoke_directory(tmp_path, expected))
    result = audit(tmp_path, expected)
    assert result['summary_rows'] == groups
    assert result['outcomes'] == {'error': expected}
