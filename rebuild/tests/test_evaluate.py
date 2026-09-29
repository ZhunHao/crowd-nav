import csv
import json

import pytest

from shipnav.evaluate import summarize, evaluate
from shipnav.maps import SeaMap


def test_failures_do_not_improve_average_completion_time():
    rows = [{'status': 'success', 'elapsed': 10, 'distance': 9, 'inference_ms': [2]},
            {'status': 'collision', 'elapsed': 1, 'distance': 1, 'inference_ms': [4]},
            {'status': 'error', 'error': 'placement failed'}]
    summary = summarize(rows)
    assert summary['runs'] == 3 and summary['successes'] == 1
    assert summary['mean_success_time_s'] == 10
    assert summary['collisions'] == 1 and summary['errors'] == 1
    assert summary['mean_step_inference_ms'] == 3


def test_evaluation_pairs_seed_and_scenario_and_saves_every_run(tmp_path):
    calls = []
    def runner(map_data, start, goal, model, policy, global_goals, seed, count):
        calls.append((policy, global_goals, seed, count))
        return {'status': 'success', 'elapsed': 10, 'distance': 9,
                'inference_ms': [1], 'traffic_definitions': [{'seed': seed, 'count': count}]}
    rows = evaluate(SeaMap((0, 0, 24, 24)).to_dict(), '', tmp_path,
                    seeds=range(2), counts=(5,), runner=runner)
    assert len(calls) == 8 and len(rows) == 4
    assert len(list(tmp_path.glob('*.json'))) == 8
    assert (tmp_path/'summary.csv').exists()
    assert all(r['runs'] == 2 for r in rows)


@pytest.mark.parametrize('exception', [ValueError, FileNotFoundError, RuntimeError])
def test_handled_evaluation_exception_retains_error_json_and_summary_count(tmp_path, exception):
    def runner(map_data, start, goal, model, policy, global_goals, seed, count):
        if policy == 'sarl' and global_goals and seed == 0:
            raise exception('injected placement failure')
        return {'status': 'success', 'elapsed': 10, 'distance': 9,
                'inference_ms': [1], 'traffic_definitions': [{'seed': seed, 'count': count}]}
    rows = evaluate(SeaMap((0, 0, 24, 24)).to_dict(), '', tmp_path,
                    seeds=range(2), counts=(5,), runner=runner)
    retained = json.loads((tmp_path/'global_sarl-n5-seed0.json').read_text())
    assert retained == {'status': 'error', 'error': 'injected placement failure', 'seed': 0, 'count': 5}
    assert len(list(tmp_path.glob('*.json'))) == 8
    summary = next(row for row in rows if row['variant'] == 'global_sarl')
    assert (summary['runs'], summary['successes'], summary['errors']) == (2, 1, 1)
    with (tmp_path/'summary.csv').open() as stream:
        persisted = next(row for row in csv.DictReader(stream) if row['variant'] == 'global_sarl')
    assert (persisted['runs'], persisted['successes'], persisted['errors']) == ('2', '1', '1')


def test_evaluation_rejects_traffic_mismatch_loudly(tmp_path):
    def runner(map_data, start, goal, model, policy, global_goals, seed, count):
        return {'status': 'success', 'elapsed': 10, 'distance': 9, 'inference_ms': [1],
                'traffic_definitions': [{'seed': seed, 'count': count + (policy == 'orca')}]}
    with pytest.raises(AssertionError, match='Paired scenarios differ'):
        evaluate(SeaMap((0, 0, 24, 24)).to_dict(), '', tmp_path,
                 seeds=range(1), counts=(5,), runner=runner)
    assert not (tmp_path/'summary.csv').exists()
    assert not (tmp_path/'global_orca-n5-seed0.json').exists()
