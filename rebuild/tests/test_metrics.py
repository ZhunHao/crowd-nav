import json
from hashlib import sha256
from pathlib import Path
import pytest
from shipnav.metrics import metrics, paired_interval, quantile
from shipnav.benchmark import benchmark
from shipnav.maps import SeaMap
from shipnav.planning import InvalidRoute, NoPath, PlanningLimit
from shipnav.service import execute


def test_tail_and_paired_direction():
    assert quantile([1, 2, 3, 100], .99) > 90
    result = paired_interval({'a': 1, 'b': 1}, {'a': 0, 'b': 0})
    assert result['difference'] == result['low'] == result['high'] == 1
    with pytest.raises(ValueError):
        paired_interval({'a': 1}, {'b': 1})


def test_failed_runs_remain_in_denominator(tmp_path):
    scene = {'map': {}, 'start': [0, 0], 'goal': [1, 1], 'seed': 0}
    def broken(*args, **kwargs):
        raise RuntimeError('controller failed')
    rows = benchmark([scene], '', tmp_path, {'a': {}, 'b': {}}, runner=broken)
    assert len(rows) == 2 and all(r['status'] == 'error' for r in rows)
    assert len(list(tmp_path.glob('*-scenario.json'))) == 1
    for row in rows:
        assert row['error']
        assert json.loads((tmp_path/row['trace_file']).read_text())['error'] == row['error']
        assert row['no_feasible_actions'] is None
        assert row['ship_collision'] is None
        assert row['domain_time_s'] is None
        trace = tmp_path/row['trace_file']
        assert sha256(trace.read_bytes()).hexdigest() == row['trace_hash']
        assert json.loads(trace.read_text())['scenario_hash'] == row['scenario_hash']


def test_frozen_scenarios_score_filtered_unfiltered_and_unreachable(tmp_path):
    scenes = [json.loads(Path('scenarios', n).read_text()) for n in ('head_on.json', 'unreachable.json')]
    rows = benchmark(scenes, '', tmp_path, {'direct': dict(policy_name='direct'),
                                          'direct_filtered': dict(policy_name='direct', filtered=True)})
    assert len(rows) == 4
    unfiltered = [r for r in rows if r['variant'] == 'direct' and r['status'] != 'planning_failure']
    assert unfiltered and all(r['no_feasible_actions'] is None for r in unfiltered)
    assert sum(r['status'] == 'planning_failure' for r in rows) == 2
    assert all('cross_track_max_m' in r for r in rows if r['status'] != 'planning_failure')
    assert json.loads((tmp_path/'metrics.json').read_text()) == rows


@pytest.mark.parametrize('error,status', [(NoPath('blocked'), 'planning_failure'),
                                         (PlanningLimit('budget'), 'planning_failure'),
                                         (InvalidRoute('unsafe'), 'planning_failure'),
                                         (RuntimeError('controller failed'), 'error'),
                                         (ValueError('invalid'), 'error')])
def test_benchmark_classifies_planning_failures_and_keeps_error(tmp_path, error, status):
    scene = {'map': SeaMap((0, 0, 10, 10)).to_dict(), 'start': [2, 2], 'goal': [8, 8], 'seed': 0}
    def broken(*args, **kwargs):
        raise error
    row, = benchmark([scene], '', tmp_path, {'direct': {}}, runner=broken)
    assert row['status'] == status and str(error) in row['error']


def test_empty_success_uses_filter_settings_for_feasibility():
    sea = SeaMap((0, 0, 10, 10)).to_dict()
    plain = execute(sea, (2, 2), (2, 2), policy_name='direct', count=0)
    filtered = execute(sea, (2, 2), (2, 2), policy_name='direct', count=0, filtered=True)
    assert metrics(plain)['no_feasible_actions'] is None
    assert metrics(filtered)['no_feasible_actions'] == 0
    assert metrics(plain)['decision_p99_ms'] is None
    assert metrics(plain)['domain_time_s'] == 0
    assert metrics({'status': 'error', 'error': 'x'})['solver_failures'] is None


@pytest.mark.parametrize('bad', [None, float('nan'), float('inf')])
def test_paired_interval_rejects_failed_or_nonfinite_arm_without_dropping_pair(bad):
    with pytest.raises(ValueError):
        paired_interval({'ok': 1, 'failed': bad}, {'ok': 0, 'failed': 0})
    with pytest.raises(ValueError):
        paired_interval({'ok': 1, 'failed': 1}, {'ok': 0, 'failed': bad})


def test_physical_metrics_use_recorded_scale_and_actual_exposure_intervals():
    run = {'status': 'timeout', 'map': {'metadata': {'model_scale': {'length_m': 10., 'speed_mps': 5.}}},
           'settings': {'dt': .25, 'filtered': True}, 'route': [(0, 0), (1, 0)],
           'elapsed': .35, 'distance': .4, 'frames': [{'t': 0}, {'t': .25}, {'t': .35}],
           'diagnostics': [dict(clearance=.05, decision_ms=450, override=True, no_feasible_action=False,
                                ship_collision=False, land_collision=False),
                           dict(clearance=.1, decision_ms=550, override=False, no_feasible_action=True,
                                ship_collision=False, land_collision=False)]}
    result = metrics(run)
    assert result['min_ship_clearance'] == .5
    assert result['domain_time_s'] == .5
    assert result['elapsed_s'] == .7 and result['distance_m'] == 4
    assert result['detour_ratio'] == .4
    assert result['deadline_misses'] == 1
    assert result['no_feasible_actions'] == 1


def test_partial_final_tick_contributes_only_its_actual_exposure_duration():
    run = {'status': 'timeout', 'settings': {'dt': .25, 'filtered': False},
           'frames': [{'t': 0}, {'t': .25}, {'t': .35}],
           'diagnostics': [{'clearance': 2.}, {'clearance': .5}]}
    assert metrics(run)['domain_time_s'] == pytest.approx(.1)
