"""Integrity, denominator and fixed-reference regression tests for evidence tooling."""
import gzip
import importlib.util
import json
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

def module():
    spec = importlib.util.spec_from_file_location('baseline_evidence', ROOT/'tools/baseline_evidence.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_archive_retains_exact_bytes_and_rejects_tampering(tmp_path):
    m = module()
    original = b'{ "unicode": "\\u2603", "x": 1.00 }\n'
    source = tmp_path/'trace.json'
    source.write_bytes(original)
    entry = m.archive(source)
    assert not source.exists()
    assert gzip.decompress((tmp_path/entry['file']).read_bytes()) == original
    assert entry['raw_sha256'] == sha256(original).hexdigest()
    assert m.load_verified(tmp_path, entry) == {'unicode': '\u2603', 'x': 1.0}
    stored = (tmp_path/entry['file']).read_bytes()
    source.write_bytes(original)
    assert m.archive(source) == entry
    assert (tmp_path/entry['file']).read_bytes() == stored
    (tmp_path/entry['file']).write_bytes(gzip.compress(b'{}'))
    with pytest.raises(ValueError, match='hash'):
        m.load_verified(tmp_path, entry)


def test_reference_metrics_use_frozen_route_and_physical_units():
    m = module()
    run = {'map': {'metadata': {'model_scale': {'length_m': 10, 'speed_mps': 5}}},
           'frames': [{'position': [0, 2]}, {'position': [4, 2]}],
           'route': [[0, 2], [4, 2]], 'distance': 4}
    result = m.reference_metrics(run, [[0, 0], [4, 0]])
    assert result == {'reference_cross_track_mean_m': 20, 'reference_cross_track_max_m': 20,
                      'reference_detour_ratio': 1}
    assert m.reference_metrics(run, None)['reference_detour_ratio'] is None


def test_joint_eligibility_retains_failure_in_primary_denominator():
    m = module()
    rows = [dict(scenario_hash=i, variant=v, status=s, elapsed_s=t,
                 ship_collision=None, land_collision=None)
            for i, v, s, t in [('1', 'a', 'success', 2), ('1', 'b', 'timeout', 100),
                               ('2', 'a', 'planning_failure', None), ('2', 'b', 'success', 3),
                               ('3', 'a', 'success', 6), ('3', 'b', 'success', 4)]]
    result = m.compare(rows, 'a', 'b', samples=100)
    assert result['success']['pairs'] == 3
    assert result['success']['difference'] == 0
    assert result['completion_time_s']['pairs'] == 1
    assert result['completion_time_s']['difference'] == 2
    assert result['excluded_joint_success'] == ['1', '2']
    with pytest.raises(ValueError, match='matching'):
        m.compare(rows[:-1], 'a', 'b')


def test_timing_categories_and_physical_deadlines():
    m = module()
    run = {'inference_ms': [600, 2, 3, 4, 5, 6],
           'diagnostics': [{'decision_ms': v} for v in [700, 12, 13, 14, 15, 16]],
           'settings': {'dt': .25},
           'map': {'metadata': {'model_scale': {'length_m': 10, 'speed_mps': 5}}}}
    timing = m.timing_samples(run)
    assert timing['cold_instance']['inference'] == [600]
    assert timing['transition']['decision'] == [12, 13, 14, 15]
    assert timing['warmed']['decision'] == [16]
    assert m.summarize_timing([600, 2], [500, 500])['deadline_misses'] == 1


def test_family_perturbations_change_physical_initial_conditions():
    m = module()
    source = json.loads((ROOT/'scenarios/head_on.json').read_text())
    cases = [m.perturb(source, i) for i in range(1, 9)]
    assert len({json.dumps({k: c[k] for k in ('start', 'goal', 'traffic')}, sort_keys=True)
                for c in cases}) == 8
    assert all(c['map'] == source['map'] for c in cases)
    assert source['start'] == [3, 12]


def test_extension_preserves_originals_and_distinct_voyages(tmp_path):
    spec = importlib.util.spec_from_file_location('baseline_freeze', ROOT/'tools/baseline_freeze.py')
    m = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(ROOT/'tools'))
    spec.loader.exec_module(m)
    before = {p.name: sha256(p.read_bytes()).hexdigest() for p in (ROOT/'scenarios').glob('*.json')}
    manifest = m.generate(tmp_path, real_maps=False)
    assert len(manifest['splits']['test']) == 99
    assert before == {p.name: sha256(p.read_bytes()).hexdigest() for p in (ROOT/'scenarios').glob('*.json')}
    cases = [module().load_verified(tmp_path, e) for e in manifest['splits']['test']]
    physical = [{k: c[k] for k in ('map', 'start', 'goal', 'traffic')} for c in cases]
    assert len({json.dumps(c, sort_keys=True) for c in physical}) == 99
    assert len([e for e in manifest['splits']['test'] if e['reference_route'] is None]) == 9
    other = tmp_path/'again'
    repeat = m.generate(other, real_maps=False)
    assert repeat == manifest


def test_one_unit_archives_benchmark_failure_and_keeps_raw_identity(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT/'tools'))
    spec = importlib.util.spec_from_file_location('baseline_run', ROOT/'tools/baseline_run.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    scenario = json.loads((ROOT/'scenarios/unreachable.json').read_text())
    row = m.run_unit(scenario, {'id': 'unreachable', 'reference_route': None}, 'direct_reference', '', tmp_path)
    assert row['status'] == 'planning_failure'
    assert row['trace_hash'] == row['archive']['raw_sha256']
    assert row['trace_file'].endswith('.json.gz')
    assert not list(tmp_path.glob('*.json'))
    run = module().load_verified(tmp_path, row['archive'])
    assert run['status'] == 'planning_failure'


def test_real_map_extension_respects_geographic_split_and_corridor(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT/'tools'))
    from baseline_freeze import real_cases
    from shipnav.maps import SeaMap
    from freeze_scenarios import validate
    scenarios = list(real_cases('singapore-southern-islands.json', 'south-island-detour', 2, 'test', 8000))
    for name, scenario in scenarios:
        validate(scenario)
        assert scenario['split'] == 'test'
        assert scenario['family'] == 'southern_corridor'
        assert scenario['map']['metadata']['model_scale']['profile'] == 'harbour_craft'
        assert len(scenario['traffic']) in (1, 2)
        assert all(t['kind'] == 'course_change' for t in scenario['traffic'])
    assert scenarios[0][1]['start'] != scenarios[1][1]['start']


def test_unit_uses_shared_reference_timeout_instead_of_own_planner_length(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT/'tools'))
    from baseline_run import run_unit
    scenario = json.loads((ROOT/'scenarios/head_on.json').read_text())
    scenario['traffic'] = []
    scenario['map']['metadata']['model_scale'] = {'length_m': 10., 'speed_mps': 5.,
                                                'radius_m': 5., 'margin_m': 10., 'resolution_m': 50.}
    # Fixed valid reference is38 units long, while direct own route is18.
    reference = [[3, 12], [3, 22], [21, 22], [21, 12]]
    row = run_unit(scenario, {'id': 'limit', 'reference_route': reference}, 'direct_reference', '', tmp_path)
    run = module().load_verified(tmp_path, row['archive'])
    assert run['settings']['limit'] == 114


def test_audit_rejects_changed_recorded_limit_and_accepts_unexecuted_failure():
    import sys
    sys.path.insert(0, str(ROOT/'tools'))
    import baseline_run
    assert hasattr(baseline_run, 'verify_limit')
    with pytest.raises(ValueError, match='limit'):
        baseline_run.verify_limit({'settings': {'limit': 100}}, 114)
    baseline_run.verify_limit({'settings': {'limit': 114}}, 114)
    baseline_run.verify_limit({'status': 'planning_failure'}, 114)
