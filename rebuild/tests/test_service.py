import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from shipnav.maps import SeaMap
from shipnav.planning import NoPath
from shipnav.scenarios import make_scenario
from shipnav.service import execute

_REBUILD_ROOT = Path(__file__).resolve().parent.parent


def test_service_is_deterministic_and_records_its_settings():
    sea = SeaMap((0, 0, 24, 24)).to_dict()
    a = execute(sea, (2, 2), (22, 22), policy_name='direct', count=0)
    b = execute(sea, (2, 2), (22, 22), policy_name='direct', count=0)
    assert a['frames'] == b['frames']
    assert a['status'] == 'success'
    assert a['settings']['query_env'] is False
    assert a['model_hashes'] == {}


def test_paired_planner_global_goals_and_filter_variants_keep_identical_scenario():
    sea = SeaMap((0, 0, 24, 24), ((9, 5, 14, 18),)).to_dict()
    scenario = make_scenario(SeaMap.from_dict(sea), (2, 2), (22, 22), 3, 3)
    original = copy.deepcopy(scenario)
    results = []
    for planner in ('astar_smooth', 'theta'):
        for global_goals in (True, False):
            for filtered in (False, True):
                result = execute(sea, (2, 2), (22, 22), policy_name='direct', scenario=scenario,
                                 planner=planner, global_goals=global_goals, filtered=filtered)
                results.append((planner, global_goals, filtered, result))
    assert scenario == original
    hashes = {result['scenario_hash'] for *_, result in results}
    assert len(hashes) == 1
    for planner, global_goals, filtered, result in results:
        assert result['settings']['planner'] == planner
        assert result['settings']['global_goals'] == global_goals
        assert result['settings']['filtered'] == filtered


def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError, match='Policy must'):
        execute(SeaMap((0, 0, 24, 24)).to_dict(), (2, 2), (22, 22), policy_name='missing', count=0)


def test_invalid_map_schema_is_rejected():
    with pytest.raises(ValueError):
        execute({'schema': 2}, (0, 0), (1, 1), policy_name='direct', count=0)


def test_json_round_tripped_scenario_map_is_accepted():
    sea = SeaMap((0, 0, 24, 24), ((9, 5, 14, 18),)).to_dict()
    scenario = make_scenario(SeaMap.from_dict(sea), (2, 2), (22, 22), 2, 1)
    round_tripped = json.loads(json.dumps(scenario))
    result = execute(round_tripped['map'], tuple(round_tripped['start']), tuple(round_tripped['goal']),
                     policy_name='direct', scenario=round_tripped)
    assert result['status']


def test_genuinely_different_map_is_rejected():
    sea = SeaMap((0, 0, 24, 24)).to_dict()
    scenario = make_scenario(SeaMap.from_dict(sea), (2, 2), (22, 22), 0, 1)
    different_map = SeaMap((0, 0, 30, 30)).to_dict()
    with pytest.raises(ValueError, match='differ'):
        execute(different_map, (2, 2), (22, 22), policy_name='direct', scenario=scenario)


def test_unreachable_scenario_raises_no_path():
    scenario = json.loads((_REBUILD_ROOT/'scenarios/unreachable.json').read_text())
    with pytest.raises(NoPath):
        execute(scenario['map'], tuple(scenario['start']), tuple(scenario['goal']),
               policy_name='direct', scenario=scenario)


def test_reactive_scenario_wraps_plain_traffic_as_reactive():
    scenario = json.loads((_REBUILD_ROOT/'scenarios/noncooperative_reactive.json').read_text())
    result = execute(scenario['map'], tuple(scenario['start']), tuple(scenario['goal']),
                     policy_name='direct', scenario=scenario)
    assert result['settings']['traffic_model'] == 'reactive'


def test_reactive_scenario_rejects_course_change_entries():
    scenario = json.loads((_REBUILD_ROOT/'scenarios/course_change.json').read_text())
    scenario = {**scenario, 'traffic_mode': 'reactive'}
    with pytest.raises(ValueError, match='course_change'):
        execute(scenario['map'], tuple(scenario['start']), tuple(scenario['goal']),
               policy_name='direct', scenario=scenario)


def test_mpc_policy_path_executes():
    sea = SeaMap((0, 0, 24, 24)).to_dict()
    result = execute(sea, (2, 2), (10, 2), policy_name='mpc', count=0)
    assert result['settings']['policy'] == 'mpc'
    assert result['status']


def test_marine_dynamics_path_executes():
    sea = SeaMap((0, 0, 24, 24)).to_dict()
    result = execute(sea, (2, 2), (10, 2), policy_name='direct', count=0, dynamics='marine')
    assert result['settings']['dynamics'] == 'marine'
    assert result['status']


def test_result_includes_provenance_metadata():
    sea = SeaMap((0, 0, 24, 24)).to_dict()
    result = execute(sea, (2, 2), (10, 2), policy_name='direct', count=0)
    prov = result['provenance']
    assert {'git_revision', 'dirty_diff_sha256', 'git_error', 'settings_sha256',
            'python_version', 'python_implementation', 'platform', 'uv_lock_sha256'} <= prov.keys()


def test_cli_runs_frozen_scenario_and_exits_zero(tmp_path):
    output = tmp_path/'run.json'
    run = subprocess.run([sys.executable, '-m', 'shipnav.service', '--scenario', 'scenarios/head_on.json',
                          '--policy', 'direct', '--output', str(output)],
                         cwd=_REBUILD_ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(output.read_text())['status']


def test_cli_unreachable_scenario_exits_two(tmp_path):
    output = tmp_path/'run.json'
    run = subprocess.run([sys.executable, '-m', 'shipnav.service', '--scenario', 'scenarios/unreachable.json',
                          '--policy', 'direct', '--output', str(output)],
                         cwd=_REBUILD_ROOT, capture_output=True, text=True)
    assert run.returncode == 2
    assert not output.exists()
