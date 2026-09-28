"""Smoke: every canonical test-split scenario through direct/orca/mpc,
filtered and unfiltered, holonomic, via the service entry point."""
import importlib.util
import json
from pathlib import Path

import pytest

from shipnav.planning import NoPath
from shipnav.service import execute

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ('direct', 'orca', 'mpc')


def _freeze_module():
    spec = importlib.util.spec_from_file_location('freeze_scenarios', ROOT / 'tools/freeze_scenarios.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _test_split():
    splits = json.loads((ROOT / 'scenarios/splits.json').read_text())
    return [json.loads((ROOT / 'scenarios' / e['file']).read_text()) for e in splits['splits']['test']]


def _run(scenario, policy, filtered):
    return execute(scenario['map'], tuple(scenario['start']), tuple(scenario['goal']),
                   policy_name=policy, scenario=scenario, filtered=filtered)


@pytest.mark.parametrize('scenario', _test_split(), ids=lambda s: s['family'])
def test_canonical_scenario_runs_every_policy_filtered_and_unfiltered(scenario):
    module = _freeze_module()
    encounter = scenario['family'] not in ('unreachable', 'shore_goal')
    for policy in POLICIES:
        for filtered in (False, True):
            if scenario['family'] == 'unreachable':
                with pytest.raises(NoPath):
                    _run(scenario, policy, filtered)
                continue
            result = _run(scenario, policy, filtered)
            assert result['status'] in ('success', 'collision', 'timeout'), (policy, filtered)
            if encounter and policy == 'direct' and not filtered:
                # min_dynamic_clearance already subtracts both radii, so the
                # CPA threshold (radii + CPA_MARGIN) is CPA_MARGIN here.
                assert result['min_dynamic_clearance'] < module.CPA_MARGIN
            if not encounter and policy == 'mpc':
                assert result['status'] == 'success', filtered
