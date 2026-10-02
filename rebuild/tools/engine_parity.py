"""Timing-free digests of complete episodes, captured before the step-engine refactor.

Run `python tools/engine_parity.py` once on the pre-refactor source to write the
fixture; afterwards tests/test_engine_parity.py asserts the digests never change.
"""
from hashlib import sha256
from pathlib import Path
import json

from shipnav.scale import map_options
from shipnav.service import execute

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/'tests/fixtures/engine_parity.json'
SARL = ROOT.parent/'CrowdNav-20250813-DIP/crowd_nav/data/output_trained'
DEGRADED = {'noise': .1, 'delay': .5, 'dropout': .1}
TIMING = {'decision_ms', 'deadline_miss'}
# name: (scenario file, policy, dynamics, filtered, observation)
CASES = {
    'crossing-direct-holonomic': ('crossing.json', 'direct', 'holonomic', False, {}),
    'crossing-direct-filtered': ('crossing.json', 'direct', 'holonomic', True, {}),
    'narrow-direct-filtered-degraded': ('narrow_passage.json', 'direct', 'holonomic', True, DEGRADED),
    'reactive-direct-filtered': ('noncooperative_reactive.json', 'direct', 'holonomic', True, {}),
    'course-change-mpc-marine-filtered': ('course_change.json', 'mpc', 'marine', True, {}),
    'head-on-mpc-marine-unfiltered': ('head_on.json', 'mpc', 'marine', False, {}),
    'crossing-sarl-filtered': ('crossing.json', 'sarl', 'holonomic', True, {}),
}
NEEDS_CHECKPOINT = {name for name, case in CASES.items() if case[1] == 'sarl'}


def digest(name):
    file, policy, dynamics, filtered, observation = CASES[name]
    scenario = json.loads((ROOT/'scenarios'/file).read_text())
    run = execute(scenario['map'], scenario['start'], scenario['goal'], str(SARL), policy,
                  scenario=scenario, dynamics=dynamics, filtered=filtered, observation=observation,
                  **map_options(scenario['map']))
    kept = {key: run.get(key) for key in ('status', 'elapsed', 'distance', 'min_dynamic_clearance',
                                          'frames', 'terminal_speed', 'runout_min_clearance')}
    kept['diagnostics'] = [{k: v for k, v in d.items() if k not in TIMING} for d in run['diagnostics']]
    return sha256(json.dumps(kept, sort_keys=True, default=list).encode()).hexdigest()


if __name__ == '__main__':
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps({name: digest(name) for name in CASES}, indent=2, sort_keys=True)+'\n')
    print(f'Wrote {len(CASES)} digests to {FIXTURE}')
