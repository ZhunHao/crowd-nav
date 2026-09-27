from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def compare(old, new):
    assert len(old) == len(new)
    for a, b in zip(old, new):
        np.testing.assert_allclose(a['action'], b['action'], rtol=0, atol=1e-7)
        assert len(a['calls']) == len(b['calls']) > 0
        for x, y in zip(a['calls'], b['calls']):
            np.testing.assert_allclose(x['input'], y['input'], rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(x['value'], y['value'], rtol=1e-5, atol=1e-6)


def test_cpu_inference_parity():
    old = json.loads((ROOT/'reference/inference.json').read_text())
    new = json.loads((ROOT/'migration/inference.json').read_text())
    assert len(old) == len(new) == 6
    assert [(r['n'],r['vx']) for r in old] == [(r['n'],r['vx']) for r in new]
    compare(old,new)


def test_expanded_cpu_inference_parity():
    old = json.loads((ROOT/'migration/expanded-legacy.json').read_text())
    new = json.loads((ROOT/'migration/expanded-modern.json').read_text())
    assert [r['id'] for r in old] == [r['id'] for r in new]
    compare(old,new)
    for a,b in zip(old,new):
        np.testing.assert_allclose(a['candidate_values'],b['candidate_values'],rtol=1e-5,atol=1e-6)


def test_native_orca_runs():
    from shipnav.compat.crowd_sim.envs.policy.orca import ORCA
    from shipnav.compat.crowd_sim.envs.utils.state import FullState, ObservableState, JointState
    policy = ORCA()
    policy.time_step = .25
    own = FullState(0,0,0,0,.5,10,0,1,0)
    action = policy.predict(JointState(own,[ObservableState(2,0,-1,0,.5)]))
    assert np.isfinite(action).all()
    assert np.linalg.norm(action) <= 1.000001
    assert action.vx < 1  # head-on neighbour requires avoidance


def test_bounded_replay_deterministic():
    from shipnav.replay import replay
    scenario = {'id':'empty', 'own':[0,0,0,0,.1,1,0,1,0], 'others':[]}
    a = replay(scenario, max_steps=8)
    b = replay(scenario, max_steps=8)
    assert a == b
    assert a['status'] == 'goal'
    assert a['trace'][-1]['t'] == 1
    assert len(a['trace']) == 4


def test_complete_controller_trace_parity():
    old = json.loads((ROOT/'migration/replay-legacy.json').read_text())
    new = json.loads((ROOT/'migration/replay-modern.json').read_text())
    assert len(old) == len(new) == 3
    for a,b in zip(old,new):
        assert (a['id'],a['status'],a['max_steps']) == (b['id'],b['status'],b['max_steps'])
        assert len(a['trace']) == len(b['trace'])
        for x,y in zip(a['trace'],b['trace']):
            assert x['t'] == y['t']
            np.testing.assert_allclose(x['action'],y['action'],rtol=0,atol=1e-7)
            np.testing.assert_allclose(x['own'],y['own'],rtol=0,atol=1e-7)
            np.testing.assert_allclose(x['others'],y['others'],rtol=0,atol=1e-7)
            np.testing.assert_allclose(x['clearance'],y['clearance'],rtol=0,atol=1e-7)
