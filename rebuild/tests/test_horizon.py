import pytest

from shipnav.dynamics import Vessel, motion_from
from shipnav.horizon import HOLONOMIC_STEPS, horizon_steps
from shipnav.maps import SeaMap
from shipnav.observations import Observer
from shipnav.policies import Direct
from shipnav.prediction import predict
from shipnav.safety import assess, choose
from shipnav.simulation import Traffic, run_episode


def test_holonomic_keeps_twelve_steps_marine_covers_stopping_time():
    assert HOLONOMIC_STEPS == 12
    assert horizon_steps(.25, 'holonomic', 1.) == 12
    # speed 1 m/s, deceleration .2 m/s^2 -> 5 s to stop -> 20 steps at dt=.25, +1.
    assert horizon_steps(.25, 'marine', 1.) == 21
    assert horizon_steps(.125, 'marine', 1.) == 41
    # A slow vessel never drops below the holonomic floor.
    assert horizon_steps(.25, 'marine', .1) == 12
    assert horizon_steps(.25, 'marine', 1.)*.25 >= 1./.2


def test_horizon_rejects_bad_inputs():
    with pytest.raises(ValueError):
        horizon_steps(0., 'holonomic', 1.)
    with pytest.raises(ValueError):
        horizon_steps(.25, 'hovercraft', 1.)


def test_assess_rejects_prediction_rollout_length_mismatch_instead_of_truncating():
    sea = SeaMap((0, 0, 24, 24))
    path = [(2., 2.)]*22
    short = predict([{'id': 0, 'position': (10., 10.), 'velocity': (0., 0.), 'radius': .6, 'margin': 0.}], .25, steps=12)
    with pytest.raises(ValueError, match='horizon'):
        assess(sea, path, short, .5)


def test_marine_choose_rolls_out_the_shared_marine_horizon():
    sea = SeaMap((0, 0, 100, 100))
    motion = motion_from(Vessel(50, 50, 0., 1.), 1.)
    decision = choose(sea, (50, 50), (1., 0.), [], .5, 1., .25, motion=motion)
    assert len(decision['path']) == horizon_steps(.25, 'marine', 1.)+1


def test_marine_episode_predictions_and_filter_share_one_horizon():
    sea = SeaMap((0, 0, 40, 24))
    ship = Traffic((30, 20), (30, 4))
    result = run_episode(sea, [(3, 12), (20, 12)], [ship], Direct(), dynamics='marine',
                         filtered=True, limit=1., observer=Observer())
    expected = horizon_steps(.25, 'marine', 1.)+1
    for d in result['diagnostics']:
        assert all(len(p['points']) == len(p['margins']) == expected for p in d['predictions'])
        assert len(d['path']) == expected


def test_near_shore_marine_probe_reports_terminal_speed_and_honest_runout():
    # Land at x>=30; goal 1.1 m off the shore. Direct arrives still carrying way
    # (bounded deceleration), so the zero-command run-out would ground. Success
    # semantics are unchanged; the run-out is reported, not hidden.
    sea = SeaMap((0, 0, 40, 24), ((30, 0, 40, 24),))
    result = run_episode(sea, [(20, 12), (28.9, 12)], [], Direct(), dynamics='marine')
    assert result['status'] == 'success'
    assert result['terminal_speed'] > .5
    assert result['runout_min_clearance'] < 0


def test_open_water_marine_runout_is_clear_and_holonomic_has_no_runout_fields():
    sea = SeaMap((0, 0, 40, 24))
    result = run_episode(sea, [(3, 12), (12, 12)], [], Direct(), dynamics='marine')
    assert result['status'] == 'success'
    assert result['terminal_speed'] > 0
    assert result['runout_min_clearance'] > 0
    holonomic = run_episode(sea, [(3, 12), (12, 12)], [], Direct())
    assert 'terminal_speed' not in holonomic and 'runout_min_clearance' not in holonomic


def test_failed_marine_run_has_no_runout_clearance():
    sea = SeaMap((0, 0, 40, 24))
    result = run_episode(sea, [(3, 12), (30, 12)], [], Direct(), dynamics='marine', limit=2)
    assert result['status'] == 'timeout'
    assert result['terminal_speed'] > 0
    assert result['runout_min_clearance'] is None
