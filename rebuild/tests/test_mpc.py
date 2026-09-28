import pytest
from shipnav.controllers.mpc import MPC
from shipnav.dynamics import Vessel
from shipnav.horizon import horizon_steps
from shipnav.maps import SeaMap
from shipnav.simulation import Traffic, run_episode


def test_mpc_has_explicit_infeasibility_and_bounded_command():
    m = MPC()
    m.set_context(SeaMap((0, 0, 24, 24)), Vessel(2, 2, 0, 0), [])
    u = m((2, 2), (0, 0), (20, 2), [], .5, 1, .25)
    assert u[0] > 0 and not m.solver_failed
    points = horizon_steps(.25, 'marine', 1)+1  # Vessel context -> shared marine horizon
    bad = [{'radius': 100., 'points': [(2, 2)]*points, 'margins': [0.]*points}]
    m.set_context(SeaMap((0, 0, 24, 24)), Vessel(2, 2, 0, 0), bad)
    m((2, 2), (0, 0), (20, 2), [], .5, 1, .25)
    assert m.solver_failed


def test_mpc_starts_with_declared_defaults_before_any_context():
    m = MPC()
    assert m.sea is None and m.vessel is None and m.predictions == [] and m.solver_failed is False


def test_mpc_call_before_set_context_raises_clear_error():
    m = MPC()
    with pytest.raises((RuntimeError, ValueError)):
        m((2, 2), (0, 0), (20, 2), [], .5, 1, .25)


@pytest.mark.parametrize('dynamics', ['holonomic', 'marine'])
def test_mpc_runs_through_episode_with_context_set_each_step(dynamics):
    sea = SeaMap((0, 0, 24, 24))
    route = [(2, 2), (10, 2)]
    mpc = MPC()
    result = run_episode(sea, route, [], mpc, dt=.5, limit=5, radius=.5, speed=1, dynamics=dynamics)
    assert result['diagnostics']
    for d in result['diagnostics']:
        assert isinstance(d['solver_failed'], bool)
        assert isinstance(d['deadline_miss'], bool)
    # A clear, unobstructed run toward a reachable goal should never report infeasibility.
    assert not any(d['solver_failed'] for d in result['diagnostics'])


def test_mpc_logs_solver_failed_true_without_masking_episode_truth():
    # A traffic radius far larger than the whole reachable neighbourhood makes every
    # MPC candidate infeasible; the episode must still report the honest outcome
    # (solver_failed True is recorded, and the real collision truth-check still fires)
    # rather than silently reporting success.
    sea = SeaMap((0, 0, 100, 100))
    ship = Traffic((66, 50), (50, 50), speed=8, radius=15)
    mpc = MPC()
    result = run_episode(sea, [(50, 50), (60, 50)], [ship], mpc, dt=.25, limit=.25, radius=.5, speed=1.0)
    assert any(d['solver_failed'] for d in result['diagnostics'])
    # The target closes 2 m in the single .25 s step from .5 m clearance: the
    # truth check must report the collision the failed solve could not avoid.
    assert result['status'] == 'collision'
    assert result['diagnostics'][-1]['ship_collision']


@pytest.mark.parametrize('dynamics', ['holonomic', 'marine'])
def test_mpc_reaches_the_goal_in_open_water(dynamics):
    sea = SeaMap((0, 0, 24, 24))
    result = run_episode(sea, [(3, 12), (21, 12)], [], MPC(), dynamics=dynamics)
    assert result['status'] == 'success'
    assert result['elapsed'] < 30
    assert not any(d['solver_failed'] for d in result['diagnostics'])


@pytest.mark.parametrize('dynamics', ['holonomic', 'marine'])
def test_mpc_follows_intermediate_waypoints_to_the_final_goal(dynamics):
    sea = SeaMap((0, 0, 24, 24))
    result = run_episode(sea, [(3, 3), (12, 3), (12, 12)], [], MPC(), dynamics=dynamics)
    assert result['status'] == 'success'
    assert result['frames'][-1]['goal_index'] == 2


def test_mpc_does_not_stop_short_next_to_the_goal():
    # The former cost (distance of the 12-step endpoint) made every moving
    # candidate overshoot a goal .625 m away, so "stop" won forever.
    mpc = MPC()
    mpc.set_context(SeaMap((0, 0, 24, 24)), None, [])
    command = mpc((20.375, 12.), (0., 0.), (21., 12.), [], .5, 1., .25)
    assert command[0] > 0 and abs(command[1]) < 1e-9


def test_mpc_holds_at_a_final_goal_next_to_land_instead_of_rejecting_arrival():
    # Goal 1.1 m off land: moving candidates only look infeasible if the rollout
    # runs past the final goal; with the shared hold-at-goal rollout they are fine.
    sea = SeaMap((0, 0, 40, 24), ((30, 0, 40, 24),))
    result = run_episode(sea, [(20, 12), (28.9, 12)], [], MPC())
    assert result['status'] == 'success'
    assert not any(d['solver_failed'] for d in result['diagnostics'])
