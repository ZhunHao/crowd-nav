import pytest
from math import dist
from shipnav.maps import SeaMap
from shipnav.simulation import Traffic, run_episode, swept_clearance


def toward(p, v, goal, neighbours, radius, speed, dt):
    d = dist(p, goal)
    return tuple((b-a)/d*min(speed, d/dt) if d else 0 for a, b in zip(p, goal))


def test_waypoints_keep_one_clock_and_continuous_motion():
    sea = SeaMap((0, 0, 20, 20))
    route = [(2, 2), (8, 2), (8, 8)]
    result = run_episode(sea, route, [], toward)
    assert result['status'] == 'success'
    frames = result['frames']
    assert [f['t'] for f in frames] == sorted(set(f['t'] for f in frames))
    assert result['elapsed'] > 10
    assert all(dist(a['position'], b['position']) <= .250001 for a, b in zip(frames, frames[1:]))


def test_land_collision_timeout_cancellation_and_already_arrived():
    sea = SeaMap((0, 0, 10, 10), ((4, 4, 6, 6),))
    assert run_episode(sea, [(2, 5), (8, 5)], [], toward)['status'] == 'collision'
    assert run_episode(sea, [(2, 2), (8, 2)], [], toward, limit=.25)['status'] == 'timeout'
    assert run_episode(sea, [(2, 2), (8, 2)], [], toward, cancel=lambda: True)['status'] == 'cancelled'
    assert run_episode(sea, [(2, 2)], [], toward)['status'] == 'success'


def test_swept_crossing_and_stopped_ship_are_checked():
    ship = Traffic((5, 2), (5, 8), speed=12, radius=.1)
    assert swept_clearance((2, 5), (8, 5), ship, 0, .5, .1) < 0
    stopped = Traffic((5, 4), (5, 5), speed=10, radius=.1)
    assert swept_clearance((2, 5), (8, 5), stopped, 0, 1, .1) < 0


def stay(p, v, goal, neighbours, radius, speed, dt):
    return (0.0, 0.0)


def test_crossing_between_samples_is_caught_by_truth_and_filter():
    # Ego crosses (2,5)->(8,5) in a single dt=1s step at speed 6. The target
    # crosses the ego's path exactly at t=.5s, strictly between the t=0 and
    # t=1 samples, so an endpoint-only check would miss it entirely.
    sea = SeaMap((0, 0, 10, 10))
    ship = Traffic((5, 1), (5, 9), speed=8, radius=.6)
    bare = run_episode(sea, [(2, 5), (8, 5)], [ship], toward, dt=1, limit=1, radius=.5, speed=6)
    assert bare['status'] == 'collision'
    safe = run_episode(sea, [(2, 5), (8, 5)], [ship], toward, dt=1, limit=1, radius=.5, speed=6, filtered=True)
    assert safe['status'] != 'collision'
    assert any(d['override'] for d in safe['diagnostics'])
    assert not any(d['no_feasible_action'] for d in safe['diagnostics'])


def test_moving_target_hitting_stationary_ego_is_not_assumed_safe():
    # Ego holds still (a policy that never moves); a fast target sweeps
    # straight through the ego's stationary position between two samples.
    # A large open map keeps the safety filter's 3-second lookahead clear of
    # bounds so only the crossing threat itself is being tested.
    sea = SeaMap((0, 0, 1000, 1000))
    ship = Traffic((497, 500), (503, 500), speed=24, radius=.6)
    route = [(500, 500), (500, 500.0001)]
    bare = run_episode(sea, route, [ship], stay, dt=.25, limit=.25, radius=.5, speed=20)
    assert bare['status'] == 'collision'
    safe = run_episode(sea, route, [ship], stay, dt=.25, limit=.25, radius=.5, speed=20, filtered=True)
    assert safe['status'] != 'collision'
    assert any(d['override'] for d in safe['diagnostics'])


def test_filter_no_feasible_action_is_logged_not_fabricated_as_success():
    # A target with a radius far larger than the whole reachable neighbourhood
    # makes every candidate (including standing still) infeasible; the filter
    # must log that honestly instead of reporting success.
    sea = SeaMap((0, 0, 100, 100))
    ship = Traffic((66, 50), (50, 50), speed=8, radius=15)
    result = run_episode(sea, [(50, 50), (60, 50)], [ship], toward, dt=.25, limit=.25,
                         radius=.5, speed=1.0, filtered=True)
    assert result['status'] != 'success'
    assert any(d['no_feasible_action'] for d in result['diagnostics'])
