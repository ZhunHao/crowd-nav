import pytest
from math import dist
from shipnav.maps import SeaMap
from shipnav.simulation import Traffic, run_episode, traffic_for_route, swept_clearance


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


def test_scenarios_are_seeded_and_not_recreated_at_waypoints():
    sea = SeaMap((0, 0, 24, 24))
    route = [(2, 2), (12, 12), (22, 22)]
    a = traffic_for_route(sea, route, 5, 7)
    b = traffic_for_route(sea, route, 5, 7)
    assert a == b and len(a) == 5
    result = run_episode(sea, route, a, toward)
    for frame in result['frames']:
        assert frame['traffic'] == [list(ship.at(frame['t'])[0]) for ship in a]
