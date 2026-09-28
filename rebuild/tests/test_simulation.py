from math import dist
from shipnav.maps import SeaMap
from shipnav.simulation import CourseChangeTraffic, Traffic, run_episode, swept_clearance


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


def test_swept_clearance_splits_at_every_course_change_breakpoint():
    # Ego travels (0,5)->(10,5) at constant speed 5 over one dt=2 step,
    # reaching the midpoint (5,5) at exactly t=1. The target turns through
    # that exact point at that exact time: (5,0)->(5,5) over [0,1], then
    # (5,5)->(0,5) over [1,2]. At both step endpoints (t=0 and t=2) target
    # and ego are far apart; the collision only exists strictly inside the
    # step, at the turn breakpoint t=1. Splitting only at the step endpoints
    # (the pre-generalization behaviour, keyed off `ship.arrival` alone)
    # linearly interpolates the *relative* position across the whole [0,2]
    # window and misses it; splitting at every breakpoint the target
    # reports (via `ship.breakpoints`) must catch it.
    target = CourseChangeTraffic(((0., (5., 0.)), (1., (5., 5.)), (2., (0., 5.))))
    assert swept_clearance((0, 5), (10, 5), target, 0, 2, .1) < 0


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


def test_target_land_and_target_target_invalidity_are_scored_separately_from_ego():
    # A scripted target whose bent, mid-path waypoint dips into land -- while
    # its straight start->goal chord (the only thing the episode-start
    # validation checks) stays clear -- must be flagged as
    # `target_land_invalid` mid-episode without ever being treated as an ego
    # collision: the ego is elsewhere, on an unrelated route.
    sea = SeaMap((0, 0, 20, 20), ((8, 8, 12, 12),))
    target = CourseChangeTraffic(((0., (2., 14.)), (1., (10., 10.)), (2., (18., 14.))))
    result = run_episode(sea, [(2, 2), (18, 2)], [target], toward, dt=1, limit=3, radius=.5, speed=4)
    assert result['status'] != 'collision'
    assert any(d['target_land_invalid'] for d in result['diagnostics'])
    assert not any(d['land_collision'] or d['ship_collision'] for d in result['diagnostics'])

    # Two targets whose paths cross each other -- not the ego -- must be
    # flagged via `target_target_min_clearance` going negative, again
    # without affecting the ego's own collision status.
    sea2 = SeaMap((0, 0, 20, 20))
    a = Traffic((10, 15), (10, 5), speed=10, radius=.6)  # crosses (10,10) at t=.5
    b = Traffic((5, 10), (15, 10), speed=10, radius=.6)  # crosses (10,10) at t=.5
    result2 = run_episode(sea2, [(2, 2), (18, 2)], [a, b], toward, dt=.5, limit=1.5, radius=.5, speed=4)
    assert result2['status'] != 'collision'
    assert any(d['target_target_min_clearance'] is not None and d['target_target_min_clearance'] < 0
               for d in result2['diagnostics'])


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


def test_unfiltered_diagnostics_mark_feasibility_as_not_evaluated():
    result = run_episode(SeaMap((0, 0, 20, 20)), [(2, 2), (8, 2)], [], toward, limit=1)
    assert all(d['no_feasible_action'] is None for d in result['diagnostics'])
    filtered = run_episode(SeaMap((0, 0, 20, 20)), [(2, 2), (8, 2)], [], toward, limit=1, filtered=True)
    assert all(d['no_feasible_action'] is False for d in filtered['diagnostics'])
