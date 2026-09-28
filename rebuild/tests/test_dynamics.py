from math import hypot

from shipnav.dynamics import Vessel, advance, motion_from
from shipnav.maps import SeaMap
from shipnav.simulation import run_episode
from shipnav.policies import Direct


def test_heading_and_acceleration_cannot_jump():
    a = Vessel(2, 2, 0, 1)
    b = advance(a, (-1, 0), .25)
    assert abs(b.heading-a.heading) <= .35*.25+1e-12
    assert abs(b.speed-a.speed) <= .2*.25+1e-12
    assert b.x > a.x  # inertia prevents instantaneous reversal
    assert motion_from(a)((-1, 0), .25, 1)[1] == (b.x, b.y)


def test_marine_episode_uses_reachable_motion():
    r = run_episode(SeaMap((0, 0, 24, 24)), [(2, 2), (20, 2)], [], Direct(), dynamics='marine', limit=2)
    assert r['frames'][1]['position'][0]-2 < .25
    assert all(d['speed'] <= 1 for d in r['diagnostics'])


def test_near_shore_braking_filtered_avoids_land_unfiltered_collides():
    # Land wall spans x in [10,12]; vessel starts well clear of it heading straight at it
    # at full speed, close enough that stopping distance under bounded deceleration matters.
    sea = SeaMap((0, 0, 20, 20), ((10, 0, 12, 20),))
    route = [(6, 10), (16, 10)]

    unfiltered = run_episode(sea, route, [], Direct(), dynamics='marine', limit=6, filtered=False)
    assert unfiltered['status'] == 'collision'
    assert any(d['land_collision'] for d in unfiltered['diagnostics'])

    filtered = run_episode(sea, route, [], Direct(), dynamics='marine', limit=6, filtered=True)
    # This fixture is deterministic: with this much stopping distance, the filter always
    # finds a feasible avoiding command (never reports no_feasible_action) and never lets
    # the vessel touch land. Assert that actual outcome directly rather than a permissive
    # either/or — a masked collision would fail these, not slip through.
    assert not any(d['no_feasible_action'] for d in filtered['diagnostics'])
    assert not any(d['land_collision'] for d in filtered['diagnostics'])
    assert filtered['status'] != 'collision'


def test_filter_candidate_rollout_uses_motion_from_not_instant_stop():
    # The (0,0) "stop" candidate must still decelerate through motion_from's reachable
    # dynamics, not teleport to a standstill in one step.
    sea = SeaMap((0, 0, 20, 20), ((10, 0, 12, 20),))
    vessel = Vessel(8, 10, 0., 1.)
    motion = motion_from(vessel, 1.)
    path = motion((0., 0.), .25, 12)
    # Under bounded deceleration (.2 m/s^2), it takes speed/accel = 5s to stop from speed 1;
    # over the first step the vessel is still moving forward, not frozen at the start point.
    assert path[0] == (vessel.x, vessel.y)
    assert path[1][0] > vessel.x
    # And it should NOT have reached full stop (x offset of 0) by the very first sample.
    stopped_immediately = all(p == path[0] for p in path[1:2])
    assert not stopped_immediately


def test_high_speed_turning_bounded_heading_change_filtered_and_unfiltered():
    sea = SeaMap((0, 0, 24, 24))
    route = [(2, 2), (2, 20), (20, 20)]  # sharp waypoint turn at full speed
    dt = .25
    yaw_rate = .35
    for filtered in (False, True):
        r = run_episode(sea, route, [], Direct(), dynamics='marine', limit=10, filtered=filtered, dt=dt)
        headings = [d['heading'] for d in r['diagnostics']]
        for h0, h1 in zip(headings, headings[1:]):
            delta = (h1-h0+3.141592653589793) % (2*3.141592653589793) - 3.141592653589793
            assert abs(delta) <= yaw_rate*dt + 1e-9


def test_timestep_refinement_unfiltered_integration_agrees_closely():
    # Isolates the dynamics.advance integration scheme itself (no safety filter in the
    # loop): halving dt should change the collision outcome and the contact position by
    # only the discretisation error of one forward-Euler step, not by a material amount.
    sea = SeaMap((0, 0, 20, 20), ((10, 0, 12, 20),))
    route = [(6, 10), (16, 10)]

    coarse = run_episode(sea, route, [], Direct(), dynamics='marine', limit=6, filtered=False, dt=.25)
    fine = run_episode(sea, route, [], Direct(), dynamics='marine', limit=6, filtered=False, dt=.125)

    assert coarse['status'] == fine['status'] == 'collision'
    last_coarse = coarse['frames'][-1]['position']
    last_fine = fine['frames'][-1]['position']
    # Tolerance: one coarse step's worst-case chord error at max speed (1 m/s * .25 s =
    # .25 m); observed divergence is ~0.06 m, well inside this bound.
    assert hypot(last_coarse[0]-last_fine[0], last_coarse[1]-last_fine[1]) <= .25


def test_timestep_refinement_filtered_status_agrees_but_horizon_shifts_trajectory():
    # With the safety filter engaged, `choose()` rolls candidates out the shared
    # marine horizon (`shipnav.horizon.horizon_steps`), which is defined in seconds
    # of stopping time (ceil(max_speed/acceleration/dt)+1 steps): 5.25 s at dt=.25
    # versus 5.125 s at dt=.125. Refining the timestep therefore now refines
    # integration accuracy without also halving the filter's look-ahead (the
    # previous fixed steps=12 coupling gave ~1.04 m divergence here).
    sea = SeaMap((0, 0, 20, 20), ((10, 0, 12, 20),))
    route = [(6, 10), (16, 10)]

    coarse = run_episode(sea, route, [], Direct(), dynamics='marine', limit=6, filtered=True, dt=.25)
    fine = run_episode(sea, route, [], Direct(), dynamics='marine', limit=6, filtered=True, dt=.125)

    assert coarse['status'] == fine['status']
    assert not any(d['land_collision'] for d in coarse['diagnostics'])
    assert not any(d['land_collision'] for d in fine['diagnostics'])
    last_coarse = coarse['frames'][-1]['position']
    last_fine = fine['frames'][-1]['position']
    divergence = hypot(last_coarse[0]-last_fine[0], last_coarse[1]-last_fine[1])
    # Observed divergence is ~0.013 m; bound by one coarse step's worst-case chord
    # error at max speed (.25 m), the same tolerance as the unfiltered comparison.
    assert divergence <= .25


def test_marine_vessel_starts_on_the_first_route_leg_bearing():
    # Due north first leg: the initial heading must be pi/2, so the first
    # step already moves north instead of spending steps turning from east.
    r = run_episode(SeaMap((0, 0, 24, 24)), [(12, 2), (12, 20)], [], Direct(), dynamics='marine', limit=.25)
    assert abs(r['diagnostics'][0]['heading'] - 1.5707963267948966) < 1e-12
    assert abs(r['frames'][1]['position'][0] - 12) < 1e-12
    assert r['frames'][1]['position'][1] > 2
