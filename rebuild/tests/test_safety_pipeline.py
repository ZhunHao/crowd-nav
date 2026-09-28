import pytest
from shipnav.maps import SeaMap
from shipnav.scenarios import make_scenario, scenario_hash
from shipnav.observations import Observer
from shipnav.simulation import Traffic, run_episode
from shipnav.policies import Direct
from shipnav.safety import choose, rollout


def test_scenario_ignores_planner_and_hash_detects_changes():
    sea = SeaMap((0, 0, 24, 24))
    a = make_scenario(sea, (2, 2), (22, 22), 4, 9)
    b = make_scenario(sea, (2, 2), (22, 22), 4, 9)
    assert scenario_hash(a) == scenario_hash(b)
    b['seed'] = 10
    assert scenario_hash(a) != scenario_hash(b)


def test_noise_is_keyed_and_does_not_mutate_truth():
    traffic = [Traffic((4, 4), (9, 4))]
    truth = traffic[0].at(1.)
    a = Observer(7, noise=.2).observe(traffic, 1., 4)
    b = Observer(7, noise=.2).observe(traffic, 1., 4)
    assert a == b and tuple(a[0]['position']) != truth[0]
    assert traffic[0].at(1.) == truth


def test_filter_prevents_land_command_and_reports_no_solution():
    sea = SeaMap((0, 0, 10, 10), ((4, 0, 6, 10),))
    r = choose(sea, (3, 5), (1, 0), [], .5, 1, .25)
    assert r['override'] and not r['no_feasible_action']
    assert all(sea.clear(a, b, .5) for a, b in zip(r['path'], r['path'][1:]))
    impossible = [{'radius': 10., 'points': [(3, 5)] * 13, 'margins': [0.] * 13}]
    r = choose(sea, (3, 5), (1, 0), impossible, .5, 1, .25)
    assert r['no_feasible_action']


def test_filter_is_in_the_episode_loop():
    sea = SeaMap((0, 0, 10, 10), ((4, 0, 6, 10),))
    bare = run_episode(sea, [(2, 5), (8, 5)], [], Direct(), limit=4)
    safe = run_episode(sea, [(2, 5), (8, 5)], [], Direct(), limit=4, filtered=True)
    assert bare['status'] == 'collision'
    assert safe['status'] == 'timeout'
    assert any(d['override'] for d in safe['diagnostics'])
    assert not any(d['land_collision'] for d in safe['diagnostics'])


def test_rollout_holds_at_final_goal_after_arrival():
    path = rollout((0., 0.), (1., 0.), .25, 12, goal=(1., 0.), arrival=.5)
    first = next(k for k, p in enumerate(path) if abs(p[0]-1.) < .5)
    assert all(p == path[first] for p in path[first:])
    assert rollout((0., 0.), (1., 0.), .25, 12)[-1] == (3., 0.)


@pytest.mark.parametrize('dynamics', ['holonomic', 'marine'])
def test_conflict_free_filtered_direct_to_shore_goal_logs_no_overrides(dynamics):
    # Goal 1.1 m off land (x>=30): any rollout that keeps going past the goal
    # would hit land, but the episode ends on arrival, so the nominal is safe.
    sea = SeaMap((0, 0, 40, 24), ((30, 0, 40, 24),))
    result = run_episode(sea, [(20, 12), (28.9, 12)], [], Direct(), filtered=True, dynamics=dynamics)
    assert result['status'] == 'success'
    assert not any(d['override'] for d in result['diagnostics'])


def test_intermediate_waypoint_near_shore_is_not_goal_aware():
    # Only the FINAL waypoint holds; an intermediate waypoint next to land still
    # sees the full straight rollout and the filter intervenes.
    sea = SeaMap((0, 0, 40, 24), ((30, 0, 40, 24),))
    decision = choose(sea, (27.5, 12), (1., 0.), [], .5, 1., .25, goal=(28.9, 12), final=False)
    assert decision['override']
    decision = choose(sea, (27.5, 12), (1., 0.), [], .5, 1., .25, goal=(28.9, 12), final=True)
    assert not decision['override']


def test_rollout_hold_starts_only_after_a_step_like_episode_arrival():
    # Arrival is checked after each step; a candidate that leaves the arrival
    # disk on its first step must not be treated as already arrived.
    path = rollout((0., 0.), (4., 0.), .25, 4, goal=(0., .0001), arrival=.5)
    assert path[-1] == (4., 0.)
