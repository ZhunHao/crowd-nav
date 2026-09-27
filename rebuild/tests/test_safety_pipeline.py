from shipnav.maps import SeaMap
from shipnav.scenarios import make_scenario, scenario_hash
from shipnav.observations import Observer
from shipnav.simulation import Traffic, run_episode
from shipnav.policies import Direct
from shipnav.safety import choose


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
