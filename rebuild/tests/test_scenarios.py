import json

import pytest

from shipnav.maps import SeaMap
from shipnav.scenarios import (load_traffic, make_scenario, traffic_to_dict, validate_observation,
                               validate_scenario)
from shipnav.simulation import CourseChangeTraffic, Traffic


def test_plain_traffic_entries_still_load_as_traffic():
    ship = Traffic((1., 2.), (3., 4.), speed=.7, radius=.8)
    data = {'traffic': [traffic_to_dict(ship)]}
    assert 'kind' not in data['traffic'][0]
    loaded = load_traffic(data)
    assert loaded == [ship]


def test_course_change_traffic_round_trips_through_scenario_json():
    ship = CourseChangeTraffic(((0., (1., 2.)), (5., (3., 2.)), (9., (3., 6.))), radius=.4)
    entry = traffic_to_dict(ship)
    assert entry['kind'] == 'course_change'
    data = {'traffic': [entry]}
    [loaded] = load_traffic(data)
    assert isinstance(loaded, CourseChangeTraffic)
    assert loaded.waypoints == ship.waypoints
    assert loaded.radius == ship.radius
    assert loaded.at(2.5) == ship.at(2.5)


def test_mixed_traffic_list_round_trips_each_by_kind():
    fixed = Traffic((0., 0.), (10., 0.))
    scripted = CourseChangeTraffic(((0., (0., 5.)), (4., (10., 5.))))
    data = {'traffic': [traffic_to_dict(fixed), traffic_to_dict(scripted)]}
    loaded = load_traffic(data)
    assert loaded == [fixed, scripted]


def _valid():
    return json.loads(json.dumps(make_scenario(SeaMap((0, 0, 24, 24)), (2, 2), (22, 22), 2, 5)))


def test_generated_and_round_tripped_scenarios_validate():
    validate_scenario(_valid())
    validate_scenario(make_scenario(SeaMap((0, 0, 24, 24)), (2, 2), (22, 22), 2, 5))


@pytest.mark.parametrize('mutate, message', [
    (lambda s: s.pop('traffic'), 'traffic'),
    (lambda s: s.pop('map'), 'map'),
    (lambda s: s['map'].pop('features'), 'features'),
    (lambda s: s.update(start=[1]), 'start'),
    (lambda s: s.update(goal=['a', 2]), 'goal'),
    (lambda s: s.update(seed='0'), 'seed'),
    (lambda s: s.update(seed=True), 'seed'),
    (lambda s: s.update(traffic_mode=3), 'traffic_mode'),
    (lambda s: s.update(traffic={}), 'traffic'),
    (lambda s: s['traffic'][0].pop('speed'), 'speed'),
    (lambda s: s['traffic'][0].update(colour='red'), 'colour'),
    (lambda s: s['traffic'][0].update(radius='big'), 'radius'),
    (lambda s: s['traffic'].append({'kind': 'course_change', 'radius': .6}), 'waypoints'),
    (lambda s: s['traffic'].append({'kind': 'teleport'}), 'kind'),
    (lambda s: s['traffic'].append([1, 2]), 'traffic'),
])
def test_malformed_scenarios_raise_clear_value_errors(mutate, message):
    scenario = _valid()
    mutate(scenario)
    with pytest.raises(ValueError, match=message):
        validate_scenario(scenario)


def test_non_dict_scenario_is_rejected():
    with pytest.raises(ValueError, match='object'):
        validate_scenario([])


@pytest.mark.parametrize('observation, message', [
    ({'seed': 1}, 'seed'),
    ({'jitter': .1}, 'jitter'),
    ({'noise': 'x'}, 'noise'),
    ({'dropout': True}, 'dropout'),
    ([], 'object'),
])
def test_observation_settings_reject_unknown_keys_and_bad_types(observation, message):
    with pytest.raises(ValueError, match=message):
        validate_observation(observation)


def test_valid_observation_settings_pass():
    validate_observation({})
    validate_observation({'noise': .1, 'delay': .25, 'dropout': .2, 'stale_speed_bound': 2})
