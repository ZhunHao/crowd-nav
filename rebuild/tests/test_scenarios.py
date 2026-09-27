from shipnav.scenarios import load_traffic, traffic_to_dict
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
