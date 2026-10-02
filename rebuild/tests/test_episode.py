import pytest

from shipnav.episode import SYNTHETIC_LIMIT, build_traffic, episode_limit, plan_route
from shipnav.maps import SeaMap
from shipnav.reactive import ReactiveTraffic
from shipnav.scenarios import make_scenario


def test_plan_route_matches_both_planners_and_rejects_unknown():
    sea = SeaMap((0, 0, 24, 24), ((9, 5, 14, 18),))
    assert plan_route(sea, (2, 2), (22, 22))[0] == (2, 2)
    assert plan_route(sea, (2, 2), (22, 22), 'theta')[-1] == (22, 22)
    with pytest.raises(ValueError, match='Unknown planner'):
        plan_route(sea, (2, 2), (22, 22), 'dijkstra')


def test_build_traffic_wraps_reactive_and_rejects_unknown_modes():
    sea = SeaMap((0, 0, 24, 24))
    scenario = make_scenario(sea, (2, 2), (22, 22), 2, 4)
    assert len(build_traffic(scenario, sea)) == 2
    reactive = {**scenario, 'traffic_mode': 'reactive'}
    assert all(isinstance(s, ReactiveTraffic) for s in build_traffic(reactive, sea))
    with pytest.raises(ValueError, match='Unknown traffic model'):
        build_traffic({**scenario, 'traffic_mode': 'learned'}, sea)


def test_reactive_mode_rejects_course_change_entries():
    sea = SeaMap((0, 0, 24, 24))
    scenario = make_scenario(sea, (2, 12), (22, 12), 1, 5, corridor=[(2, 12), (22, 12)])
    with pytest.raises(ValueError, match='course_change'):
        build_traffic({**scenario, 'traffic_mode': 'reactive'}, sea)


def test_episode_limit_is_fixed_for_synthetic_and_route_scaled_for_real_maps():
    route = [(0, 0), (60, 0)]
    assert episode_limit({'metadata': {}}, route) == SYNTHETIC_LIMIT
    assert episode_limit({'metadata': {'model_scale': {}}}, route) == 180.
    assert episode_limit({'metadata': {'model_scale': {}}}, [(0, 0), (1, 0)]) == SYNTHETIC_LIMIT
