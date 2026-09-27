import pytest
from math import dist
from shipnav.maps import SeaMap
from shipnav.theta import theta_star
from shipnav.planning import NoPath

def test_theta_segments_and_endpoints():
    sea = SeaMap((0, 0, 24, 24), ((9, 5, 14, 18),))
    route = theta_star(sea, (2, 2), (22, 22))
    assert route[0] == (2, 2) and route[-1] == (22, 22)
    assert all(sea.clear(a, b, .7) for a, b in zip(route, route[1:]))
    assert sum(dist(a, b) for a, b in zip(route, route[1:])) >= dist(route[0], route[-1])

def test_theta_rejects_disconnected_water():
    with pytest.raises(NoPath):
        theta_star(SeaMap((0, 0, 10, 10), ((4, 0, 6, 10),)), (2, 5), (8, 5))
