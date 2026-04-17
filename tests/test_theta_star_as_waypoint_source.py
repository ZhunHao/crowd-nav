"""WaypointSourceFn compatibility - ThetaStar + GoalAllocator (R4 / WP-4)."""

from __future__ import annotations

import pytest

from crowd_sim.envs.utils.goal_allocator import GoalAllocator
from crowd_sim.envs.utils.seeding import seed_everything
from crowd_sim.envs.utils.static_map import Obstacle, StaticMap


def _map_with_centre_rect() -> StaticMap:
    return StaticMap(
        obstacles=(Obstacle(kind="rect", cx=0.0, cy=0.0, w=4.0, h=4.0),),
        margin=0.5,
    )


@pytest.mark.unit
def test_theta_star_waypoint_source_returns_exactly_n_points() -> None:
    from crowd_nav.planner.theta_star import ThetaStar

    seed_everything(42)
    planner = ThetaStar(
        static_map=_map_with_centre_rect(),
        inflation=0.5,
        grid_resolution=0.25,
        bounds=(-8.0, 8.0, -8.0, 8.0),
    )
    source = planner.as_waypoint_source(n=10)
    wps = source((-5.0, 0.0), (5.0, 0.0), 10)
    assert len(wps) == 10
    assert wps[-1] == pytest.approx((5.0, 0.0), abs=1e-6)


@pytest.mark.unit
def test_theta_star_plugged_into_allocate_waypoints() -> None:
    from crowd_nav.planner.theta_star import ThetaStar

    seed_everything(42)
    static_map = _map_with_centre_rect()
    planner = ThetaStar(
        static_map=static_map,
        inflation=0.5,
        grid_resolution=0.25,
        bounds=(-8.0, 8.0, -8.0, 8.0),
    )
    allocator = GoalAllocator(max_tries=500)
    wps = allocator.allocate_waypoints(
        start=(-5.0, 0.0),
        goal=(5.0, 0.0),
        num_waypoints=10,
        min_inter_dist=0.5,
        is_free=static_map.is_free,
        waypoint_source=planner.as_waypoint_source(10),
    )
    assert len(wps) == 10
    assert wps[-1] == pytest.approx((5.0, 0.0), abs=1e-6)
    for x, y in wps:
        assert static_map.is_free(
            x, y, margin=0.5
        ), f"waypoint ({x}, {y}) inside obstacle"


@pytest.mark.unit
def test_theta_star_waypoint_source_rejects_mismatched_n() -> None:
    from crowd_nav.planner.theta_star import ThetaStar

    planner = ThetaStar(static_map=_map_with_centre_rect())
    source = planner.as_waypoint_source(n=10)
    with pytest.raises(ValueError, match="bound to n=10"):
        source((-5.0, 0.0), (5.0, 0.0), 11)
