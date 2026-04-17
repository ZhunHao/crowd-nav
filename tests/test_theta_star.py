"""Unit tests for ThetaStar (R4 / WP-4)."""

from __future__ import annotations

import math

import pytest

from crowd_sim.envs.utils.static_map import Obstacle, StaticMap


def _empty_map() -> StaticMap:
    return StaticMap(obstacles=(), margin=0.0)


def _rect_map() -> StaticMap:
    """Single 4x4 rectangle centred on origin - blocks any straight line from
    (-5, 0) to (5, 0) through y approx 0."""
    return StaticMap(
        obstacles=(Obstacle(kind="rect", cx=0.0, cy=0.0, w=4.0, h=4.0),),
        margin=0.0,
    )


@pytest.mark.unit
def test_line_of_sight_clear_on_empty_map() -> None:
    from crowd_nav.planner.theta_star import ThetaStar

    planner = ThetaStar(static_map=_empty_map(), inflation=0.0)
    assert planner.line_of_sight((-5.0, -5.0), (5.0, 5.0)) is True


@pytest.mark.unit
def test_line_of_sight_blocked_by_rect() -> None:
    from crowd_nav.planner.theta_star import ThetaStar

    planner = ThetaStar(static_map=_rect_map(), inflation=0.0)
    assert planner.line_of_sight((-5.0, 0.0), (5.0, 0.0)) is False


@pytest.mark.unit
def test_line_of_sight_respects_inflation() -> None:
    from crowd_nav.planner.theta_star import ThetaStar

    # Rect half-width 2; inflate by 1 m -> effective half-width 3.
    planner = ThetaStar(static_map=_rect_map(), inflation=1.0)
    assert planner.line_of_sight((-5.0, 2.5), (5.0, 2.5)) is False
    assert planner.line_of_sight((-5.0, 3.5), (5.0, 3.5)) is True


@pytest.mark.unit
def test_plan_returns_goal_when_los_clear() -> None:
    from crowd_nav.planner.theta_star import ThetaStar

    planner = ThetaStar(static_map=_empty_map(), inflation=0.0)
    path = planner.plan(start=(-3.0, -3.0), goal=(3.0, 3.0))
    assert path == [(3.0, 3.0)]


@pytest.mark.unit
def test_plan_routes_around_rect() -> None:
    from crowd_nav.planner.theta_star import ThetaStar

    planner = ThetaStar(
        static_map=_rect_map(),
        inflation=0.2,
        grid_resolution=0.25,
        bounds=(-8.0, 8.0, -8.0, 8.0),
    )
    path = planner.plan(start=(-5.0, 0.0), goal=(5.0, 0.0))
    assert len(path) >= 2
    assert path[-1] == pytest.approx((5.0, 0.0), abs=1e-6)
    for x, y in path:
        assert planner.static_map.is_free(x, y, margin=planner.inflation)


@pytest.mark.unit
def test_plan_no_segment_enters_obstacle() -> None:
    from crowd_nav.planner.theta_star import ThetaStar

    planner = ThetaStar(
        static_map=_rect_map(),
        inflation=0.2,
        grid_resolution=0.25,
        bounds=(-8.0, 8.0, -8.0, 8.0),
    )
    path = planner.plan(start=(-5.0, 0.0), goal=(5.0, 0.0))
    prev = (-5.0, 0.0)
    for nxt in path:
        assert planner.line_of_sight(prev, nxt)
        prev = nxt


@pytest.mark.unit
def test_plan_raises_no_path_found_when_goal_inside_obstacle() -> None:
    from crowd_nav.planner.theta_star import NoPathFound, ThetaStar

    planner = ThetaStar(
        static_map=_rect_map(),
        inflation=0.0,
        grid_resolution=0.25,
        bounds=(-8.0, 8.0, -8.0, 8.0),
    )
    with pytest.raises(NoPathFound):
        planner.plan(start=(-5.0, 0.0), goal=(0.0, 0.0))


@pytest.mark.unit
def test_plan_raises_value_error_when_start_equals_goal() -> None:
    from crowd_nav.planner.theta_star import ThetaStar

    planner = ThetaStar(static_map=_empty_map())
    with pytest.raises(ValueError, match="start == goal"):
        planner.plan(start=(1.0, 1.0), goal=(1.0, 1.0))


@pytest.mark.unit
def test_plan_is_deterministic() -> None:
    from crowd_nav.planner.theta_star import ThetaStar

    planner = ThetaStar(
        static_map=_rect_map(),
        inflation=0.2,
        grid_resolution=0.25,
        bounds=(-8.0, 8.0, -8.0, 8.0),
    )
    a = planner.plan(start=(-5.0, 0.0), goal=(5.0, 0.0))
    b = planner.plan(start=(-5.0, 0.0), goal=(5.0, 0.0))
    assert a == b


@pytest.mark.unit
def test_plan_rejects_invalid_bounds() -> None:
    from crowd_nav.planner.theta_star import ThetaStar

    with pytest.raises(ValueError, match="degenerate bounds"):
        ThetaStar(static_map=_empty_map(), bounds=(5.0, -5.0, -5.0, 5.0))
