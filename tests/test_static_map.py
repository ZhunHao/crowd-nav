"""Unit tests for StaticMap geometry primitives (R3 / WP-3)."""

from __future__ import annotations

import dataclasses
import math

import pytest


@pytest.mark.unit
def test_is_free_inside_rect_returns_false() -> None:
    from crowd_sim.envs.utils.static_map import Obstacle, StaticMap

    sm = StaticMap(
        obstacles=(Obstacle(kind="rect", cx=0.0, cy=0.0, w=2.0, h=4.0),),
        margin=0.0,
    )
    assert sm.is_free(5.0, 0.0) is True
    assert sm.is_free(0.0, 0.0) is False
    assert sm.is_free(1.0, 2.0) is False  # corner (boundary counted as occupied)


@pytest.mark.unit
def test_is_free_respects_margin() -> None:
    from crowd_sim.envs.utils.static_map import Obstacle, StaticMap

    obs = Obstacle(kind="rect", cx=0.0, cy=0.0, w=2.0, h=2.0)
    sm = StaticMap(obstacles=(obs,), margin=0.0)
    assert sm.is_free(1.6, 0.0) is True
    # With margin=1.0, the inflated rect reaches x=2.0; 1.6 is now occupied.
    assert sm.is_free(1.6, 0.0, margin=1.0) is False


@pytest.mark.unit
def test_is_free_uses_default_margin_when_none() -> None:
    from crowd_sim.envs.utils.static_map import Obstacle, StaticMap

    obs = Obstacle(kind="rect", cx=0.0, cy=0.0, w=2.0, h=2.0)
    sm = StaticMap(obstacles=(obs,), margin=0.5)
    # Default margin inflates half-width from 1.0 to 1.5.
    assert sm.is_free(1.4, 0.0) is False
    assert sm.is_free(1.6, 0.0) is True


@pytest.mark.unit
def test_is_free_circle_obstacle() -> None:
    from crowd_sim.envs.utils.static_map import Obstacle, StaticMap

    sm = StaticMap(
        obstacles=(Obstacle(kind="circle", cx=0.0, cy=0.0, w=0.0, h=0.0, r=1.0),),
        margin=0.0,
    )
    assert sm.is_free(0.0, 0.0) is False
    assert sm.is_free(0.5, 0.5) is False  # inside r=1
    assert sm.is_free(2.0, 0.0) is True
    assert sm.is_free(0.0, 0.5, margin=0.6) is False  # inflated reach 1.6


@pytest.mark.unit
def test_from_static_obstacles_roundtrip() -> None:
    from crowd_sim.envs.utils.static_map import StaticMap

    obs_list = [
        {"type": "rect", "cx": 1.0, "cy": 2.0, "w": 3.0, "h": 4.0},
        {"type": "circle", "cx": -1.0, "cy": -2.0, "r": 0.5},
    ]
    sm = StaticMap.from_static_obstacles(obs_list, margin=0.25)
    assert len(sm.obstacles) == 2
    assert sm.obstacles[0].kind == "rect"
    assert sm.obstacles[0].cx == 1.0
    assert sm.obstacles[0].w == 3.0
    assert sm.obstacles[1].kind == "circle"
    assert sm.obstacles[1].r == 0.5
    assert sm.margin == 0.25
    # Functional check — a point at the rect center is blocked.
    assert sm.is_free(1.0, 2.0) is False


@pytest.mark.unit
def test_from_static_obstacles_rejects_unknown_type() -> None:
    from crowd_sim.envs.utils.static_map import StaticMap

    with pytest.raises(ValueError, match="unknown obstacle type"):
        StaticMap.from_static_obstacles([{"type": "triangle", "cx": 0, "cy": 0}])


@pytest.mark.unit
def test_rect_vertices_ccw_order() -> None:
    from crowd_sim.envs.utils.static_map import Obstacle, rect_vertices_ccw

    obs = Obstacle(kind="rect", cx=0.0, cy=0.0, w=2.0, h=4.0)
    verts = rect_vertices_ccw(obs)
    assert verts == [(-1.0, -2.0), (1.0, -2.0), (1.0, 2.0), (-1.0, 2.0)]
    # Cross-product sanity: CCW has positive signed area.
    area = 0.0
    for i in range(len(verts)):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % len(verts)]
        area += x1 * y2 - x2 * y1
    assert area > 0


@pytest.mark.unit
def test_rect_vertices_ccw_rejects_non_rect() -> None:
    from crowd_sim.envs.utils.static_map import Obstacle, rect_vertices_ccw

    circle = Obstacle(kind="circle", cx=0.0, cy=0.0, w=0.0, h=0.0, r=1.0)
    with pytest.raises(ValueError):
        rect_vertices_ccw(circle)


@pytest.mark.unit
def test_static_map_is_frozen_dataclass() -> None:
    from crowd_sim.envs.utils.static_map import Obstacle, StaticMap

    sm = StaticMap(obstacles=(Obstacle(kind="rect", cx=0, cy=0, w=1, h=1),), margin=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        sm.margin = 1.0  # type: ignore[misc]


@pytest.mark.unit
def test_obstacle_is_frozen_dataclass() -> None:
    from crowd_sim.envs.utils.static_map import Obstacle

    o = Obstacle(kind="rect", cx=0, cy=0, w=1, h=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.cx = 5.0  # type: ignore[misc]


@pytest.mark.unit
def test_is_free_no_obstacles_everywhere_free() -> None:
    from crowd_sim.envs.utils.static_map import StaticMap

    sm = StaticMap(obstacles=(), margin=0.5)
    assert sm.is_free(0.0, 0.0) is True
    assert sm.is_free(100.0, -100.0, margin=10.0) is True


@pytest.mark.unit
def test_from_static_obstacles_empty_list() -> None:
    from crowd_sim.envs.utils.static_map import StaticMap

    sm = StaticMap.from_static_obstacles([], margin=0.3)
    assert sm.obstacles == ()
    assert sm.margin == 0.3
    assert math.isfinite(sm.margin)


@pytest.mark.unit
def test_project_to_free_returns_input_when_already_free() -> None:
    from crowd_sim.envs.utils.static_map import Obstacle, StaticMap

    sm = StaticMap(
        obstacles=(Obstacle(kind="rect", cx=0.0, cy=0.0, w=2.0, h=2.0),),
        margin=0.0,
    )
    # (5, 5) is far outside the rect — should pass through unchanged.
    assert sm.project_to_free(5.0, 5.0) == (5.0, 5.0)


@pytest.mark.unit
def test_project_to_free_snaps_out_of_rect() -> None:
    from crowd_sim.envs.utils.static_map import Obstacle, StaticMap

    sm = StaticMap(
        obstacles=(Obstacle(kind="rect", cx=0.0, cy=0.0, w=4.0, h=2.0),),
        margin=0.0,
    )
    # (0, 0) is at the rect center — projection should land outside.
    projected = sm.project_to_free(0.0, 0.0, margin=0.1)
    assert sm.is_free(*projected, margin=0.1)
    # The projection can't be farther than the max search radius.
    assert math.hypot(projected[0], projected[1]) <= 10.0


@pytest.mark.unit
def test_project_to_free_snaps_out_of_circle() -> None:
    from crowd_sim.envs.utils.static_map import Obstacle, StaticMap

    sm = StaticMap(
        obstacles=(Obstacle(kind="circle", cx=0.0, cy=0.0, w=0.0, h=0.0, r=1.5),),
        margin=0.0,
    )
    projected = sm.project_to_free(0.3, 0.2)
    assert sm.is_free(*projected)


@pytest.mark.unit
def test_project_to_free_respects_default_margin_when_none() -> None:
    from crowd_sim.envs.utils.static_map import Obstacle, StaticMap

    # Rect at origin, default margin 0.5 inflates it substantially.
    sm = StaticMap(
        obstacles=(Obstacle(kind="rect", cx=0.0, cy=0.0, w=2.0, h=2.0),),
        margin=0.5,
    )
    # (1.2, 0) is outside the bare rect (half-width 1.0) but inside the
    # margin-inflated one (half-width 1.5). Projection with margin=None
    # (default) must keep that margin.
    projected = sm.project_to_free(1.2, 0.0)
    assert sm.is_free(*projected)  # default margin
    # Confirm we actually moved — the input point was not free under the default margin.
    assert projected != (1.2, 0.0)


@pytest.mark.unit
def test_project_to_free_raises_when_max_radius_exhausted() -> None:
    from crowd_sim.envs.utils.static_map import Obstacle, StaticMap

    # A huge rect that covers the entire search area.
    sm = StaticMap(
        obstacles=(Obstacle(kind="rect", cx=0.0, cy=0.0, w=100.0, h=100.0),),
        margin=0.0,
    )
    with pytest.raises(RuntimeError, match="could not project"):
        sm.project_to_free(0.0, 0.0, max_radius=5.0)


@pytest.mark.unit
def test_project_to_free_distance_bounded_by_max_radius() -> None:
    from crowd_sim.envs.utils.static_map import Obstacle, StaticMap

    sm = StaticMap(
        obstacles=(Obstacle(kind="rect", cx=0.0, cy=0.0, w=3.0, h=3.0),),
        margin=0.0,
    )
    x0, y0 = 0.5, 0.5
    projected = sm.project_to_free(x0, y0, max_radius=5.0)
    assert math.hypot(projected[0] - x0, projected[1] - y0) <= 5.0 + 1e-6
