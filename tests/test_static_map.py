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
