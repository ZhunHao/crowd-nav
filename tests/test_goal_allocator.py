"""Unit tests for GoalAllocator sampling primitives (R2 / WP-2)."""

from __future__ import annotations

import math

import pytest

from crowd_sim.envs.utils.seeding import seed_everything


@pytest.mark.unit
def test_sample_unused_position_returns_within_bounds() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    seed_everything(42)
    allocator = GoalAllocator()
    bounds = (-10.0, 10.0, -10.0, 10.0)  # (xmin, xmax, ymin, ymax)
    p = allocator.sample_unused_position(
        used_regions=[],
        bounds=bounds,
        min_dist=1.0,
    )
    x, y = p
    assert -10.0 <= x <= 10.0
    assert -10.0 <= y <= 10.0


@pytest.mark.unit
def test_sample_unused_position_respects_min_dist() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    seed_everything(42)
    allocator = GoalAllocator()
    used = [(0.0, 0.0)]
    p = allocator.sample_unused_position(
        used_regions=used,
        bounds=(-5.0, 5.0, -5.0, 5.0),
        min_dist=2.0,
    )
    assert math.hypot(p[0], p[1]) >= 2.0


@pytest.mark.unit
def test_sample_unused_position_is_deterministic_under_seed() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    seed_everything(42)
    a = GoalAllocator().sample_unused_position([], (-5, 5, -5, 5), 1.0)
    seed_everything(42)
    b = GoalAllocator().sample_unused_position([], (-5, 5, -5, 5), 1.0)
    assert a == b


@pytest.mark.unit
def test_sample_unused_position_raises_on_max_tries_exhausted() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    seed_everything(42)
    # Bounds entirely covered by a forbidden region — impossible to sample.
    allocator = GoalAllocator(max_tries=10)
    with pytest.raises(RuntimeError, match="could not sample"):
        allocator.sample_unused_position(
            used_regions=[(0.0, 0.0)],
            bounds=(-0.5, 0.5, -0.5, 0.5),
            min_dist=100.0,
        )


@pytest.mark.unit
def test_sample_unused_position_respects_is_free_callable() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    seed_everything(42)
    # is_free forbids the entire left half — returned x must be > 0.
    allocator = GoalAllocator(max_tries=200)
    p = allocator.sample_unused_position(
        used_regions=[],
        bounds=(-5.0, 5.0, -5.0, 5.0),
        min_dist=0.1,
        is_free=lambda x, y: x > 0.0,
    )
    assert p[0] > 0.0


@pytest.mark.unit
def test_allocate_waypoints_starts_at_start_ends_at_goal() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    seed_everything(42)
    allocator = GoalAllocator()
    wps = allocator.allocate_waypoints(
        start=(-11.0, -11.0),
        goal=(6.0, 11.0),
        num_waypoints=4,
        min_inter_dist=0.5,
    )
    assert len(wps) == 4
    # Last waypoint is always the global goal.
    assert wps[-1] == pytest.approx((6.0, 11.0))


@pytest.mark.unit
def test_allocate_waypoints_respects_min_inter_dist() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    seed_everything(42)
    wps = GoalAllocator().allocate_waypoints(
        start=(-11.0, -11.0),
        goal=(11.0, 11.0),
        num_waypoints=5,
        min_inter_dist=0.8,
    )
    # Pairwise distances between successive waypoints (not including start).
    for a, b in zip(wps[:-1], wps[1:]):
        assert math.hypot(a[0] - b[0], a[1] - b[1]) >= 0.8 - 1e-9


@pytest.mark.unit
def test_allocate_waypoints_uses_injected_source_when_provided() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    sentinel = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    seed_everything(42)
    wps = GoalAllocator().allocate_waypoints(
        start=(0.0, 0.0),
        goal=(3.0, 3.0),
        num_waypoints=3,
        min_inter_dist=0.0,
        waypoint_source=lambda s, g, n: sentinel,
    )
    assert wps == sentinel


@pytest.mark.unit
def test_allocate_waypoints_num_waypoints_must_be_positive() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    with pytest.raises(ValueError, match="num_waypoints"):
        GoalAllocator().allocate_waypoints(
            start=(0.0, 0.0), goal=(1.0, 1.0), num_waypoints=0, min_inter_dist=0.1
        )


@pytest.mark.unit
def test_allocate_human_positions_returns_n_pairs() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    seed_everything(42)
    pairs = GoalAllocator().allocate_human_positions(
        robot_start=(-11.0, -11.0),
        robot_goal=(6.0, 11.0),
        occupied=[(-11.0, -11.0)],
        human_num=5,
        min_dist=0.8,
    )
    assert len(pairs) == 5
    for start, goal in pairs:
        assert isinstance(start, tuple) and len(start) == 2
        assert isinstance(goal, tuple) and len(goal) == 2


@pytest.mark.unit
def test_allocate_human_positions_avoids_occupied() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    seed_everything(42)
    occupied = [(0.0, 0.0)]
    pairs = GoalAllocator().allocate_human_positions(
        robot_start=(-3.0, -3.0),
        robot_goal=(3.0, 3.0),
        occupied=occupied,
        human_num=3,
        min_dist=1.2,
    )
    for start, goal in pairs:
        assert math.hypot(start[0], start[1]) >= 1.2
        assert math.hypot(goal[0], goal[1]) >= 1.2


@pytest.mark.unit
def test_allocate_human_positions_is_deterministic() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    def run() -> list[tuple[tuple[float, float], tuple[float, float]]]:
        seed_everything(42)
        return GoalAllocator().allocate_human_positions(
            robot_start=(-3.0, -3.0),
            robot_goal=(3.0, 3.0),
            occupied=[],
            human_num=4,
            min_dist=0.5,
        )

    assert run() == run()
