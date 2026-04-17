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
