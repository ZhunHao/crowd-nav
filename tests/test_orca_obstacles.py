"""Tier C regression: ORCA humans steer around static obstacle polygons."""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_orca_trajectory_avoids_rect_obstacle() -> None:
    """Human at (-5,0) heading (5,0) must not cross the AABB of a rect at the origin."""
    from crowd_sim.envs.policy.orca import ORCA
    from crowd_sim.envs.utils.action import ActionXY
    from crowd_sim.envs.utils.state import FullState, ObservableState, JointState

    policy = ORCA()
    policy.time_step = 0.25
    # Obstacle rect cx=0, cy=0, w=2, h=4 => AABB [-1,1] x [-2,2].
    rect_ccw = [(-1.0, -2.0), (1.0, -2.0), (1.0, 2.0), (-1.0, 2.0)]
    policy.set_static_obstacles([rect_ccw])

    # Simulated human state.
    px, py = -5.0, 0.0
    vx, vy = 0.0, 0.0
    radius = 0.3
    v_pref = 1.0
    gx, gy = 5.0, 0.0
    # No other humans.
    ghost = ObservableState(100.0, 100.0, 0.0, 0.0, 0.01)

    positions = [(px, py)]
    for _ in range(60):
        self_state = FullState(px, py, vx, vy, radius, gx, gy, v_pref, 0.0)
        state = JointState(self_state, [ghost])
        action = policy.predict(state)
        # Integrate Euler step.
        vx, vy = action.vx, action.vy
        px += vx * policy.time_step
        py += vy * policy.time_step
        positions.append((px, py))
        if (px - gx) ** 2 + (py - gy) ** 2 < 0.1:
            break

    # Every intermediate position must remain outside AABB [-1,1] x [-2,2].
    for (x, y) in positions:
        inside = (-1.0 < x < 1.0) and (-2.0 < y < 2.0)
        assert not inside, f"ORCA trajectory entered obstacle at {(x, y)}"


@pytest.mark.unit
def test_orca_set_static_obstacles_resets_sim() -> None:
    """Calling set_static_obstacles mid-run must force a sim rebuild."""
    from crowd_sim.envs.policy.orca import ORCA

    policy = ORCA()
    policy.sim = object()  # sentinel to prove the setter invalidates it
    policy.set_static_obstacles([[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]])
    assert policy.sim is None
    assert policy.static_obstacle_polygons == [
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    ]
