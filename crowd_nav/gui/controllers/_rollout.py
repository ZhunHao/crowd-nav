"""Rollout extraction (WP-5 / Task 3b). Stub until that task lands."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

from crowd_sim.envs.utils.static_map import StaticMap

if TYPE_CHECKING:
    from crowd_sim.envs.policy.policy import Policy


def run_waypoint_rollout(
    *,
    env_config_path: Path,
    static_map: StaticMap,
    policy: "Policy",
    start: tuple[float, float],
    goal: tuple[float, float],
    user_obstacles: list[dict],
    frame_callback: Optional[Callable[[int, dict], None]] = None,
) -> list[dict]:
    raise NotImplementedError("run_waypoint_rollout: see Task 3b")
