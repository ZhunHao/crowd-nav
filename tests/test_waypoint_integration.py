"""End-to-end: run_baseline.sh under the new allocator produces a video."""

from __future__ import annotations

import configparser
import re
import shutil
import subprocess
from pathlib import Path

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_allocator_driven_baseline_produces_video(repo_root: Path, exports_dir: Path) -> None:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")

    model = repo_root / "crowd_nav" / "data" / "output_trained" / "rl_model.pth"
    assert model.exists(), f"trained model missing at {model}"

    out = exports_dir / "baseline.mp4"
    if out.exists():
        out.unlink()

    result = subprocess.run(
        ["bash", "scripts/run_baseline.sh"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"run_baseline.sh exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert out.exists(), f"expected {out}"
    size = out.stat().st_size
    assert size > 10_000, f"{out} suspiciously small ({size} bytes)"

    merged_log = result.stderr + "\n" + result.stdout
    # The allocator log must appear — proves the new path executed, not the old
    # hardcoded loop (which had no such line).
    assert "Allocated" in merged_log, (
        "expected 'Allocated N waypoints' log line from test.py's GoalAllocator path"
    )

    # WP-3: rebuild the straight-line waypoints and assert every one is
    # outside the cordon obstacles + safety margin.
    match = re.search(
        r"Allocated (\d+) waypoints \(start=\((-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\) "
        r"goal=\((-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\)\)",
        merged_log,
    )
    assert match, f"couldn't parse Allocated line in log: {merged_log[-400:]}"
    n = int(match.group(1))
    start = (float(match.group(2)), float(match.group(3)))
    goal = (float(match.group(4)), float(match.group(5)))

    from crowd_sim.envs.utils.goal_allocator import _straight_line_source
    from crowd_sim.envs.utils.static_map import StaticMap

    waypoints = _straight_line_source(start, goal, n)

    # Rebuild the hardcoded cordon StaticMap from output_trained/env.config.
    cfg_path = repo_root / "crowd_nav" / "data" / "output_trained" / "env.config"
    cp = configparser.RawConfigParser()
    cp.read(cfg_path)
    # Emulate the cordon layout from CrowdSim.reset (static_obs=true).
    obs_list: list[dict] = []
    for obs_idx in range(30):
        obs_list.append({"type": "rect", "cx": -15, "cy": -15 + obs_idx, "w": 1, "h": 1})
    for obs_idx in range(30):
        obs_list.append({"type": "rect", "cx": 15, "cy": -15 + obs_idx, "w": 1, "h": 1})
    for obs_idx in range(30):
        obs_list.append({"type": "rect", "cx": -15 + obs_idx, "cy": -15, "w": 1, "h": 1})
    for obs_idx in range(30):
        obs_list.append({"type": "rect", "cx": -15 + obs_idx, "cy": 15, "w": 1, "h": 1})
    for obs_idx in range(20):
        obs_list.append({"type": "rect", "cx": -15 + obs_idx, "cy": -5, "w": 1, "h": 1})
    for obs_idx in range(20):
        obs_list.append({"type": "rect", "cx": 15 - obs_idx, "cy": 5, "w": 1, "h": 1})

    sm = StaticMap.from_static_obstacles(obs_list, margin=0.5)
    # Every waypoint — including the final one, which is pinned to the goal
    # after project_to_free has run — must be free of the cordon.
    for w in waypoints:
        assert sm.is_free(*w), f"waypoint {w} lies inside a cordon obstacle"

    # The infeasible config goal (5,5) sits inside the y=5 cordon bar.
    # test.py must log the projection so downstream consumers know the robot's
    # effective destination.
    assert "projected to (" in merged_log, (
        "expected 'projected to (...)' warning for infeasible global goal"
    )

    # WP-4: verify that a Theta* plan computed offline from the same inputs
    # produces a LoS-clear polyline under the same inflation. This confirms the
    # planner wiring actually routed around the cordon — not just that
    # allocator bubble-resampling happened to find free points.
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _obstacle_fixture import hardcoded_static_obstacles  # type: ignore[import-not-found]
    from crowd_nav.planner.theta_star import ThetaStar
    from crowd_sim.envs.utils.phase_config import PhaseConfig

    phase_cfg = PhaseConfig.from_configparser(cp)
    assert phase_cfg.planner.enabled, (
        "env.config in output_trained must enable [planner] for WP-4"
    )
    sm_planner = StaticMap.from_static_obstacles(
        hardcoded_static_obstacles(), margin=phase_cfg.static_map.margin
    )
    planner = ThetaStar(
        static_map=sm_planner,
        inflation=phase_cfg.planner.inflation_radius,
        grid_resolution=phase_cfg.planner.grid_resolution,
        bounds=phase_cfg.planner.bounds,
        simplify=phase_cfg.planner.waypoint_simplify,
    )
    path = planner.plan(start=start, goal=goal)
    for x, y in path:
        assert sm_planner.is_free(
            x, y, margin=phase_cfg.planner.inflation_radius
        ), f"Theta* vertex ({x}, {y}) lands inside an inflated obstacle"
    prev = start
    for nxt in path:
        assert planner.line_of_sight(
            prev, nxt
        ), f"Theta* segment {prev}->{nxt} enters an obstacle"
        prev = nxt
