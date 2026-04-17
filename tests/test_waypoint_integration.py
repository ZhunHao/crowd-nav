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
    # Assert on interpolated waypoints; the final waypoint is pinned to the
    # externally-specified global goal — ensuring that goal is feasible is the
    # env config's responsibility, and Tier-B collision handles the robot case.
    for w in waypoints[:-1]:
        assert sm.is_free(*w), f"waypoint {w} lies inside a cordon obstacle"
