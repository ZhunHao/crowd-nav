"""Unit tests for export_writer (WP-5)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest


def _fake_states() -> list[dict]:
    # 3 steps, 2 humans.
    return [
        {
            "t": 0.0,
            "robot": (0.0, 0.0, 0.0, 0.0),  # px, py, vx, vy
            "humans": [(1.0, 0.0, 0.0, 0.0), (-1.0, 0.0, 0.0, 0.0)],
            "waypoint_idx": 0,
            "reward": 0.0,
        },
        {
            "t": 0.25,
            "robot": (0.1, 0.0, 0.4, 0.0),
            "humans": [(0.9, 0.0, -0.4, 0.0), (-0.9, 0.0, 0.4, 0.0)],
            "waypoint_idx": 0,
            "reward": 0.0,
        },
        {
            "t": 0.50,
            "robot": (0.2, 0.0, 0.4, 0.0),
            "humans": [(0.7, 0.0, -0.4, 0.0), (-0.7, 0.0, 0.4, 0.0)],
            "waypoint_idx": 1,
            "reward": 1.0,
        },
    ]


@pytest.mark.unit
def test_export_writes_csv_with_expected_columns(tmp_path: Path) -> None:
    from crowd_sim.envs.utils.export_writer import write_exports

    out = write_exports(tmp_path, states=_fake_states(), waypoints=[(0.5, 0.0), (1.0, 0.0)])

    assert out["csv"].exists()
    with out["csv"].open() as f:
        reader = csv.reader(f)
        header = next(reader)
    assert header[:4] == ["t", "robot_x", "robot_y", "robot_vx"]
    assert "waypoint_idx" in header
    assert "reward" in header


@pytest.mark.unit
def test_export_writes_trajectory_png(tmp_path: Path) -> None:
    from crowd_sim.envs.utils.export_writer import write_exports

    out = write_exports(tmp_path, states=_fake_states(), waypoints=[(0.5, 0.0), (1.0, 0.0)])
    assert out["png"].exists()
    # PNG magic header
    assert out["png"].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.unit
def test_export_writes_mp4_when_ffmpeg_available(tmp_path: Path) -> None:
    import shutil

    from crowd_sim.envs.utils.export_writer import write_exports

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    out = write_exports(tmp_path, states=_fake_states(), waypoints=[(0.5, 0.0), (1.0, 0.0)])
    assert out["mp4"].exists()
    assert out["mp4"].stat().st_size > 1000
