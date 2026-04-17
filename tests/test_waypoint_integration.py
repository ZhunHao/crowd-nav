"""End-to-end: run_baseline.sh under the new allocator produces a video."""

from __future__ import annotations

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

    # The allocator log must appear — proves the new path executed, not the old
    # hardcoded loop (which had no such line).
    assert "Allocated" in result.stderr or "Allocated" in result.stdout, (
        "expected 'Allocated N waypoints' log line from test.py's GoalAllocator path"
    )
