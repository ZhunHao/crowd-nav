"""R1 acceptance: scripts/run_baseline.sh produces exports/baseline.mp4."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


pytestmark = [pytest.mark.smoke, pytest.mark.slow]


@pytest.fixture
def baseline_mp4(repo_root: Path, exports_dir: Path) -> Path:
    out = exports_dir / "baseline.mp4"
    if out.exists():
        out.unlink()
    return out


def test_run_baseline_produces_video(repo_root: Path, baseline_mp4: Path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH — install via apt/brew to run smoke test")

    model = repo_root / "crowd_nav" / "data" / "output_trained" / "rl_model.pth"
    assert model.exists(), f"trained model missing at {model} — cannot reproduce R1"

    result = subprocess.run(
        ["bash", "scripts/run_baseline.sh"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=600,  # 10 min guard — plenty for a CPU rollout of one test case
    )
    assert result.returncode == 0, (
        f"run_baseline.sh exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )

    assert baseline_mp4.exists(), f"expected {baseline_mp4} to be created"
    size = baseline_mp4.stat().st_size
    assert size > 10_000, f"{baseline_mp4} suspiciously small ({size} bytes) — probably a failed write"


def test_baseline_is_deterministic_across_runs(repo_root: Path, baseline_mp4: Path):
    """Re-running with the same seed must produce identical deterministic stdout.

    ffmpeg writes metadata timestamps, so we compare logged start/goal/duration
    lines instead of raw bytes.
    """
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")

    def run_and_hash() -> str:
        proc = subprocess.run(
            ["bash", "scripts/run_baseline.sh"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=600,
            check=True,
        )
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(("start:", "It takes"))]
        return "\n".join(lines)

    h1 = run_and_hash()
    h2 = run_and_hash()
    assert h1 == h2, f"rollout not deterministic under --seed 42:\nrun1:\n{h1}\nrun2:\n{h2}"
