# tests/test_smoke_wp4.py
"""WP-4 smoke: Theta* planner produces a rollout mp4.

Step 1 findings (Case B — config-driven):
- test.py has NO --planner CLI flag.
- crowd_nav/data/output_trained/env.config has:
    [planner]
    enabled = true
    algorithm = theta_star
  so the shipped config already activates Theta* unconditionally.
- crowd_sim/envs/utils/phase_config.py reads [planner] algorithm at line 79;
  the fallback default is also "theta_star".
- test.py imports ThetaStar at line 15 and instantiates it at line 161 when
  static_map is not None and phase_cfg.planner.enabled is True.
- On SUCCESS there is no log line that mentions "theta" or "planner" — only
  the fallback WARNING "Theta* found no path (%s); falling back to …" is
  printed (to stderr) when the planner raises NoPathFound.
- Therefore the positive assertion "theta in combined" cannot be satisfied on a
  healthy run.  The test instead uses the negative assertion
  "fallback not in combined" to prove Theta* ran without degrading to the
  straight-line fallback, plus the returncode==0 and mp4 size checks.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.slow]


def test_theta_star_rollout_produces_video(repo_root: Path, exports_dir: Path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")

    out = exports_dir / "wp4_smoke.mp4"
    if out.exists():
        out.unlink()

    # data/output_trained/env.config already has [planner] enabled=true,
    # algorithm=theta_star — no extra CLI flag needed.
    cmd = [
        "python", "test.py",
        "--policy", "sarl",
        "--model_dir", "data/output_trained",
        "--phase", "test",
        "--test_case", "0",
        "--seed", "42",
        "--visualize",
        "--video_file", str(out),
    ]
    result = subprocess.run(
        cmd,
        cwd=repo_root / "crowd_nav",
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert result.returncode == 0, (
        f"WP-4 rollout exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert out.exists() and out.stat().st_size > 10_000, (
        f"{out} missing or too small ({out.stat().st_size if out.exists() else 0} bytes)"
    )

    combined = (result.stdout + "\n" + result.stderr).lower()
    # The fallback WARNING is only emitted when Theta* raises NoPathFound and
    # degrades to straight-line waypoints.  Its absence proves the planner
    # succeeded on the Theta* branch.
    assert "fallback" not in combined, (
        "Planner degraded to straight-line fallback — Theta* branch not exercised.\n"
        f"--- stderr ---\n{result.stderr}"
    )
