# tests/test_smoke_wp3.py
"""WP-3 smoke: with a static map loaded, a headless rollout writes a non-empty
mp4. Uses the existing data/output_trained env config (static_obs=true,
static_obs_shapes=rect) so the static-map code path wires through
test.py -> env -> render -> ffmpeg.

NOTE: test.py has no --static_map CLI flag. WP-3 static-map support is wired
through the env config [sim] section (static_obs=true). The map_loader.py PNG
path is part of WP-5, not WP-3. This test exercises the WP-3 code path by
using the real data/output_trained config, which already enables static_obs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.slow]


def test_static_map_rollout_produces_video(repo_root: Path, exports_dir: Path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")

    out = exports_dir / "wp3_smoke.mp4"
    if out.exists():
        out.unlink()

    # data/output_trained/env.config has [sim] static_obs=true, static_obs_shapes=rect
    # and [static_map] enabled=true — this exercises the WP-3 StaticMap code path.
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
        f"WP-3 rollout exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert out.exists() and out.stat().st_size > 10_000, f"{out} missing or tiny"
