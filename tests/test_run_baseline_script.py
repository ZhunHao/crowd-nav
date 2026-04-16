import stat
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "run_baseline.sh"


@pytest.mark.unit
def test_run_baseline_script_exists_and_is_executable():
    assert SCRIPT.exists(), f"missing {SCRIPT}"
    assert SCRIPT.stat().st_mode & stat.S_IXUSR


@pytest.mark.unit
def test_run_baseline_invokes_test_py_with_video_flag():
    body = SCRIPT.read_text()
    assert "crowd_nav/test.py" in body or "test.py" in body
    assert "--policy sarl" in body
    assert "--model_dir data/output_trained" in body
    assert "--video_file" in body
    assert "exports/baseline.mp4" in body
    # --visualize is required to run one rollout per goal (vs. 500 test-size
    # episodes) and populate the position_list that render() animates into the
    # mp4. With MPLBACKEND=Agg (exported above in the script) it is headless.
    assert "--visualize" in body, "run_baseline.sh needs --visualize for the video rollout"
    assert 'MPLBACKEND="${MPLBACKEND:-Agg}"' in body, "must export MPLBACKEND=Agg for CI headlessness"


@pytest.mark.unit
def test_gitignore_excludes_export_artifacts():
    gi = (REPO / ".gitignore").read_text()
    for entry in ["exports/*.mp4", "exports/*.csv", "__pycache__/", ".pytest_cache/"]:
        assert entry in gi, f".gitignore missing {entry!r}"
