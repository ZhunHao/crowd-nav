import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.mark.integration
def test_test_py_accepts_seed_flag():
    """test.py must expose --seed so reruns are reproducible."""
    result = subprocess.run(
        [sys.executable, "crowd_nav/test.py", "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--seed" in result.stdout, f"--seed flag missing from test.py --help:\n{result.stdout}"
