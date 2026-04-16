import os
import stat
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "setup_env.sh"


@pytest.mark.unit
def test_setup_script_exists_and_is_executable():
    assert SCRIPT.exists(), f"missing {SCRIPT}"
    mode = SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, f"{SCRIPT} must be executable (chmod +x)"


@pytest.mark.unit
def test_setup_script_uses_strict_bash_flags():
    """set -euo pipefail is required so a failed pip install aborts the whole script."""
    body = SCRIPT.read_text()
    assert body.startswith("#!/usr/bin/env bash"), "must use portable shebang"
    assert "set -euo pipefail" in body, "must use strict bash flags"


@pytest.mark.unit
def test_setup_script_covers_all_install_steps():
    body = SCRIPT.read_text()
    required_tokens = [
        "conda create",
        "Python-RVO2-main",
        "pip install -e .",
        "gym==0.15.7",
    ]
    missing = [t for t in required_tokens if t not in body]
    assert not missing, f"setup_env.sh missing steps: {missing}"
