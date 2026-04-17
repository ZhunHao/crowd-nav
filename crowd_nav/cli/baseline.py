"""`crowdnav-baseline` — reproduce R1 by shelling out to scripts/run_baseline.sh.

Why shell out? The script is already CI-validated and documents the exact
flags for R1. Duplicating the argv here would create two sources of truth.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    script = _repo_root() / "scripts" / "run_baseline.sh"
    proc = subprocess.run(["bash", str(script)], cwd=str(_repo_root()))
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
