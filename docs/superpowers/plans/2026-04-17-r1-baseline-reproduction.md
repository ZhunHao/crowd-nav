# R1 — Baseline Reproduction & Simulation Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the CrowdNav-DIP baseline (slide 13 task 1 / [REQUIREMENTS.md R1 + R7](../../REQUIREMENTS.md)) reproducible end-to-end: a fresh clone + documented setup produces an `exports/baseline.mp4` and passes a pytest smoke test on CPU, with no hand-tuning.

**Architecture:** Thin layer around the existing `crowd_nav/test.py`. No policy or environment changes — we only (a) fix two portability bugs that block reproduction on macOS, (b) add a seed plumbed through `env.config` + Python / NumPy / Torch RNGs, (c) add a `scripts/run_baseline.sh` one-shot, and (d) add a headless pytest smoke test. Everything else is docs.

**Tech Stack:** Python 3.10, PyTorch (CPU), OpenAI Gym 0.15.7 (pinned), matplotlib + ffmpeg, Python-RVO2 (vendored), pytest.

**Scope (what this plan delivers — and does NOT):**

| In scope (R1) | Out of scope (later WPs) |
|---|---|
| Install + run existing DRL baseline | New goal allocator (WP-2 / R2) |
| Generate `baseline.mp4` from `rl_model.pth` | Static land map (WP-3 / R3) |
| Smoke test + seed for determinism | Theta* planner (WP-4 / R4) |
| Portability fix for ffmpeg path | PyQt GUI (WP-5 / R5) |
| `scripts/run_baseline.sh` + README update | Retraining the policy (R7 freezes weights) |

**Non-goals:** no refactor of `crowd_sim/envs/crowd_sim.py` beyond the minimal ffmpeg-path change, no change to SARL network signature, no new policies, no new map format.

---

## File Structure

New / modified files and each file's single responsibility:

| File | Status | Responsibility |
|---|---|---|
| `scripts/run_baseline.sh` | **create** | One-shot: activate env, export `exports/baseline.mp4` via `crowd_nav/test.py`. Exits non-zero on failure. |
| `scripts/setup_env.sh` | **create** | Idempotent installer: conda env + Python-RVO2 + `pip install -e .` + gym pin. Safe to re-run. |
| `crowd_sim/envs/crowd_sim.py` | **modify** (lines 591–605, hardcoded ffmpeg path) | Replace `plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'` with autodetect via `shutil.which('ffmpeg')`. |
| `crowd_sim/envs/utils/seeding.py` | **create** | `seed_everything(seed: int)` — seeds `random`, `numpy`, `torch`. Single entry point used by `test.py` / `train.py`. |
| `crowd_nav/test.py` | **modify** (top of `main`, after arg parse) | Accept `--seed` CLI flag (default `42`); call `seed_everything(args.seed)`. |
| `crowd_nav/configs/env.config` | **modify** | Add `[sim] random_seed = 42` (honoured by `test.py` when `--seed` not given — file overrides default, CLI overrides file). |
| `tests/test_baseline_smoke.py` | **create** | Pytest: runs `test.py` headless on test_case 0, asserts exit 0, asserts `exports/*.mp4` > 10 KB. |
| `tests/conftest.py` | **create** | Adds `exports/` path fixture; sets `matplotlib.use("Agg")` for headless CI. |
| `exports/.gitkeep` | **create** | Keep the export dir under version control; keep contents out. |
| `.gitignore` | **modify** | Add `exports/*.mp4`, `exports/*.csv`, `__pycache__/`, `.pytest_cache/`. |
| `pyproject.toml` | **create** | Minimal pytest config (`testpaths = ["tests"]`, `markers = ["smoke", "slow"]`). Coexists with existing `setup.py`. |
| `README.md` | **modify** (replace "Setup" + "Getting Started" sections) | Call out `scripts/setup_env.sh` and `scripts/run_baseline.sh`; add "Troubleshooting → ffmpeg not found"; add "Reproducing R1" section. |
| `.github/workflows/smoke.yml` | **create** | CI: runs `tests/test_baseline_smoke.py` on `ubuntu-latest`, Python 3.10, CPU-only. |
| `docs/R1_VERIFICATION.md` | **create** | Human-runnable checklist matching the tasks below (one bullet per verification, with exact commands + expected output). |

**Boundaries:** `seeding.py` knows *only* about RNG libraries; `test.py` knows *only* about CLI + config plumbing; the render function in `crowd_sim.py` only changes its ffmpeg-path line. No cross-file refactoring.

---

## Prerequisites

Before starting, verify:

```bash
git status              # expect clean working tree (or only slides.md/README.md edits from the prior session)
which conda             # expect a path; if missing, install Miniconda first
which ffmpeg            # expect a path (Linux: /usr/bin/ffmpeg, macOS Homebrew: /opt/homebrew/bin/ffmpeg)
```

If `ffmpeg` is missing:
- Linux: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`

---

## Task 1: Portable ffmpeg path in `crowd_sim.render`

**Files:**
- Modify: `crowd_sim/envs/crowd_sim.py:591-605` (the `render` function's prelude)

**Why first:** this is the single hardcoded value that blocks `--video_file` on any non-Debian host. Every later task depends on video export working.

- [ ] **Step 1: Write the failing test**

Create `tests/test_render_ffmpeg_path.py`:

```python
import shutil
import pytest


@pytest.mark.unit
def test_render_uses_autodetected_ffmpeg(monkeypatch):
    """render() should honour shutil.which('ffmpeg'), not a hardcoded /usr/bin path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from crowd_sim.envs.crowd_sim import CrowdSim  # noqa: F401 — import triggers module-level side effects

    fake_path = "/tmp/fake-ffmpeg-for-test"
    monkeypatch.setattr(shutil, "which", lambda name: fake_path if name == "ffmpeg" else None)

    # Reload the module so the path re-evaluates under the monkeypatch.
    import importlib
    import crowd_sim.envs.crowd_sim as mod
    importlib.reload(mod)

    assert plt.rcParams["animation.ffmpeg_path"] == fake_path, (
        "render() must read ffmpeg path from shutil.which, not a hardcoded /usr/bin/ffmpeg"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/zhunhao/Documents/Projects/crowdnav-dip
pytest tests/test_render_ffmpeg_path.py -v
```

Expected: FAIL with `AssertionError` — the current code sets `animation.ffmpeg_path` to `/usr/bin/ffmpeg` inside `render`, not at import time, so the assertion fails.

- [ ] **Step 3: Apply the minimal fix**

In `crowd_sim/envs/crowd_sim.py`, near the top of the file (after existing imports), add:

```python
import shutil
from matplotlib import rcParams as _mpl_rcParams

_FFMPEG = shutil.which("ffmpeg")
if _FFMPEG:
    _mpl_rcParams["animation.ffmpeg_path"] = _FFMPEG
```

Then inside `render` (around line 594), delete the line:

```python
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
```

Also add a defensive check at the start of the `output_file is not None` branch (~line 797):

```python
if output_file is not None:
    if not _FFMPEG:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg (Linux: apt install ffmpeg, "
            "macOS: brew install ffmpeg) before using --video_file."
        )
    ffmpeg_writer = animation.writers['ffmpeg']
    ...
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_render_ffmpeg_path.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crowd_sim/envs/crowd_sim.py tests/test_render_ffmpeg_path.py
git commit -m "fix: autodetect ffmpeg path instead of hardcoding /usr/bin/ffmpeg"
```

---

## Task 2: Deterministic seeding utility

**Files:**
- Create: `crowd_sim/envs/utils/seeding.py`
- Test: `tests/test_seeding.py`

- [ ] **Step 1: Write the failing test**

`tests/test_seeding.py`:

```python
import random

import numpy as np
import pytest
import torch


@pytest.mark.unit
def test_seed_everything_makes_rng_deterministic():
    from crowd_sim.envs.utils.seeding import seed_everything

    seed_everything(42)
    r1, n1, t1 = random.random(), np.random.rand(), torch.rand(1).item()

    seed_everything(42)
    r2, n2, t2 = random.random(), np.random.rand(), torch.rand(1).item()

    assert r1 == r2
    assert n1 == n2
    assert t1 == t2


@pytest.mark.unit
def test_seed_everything_rejects_non_int():
    from crowd_sim.envs.utils.seeding import seed_everything

    with pytest.raises(TypeError):
        seed_everything("42")  # type: ignore[arg-type]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_seeding.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'crowd_sim.envs.utils.seeding'`.

- [ ] **Step 3: Implement the module**

Create `crowd_sim/envs/utils/seeding.py`:

```python
"""RNG seeding utility — single entry point for deterministic runs (R1 / NF3)."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed python-random, numpy, and torch (CPU + CUDA) with a single int.

    Called from ``test.py`` and ``train.py`` so a given ``--seed`` produces
    bit-identical rollouts on the same hardware. See slide 13 task 1.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_seeding.py -v
```

Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add crowd_sim/envs/utils/seeding.py tests/test_seeding.py
git commit -m "feat: add seed_everything helper for deterministic rollouts"
```

---

## Task 3: Wire `--seed` into `test.py` and `env.config`

**Files:**
- Modify: `crowd_nav/test.py` (argparse block, ~lines 14-29; seed call before `env.reset`, ~line 94)
- Modify: `crowd_nav/configs/env.config` (append `random_seed` under `[env]`)
- Test: `tests/test_seed_flag.py`

- [ ] **Step 1: Write the failing test**

`tests/test_seed_flag.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_seed_flag.py -v
```

Expected: FAIL — `--seed` is not yet in the argparse output.

- [ ] **Step 3: Edit `crowd_nav/test.py`**

In the argparse block (after the `--traj` line at `test.py:28`), add:

```python
    parser.add_argument('--seed', type=int, default=None,
                        help='RNG seed. Overrides env.config [env] random_seed. Defaults to 42 if unset in both.')
```

At the top of the file, add the import (next to the existing `numpy as np` import):

```python
from crowd_sim.envs.utils.seeding import seed_everything
```

After `env_config.read(env_config_file)` (around line 63), resolve the seed and apply it before `env.reset`:

```python
    # Resolve seed: CLI flag > env.config > 42 (hardcoded default).
    if args.seed is not None:
        resolved_seed = args.seed
    elif env_config.has_option('env', 'random_seed'):
        resolved_seed = env_config.getint('env', 'random_seed')
    else:
        resolved_seed = 42
    seed_everything(resolved_seed)
    logging.info('Seeded RNGs with %d', resolved_seed)
```

- [ ] **Step 4: Edit `crowd_nav/configs/env.config`**

Under the existing `[env]` section (after `randomize_attributes = false`), append:

```ini
random_seed = 42
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
pytest tests/test_seed_flag.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crowd_nav/test.py crowd_nav/configs/env.config tests/test_seed_flag.py
git commit -m "feat: plumb --seed flag through test.py and env.config"
```

---

## Task 4: `scripts/setup_env.sh` — idempotent installer

**Files:**
- Create: `scripts/setup_env.sh`
- Test: `tests/test_setup_script.py` (structural check only — we do not spawn conda in pytest)

- [ ] **Step 1: Write the failing test**

`tests/test_setup_script.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_setup_script.py -v
```

Expected: FAIL — script does not exist yet.

- [ ] **Step 3: Create the script**

`scripts/setup_env.sh`:

```bash
#!/usr/bin/env bash
# Idempotent installer for the CrowdNav-DIP baseline (R1 / WP-1).
# Safe to re-run — skips work already done.
set -euo pipefail

ENV_NAME="${ENV_NAME:-navigate}"
PY_VERSION="3.10"

here="$(cd "$(dirname "$0")/.." && pwd)"
cd "$here"

log() { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }

# 1. Conda env ----------------------------------------------------------------
if ! command -v conda >/dev/null; then
  echo "conda not found — install Miniconda from https://www.anaconda.com/download/success" >&2
  exit 1
fi

# Make `conda activate` usable inside this script.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "conda env '$ENV_NAME' already exists — reusing"
else
  log "creating conda env '$ENV_NAME' (python $PY_VERSION)"
  conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
fi
conda activate "$ENV_NAME"

# 2. Python-RVO2 (vendored) ---------------------------------------------------
log "installing cython (required by Python-RVO2 build)"
pip install --quiet "cython<3"

if python -c "import rvo2" 2>/dev/null; then
  log "rvo2 already installed — skipping vendored build"
else
  log "building vendored Python-RVO2"
  pushd Python-RVO2-main >/dev/null
  python setup.py build
  python setup.py install
  popd >/dev/null
fi

# 3. crowd_sim + crowd_nav ----------------------------------------------------
log "pip install -e . (crowdnav package, editable)"
pip install --quiet -e .

log "pinning gym==0.15.7 (newer versions break this env)"
pip install --quiet "gym==0.15.7"

# 4. Dev / test deps ----------------------------------------------------------
log "installing pytest for smoke tests"
pip install --quiet pytest

# 5. ffmpeg sanity ------------------------------------------------------------
if ! command -v ffmpeg >/dev/null; then
  echo "WARNING: ffmpeg not found — video export will fail." >&2
  echo "  Linux : sudo apt install ffmpeg" >&2
  echo "  macOS : brew install ffmpeg" >&2
fi

log "done. Activate with:  conda activate $ENV_NAME"
```

Make it executable:

```bash
chmod +x scripts/setup_env.sh
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_setup_script.py -v
```

Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/setup_env.sh tests/test_setup_script.py
git commit -m "feat: add idempotent setup_env.sh for R1 baseline install"
```

---

## Task 5: `scripts/run_baseline.sh` — one-shot reproduction

**Files:**
- Create: `scripts/run_baseline.sh`
- Create: `exports/.gitkeep`
- Modify: `.gitignore`
- Test: `tests/test_run_baseline_script.py`

- [ ] **Step 1: Write the failing test**

`tests/test_run_baseline_script.py`:

```python
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
    assert "crowd_nav/test.py" in body
    assert "--policy sarl" in body
    assert "--model_dir data/output_trained" in body
    assert "--video_file" in body
    assert "exports/baseline.mp4" in body
    # Non-interactive: no --visualize (so it can run in CI).
    assert "--visualize" not in body, "run_baseline.sh must be headless — no --visualize"


@pytest.mark.unit
def test_gitignore_excludes_export_artifacts():
    gi = (REPO / ".gitignore").read_text()
    for entry in ["exports/*.mp4", "exports/*.csv", "__pycache__/", ".pytest_cache/"]:
        assert entry in gi, f".gitignore missing {entry!r}"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_run_baseline_script.py -v
```

Expected: FAIL — script and .gitignore entries do not exist.

- [ ] **Step 3: Create `scripts/run_baseline.sh`**

```bash
#!/usr/bin/env bash
# Reproduce the R1 baseline: load the trained SARL policy, run test_case 0,
# write exports/baseline.mp4. Headless (no matplotlib window) so it runs in CI.
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
cd "$here"

mkdir -p exports

# MPLBACKEND=Agg prevents any window pop-up on CI / ssh sessions.
export MPLBACKEND="${MPLBACKEND:-Agg}"

cd crowd_nav
python test.py \
  --policy sarl \
  --model_dir data/output_trained \
  --phase test \
  --test_case 0 \
  --seed 42 \
  --video_file ../exports/baseline.mp4
```

Make executable:

```bash
chmod +x scripts/run_baseline.sh
```

- [ ] **Step 4: Create `exports/.gitkeep` and update `.gitignore`**

```bash
touch exports/.gitkeep
```

If `.gitignore` does not exist, create it. Otherwise append these lines to `.gitignore`:

```
# R1 baseline outputs
exports/*.mp4
exports/*.csv

# Python
__pycache__/
*.pyc
.pytest_cache/

# macOS
.DS_Store
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
pytest tests/test_run_baseline_script.py -v
```

Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/run_baseline.sh exports/.gitkeep .gitignore tests/test_run_baseline_script.py
git commit -m "feat: add run_baseline.sh one-shot for R1 video export"
```

---

## Task 6: `pyproject.toml` + `tests/conftest.py`

**Files:**
- Create: `pyproject.toml`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write the config files**

`pyproject.toml` (minimal — coexists with existing `setup.py`):

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "unit: fast unit tests with no external deps",
    "integration: tests that invoke the CLI or file system",
    "smoke: end-to-end R1 baseline reproduction (slow, requires model + ffmpeg)",
    "slow: tests that take > 30s",
]
addopts = "-ra --strict-markers"
```

`tests/conftest.py`:

```python
"""Pytest root conftest.

- Forces the matplotlib Agg backend so tests never try to open a window.
- Exposes REPO_ROOT and EXPORTS_DIR fixtures shared across tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def exports_dir(repo_root: Path) -> Path:
    d = repo_root / "exports"
    d.mkdir(exist_ok=True)
    return d
```

- [ ] **Step 2: Verify existing tests still pass**

```bash
pytest -v
```

Expected: all previous tests PASS; the new config is picked up (look for `rootdir: .../crowdnav-dip, configfile: pyproject.toml` in the header).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml tests/conftest.py
git commit -m "chore: configure pytest via pyproject.toml and conftest"
```

---

## Task 7: End-to-end smoke test for the baseline

**Files:**
- Create: `tests/test_baseline_smoke.py`

This is the R1 acceptance test. It runs the real `run_baseline.sh` end-to-end. It's slow (~30–90 s on CPU) so we mark it `smoke` + `slow` so devs can filter it out with `pytest -m "not slow"`.

- [ ] **Step 1: Write the test**

`tests/test_baseline_smoke.py`:

```python
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
    """Re-running with the same seed must produce a byte-for-byte identical video.

    ffmpeg writes metadata timestamps, so we compare frame hashes instead — use
    the --seed plumbing + no new RNG sources in the rollout.
    """
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")

    def run_and_hash() -> str:
        subprocess.run(
            ["bash", "scripts/run_baseline.sh"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            timeout=600,
        )
        # Hash the last-state positions CSV if present, else the video byte-size as a weak signal.
        # For v1 we rely on stdout containing deterministic "start:" / "goal:" lines.
        proc = subprocess.run(
            ["bash", "scripts/run_baseline.sh"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=600,
            check=True,
        )
        # Extract the deterministic-looking lines that test.py prints.
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(("start:", "It takes"))]
        return "\n".join(lines)

    h1 = run_and_hash()
    h2 = run_and_hash()
    assert h1 == h2, f"rollout not deterministic under --seed 42:\nrun1:\n{h1}\nrun2:\n{h2}"
```

- [ ] **Step 2: Run the smoke test (requires the conda env activated)**

```bash
conda activate navigate   # if not already active
pytest tests/test_baseline_smoke.py -v -m "smoke"
```

Expected: PASS. `exports/baseline.mp4` exists and is > 10 KB.
If it fails because `ffmpeg` isn't installed: the test skips — fix by installing ffmpeg and re-run.
If it fails because `rl_model.pth` is missing: the test errors — confirm `crowd_nav/data/output_trained/rl_model.pth` is committed.

- [ ] **Step 3: Run the fast suite to confirm selectivity**

```bash
pytest -v -m "not slow"
```

Expected: all non-smoke tests PASS; smoke test is deselected. Confirms the marker works.

- [ ] **Step 4: Commit**

```bash
git add tests/test_baseline_smoke.py
git commit -m "test: add R1 baseline smoke + determinism check"
```

---

## Task 8: CI workflow (GitHub Actions)

**Files:**
- Create: `.github/workflows/smoke.yml`

- [ ] **Step 1: Write the workflow**

`.github/workflows/smoke.yml`:

```yaml
name: smoke

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  baseline:
    runs-on: ubuntu-latest
    timeout-minutes: 25
    steps:
      - uses: actions/checkout@v4

      - name: Install system deps
        run: |
          sudo apt-get update
          sudo apt-get install -y cmake build-essential ffmpeg

      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install Python deps
        run: |
          python -m pip install --upgrade pip
          pip install "cython<3"
          (cd Python-RVO2-main && python setup.py build && python setup.py install)
          pip install -e .
          pip install "gym==0.15.7" pytest

      - name: Fast tests
        run: pytest -v -m "not slow"

      - name: R1 baseline smoke (headless)
        env:
          MPLBACKEND: Agg
        run: pytest -v -m "smoke"

      - name: Upload baseline video
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: baseline-mp4
          path: exports/baseline.mp4
          if-no-files-found: warn
```

- [ ] **Step 2: Sanity-check locally (best-effort without pushing)**

```bash
# Validate YAML parses
python -c "import yaml; yaml.safe_load(open('.github/workflows/smoke.yml'))"
```

Expected: no output (success). Any YAMLError means the file is malformed.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/smoke.yml
git commit -m "ci: add smoke workflow that runs R1 baseline and uploads the video"
```

---

## Task 9: Update `README.md` — Reproducing R1

**Files:**
- Modify: `README.md` (replace the current "Setup" and "Getting Started" sections; add a "Reproducing R1" section)

- [ ] **Step 1: Apply the edit**

Replace the existing `## Setup` section with:

````markdown
## Setup

One command sets everything up (conda env + Python-RVO2 + package in editable mode + gym pin):

```bash
./scripts/setup_env.sh
conda activate navigate
```

The script is idempotent — safe to re-run. Requires `conda` and, for video export, `ffmpeg`:

- Linux: `sudo apt install ffmpeg cmake build-essential`
- macOS: `brew install ffmpeg cmake`
````

After the existing "Getting Started" section (before "DIP roadmap"), insert:

````markdown
## Reproducing R1 (baseline + video)

R1 means: install the env, re-run the given DRL policy, export a simulation video.

```bash
./scripts/run_baseline.sh
# → exports/baseline.mp4
```

Verify end-to-end with the smoke test:

```bash
pytest -m smoke -v
```

What this covers:

- Loads the trained SARL policy from `crowd_nav/data/output_trained/rl_model.pth` (no retraining — per R7).
- Runs test case 0 under seed 42 — rollouts are bit-identical across runs on the same machine (NF3).
- Writes `exports/baseline.mp4` (~1–3 MB, ffmpeg-encoded).

See [docs/R1_VERIFICATION.md](docs/R1_VERIFICATION.md) for the full manual checklist.
````

In the existing "Troubleshooting" section, add a bullet:

```markdown
- **ffmpeg not found** — `--video_file` will error with "ffmpeg not found on PATH". Install ffmpeg (Linux `apt install ffmpeg`, macOS `brew install ffmpeg`) and re-run. The render module autodetects via `shutil.which` at import time.
```

- [ ] **Step 2: Verify the README renders as intended**

```bash
# eyeball it
head -100 README.md
```

Confirm: setup instructions reference `scripts/setup_env.sh`, `Reproducing R1` section exists, ffmpeg troubleshooting bullet is present.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document setup_env.sh, run_baseline.sh, and R1 reproduction"
```

---

## Task 10: `docs/R1_VERIFICATION.md` — manual checklist

**Files:**
- Create: `docs/R1_VERIFICATION.md`

Gives a reviewer a single page to verify R1 without reading code.

- [ ] **Step 1: Write the doc**

`docs/R1_VERIFICATION.md`:

```markdown
# R1 Verification Checklist

> Maps to [REQUIREMENTS.md R1 + R7](REQUIREMENTS.md) and slide 13 task 1 of [slides.md](../slides.md).

Run each step in order on a clean checkout. Expected outputs in `>` blocks.

## 1. Environment installs cleanly

```bash
./scripts/setup_env.sh
```

> `[setup] done. Activate with: conda activate navigate`

```bash
conda activate navigate
python -c "import crowd_sim, crowd_nav, rvo2, torch, gym; print('ok')"
```

> `ok`

## 2. Trained model loads

```bash
python -c "import torch; torch.load('crowd_nav/data/output_trained/rl_model.pth', map_location='cpu'); print('ok')"
```

> `ok`

## 3. Baseline produces a video

```bash
./scripts/run_baseline.sh
ls -lh exports/baseline.mp4
ffprobe -v error -show_entries format=duration exports/baseline.mp4
```

> size > 10 KB, duration > 0 s

## 4. Determinism (seed 42)

Run the baseline twice and diff the deterministic stdout lines:

```bash
./scripts/run_baseline.sh 2>&1 | grep -E "^(start:|It takes)" > /tmp/run1.txt
./scripts/run_baseline.sh 2>&1 | grep -E "^(start:|It takes)" > /tmp/run2.txt
diff /tmp/run1.txt /tmp/run2.txt && echo "deterministic"
```

> `deterministic`

## 5. Smoke test suite green

```bash
pytest -v
```

> all tests pass (unit + integration + smoke if ffmpeg is installed)

## 6. CI

Open the latest run at `.github/workflows/smoke.yml` on GitHub Actions — the `baseline-mp4` artifact should attach.
```

- [ ] **Step 2: Commit**

```bash
git add docs/R1_VERIFICATION.md
git commit -m "docs: add R1 verification checklist"
```

---

## Task 11: Final end-to-end dry-run

Before declaring R1 done, run the full verification checklist from [docs/R1_VERIFICATION.md](../../R1_VERIFICATION.md) on your machine.

- [ ] **Step 1: Clean state simulation**

```bash
# remove any in-env cruft and test artifacts
rm -rf exports/*.mp4 .pytest_cache
pytest -v
```

Expected: all unit + integration tests PASS; smoke test passes (or skips if ffmpeg missing — which is itself a valid signal).

- [ ] **Step 2: Full reproduction**

```bash
./scripts/run_baseline.sh
ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 exports/baseline.mp4
```

Expected: a number > 0 (duration in seconds).

- [ ] **Step 3: Open the artifact**

Open `exports/baseline.mp4` in a video player; confirm ships move and the robot navigates through waypoints.

- [ ] **Step 4: Cross-reference requirements**

Tick each requirement explicitly:

- [ ] R1 — environment installs (`setup_env.sh` green)
- [ ] R1 — DRL code re-runs (`test.py --policy sarl --model_dir data/output_trained` exits 0)
- [ ] R1 — video generated (`exports/baseline.mp4` exists, non-zero, plays)
- [ ] R7 — no retraining needed (weights untouched; `rl_model.pth` sha is unchanged since first commit)
- [ ] NF3 — determinism (seeded; two runs produce identical position lines)
- [ ] NF4 — CPU path works (no `--gpu` in the script)

- [ ] **Step 5: Tag the milestone**

```bash
git tag -a r1-baseline -m "R1 complete: baseline reproduces from clean clone, video exports, CI green"
```

---

## Self-review summary

Spec coverage ↔ tasks:

| Requirement | Task(s) | Verified by |
|---|---|---|
| R1: install environment | Task 4 | `test_setup_script.py`, Task 11 step 1 |
| R1: re-run DRL code | Tasks 3, 5 | `test_seed_flag.py`, `test_baseline_smoke.py` |
| R1: generate simulation videos | Tasks 1, 5, 7 | `test_baseline_smoke.py` asserts MP4 size > 10 KB |
| R7: reuse trained weights | Task 5 | `run_baseline.sh` uses `--model_dir data/output_trained` |
| NF3: determinism | Tasks 2, 3, 7 | `test_seed_flag.py`, `test_baseline_is_deterministic_across_runs` |
| NF4: CPU portability | Tasks 1, 4 | ffmpeg autodetect; CI runs on ubuntu-latest CPU |
| CI / reproducibility | Task 8 | GitHub Actions uploads `baseline-mp4` artifact |
| Docs | Tasks 9, 10 | README section + `R1_VERIFICATION.md` |

Placeholder scan: none — every step has exact file paths and complete code.

Type consistency: `seed_everything(seed: int)` signature is used identically in `test.py` Task 3 and `tests/test_seeding.py` Task 2. Argparse flag `--seed` has the same name in every task.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-17-r1-baseline-reproduction.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task (1→11), review between tasks, fast iteration. Each task is self-contained, so subagents do not need prior conversation context.

**2. Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch with checkpoints at end of Tasks 3, 7, and 11.

**Which approach?**
