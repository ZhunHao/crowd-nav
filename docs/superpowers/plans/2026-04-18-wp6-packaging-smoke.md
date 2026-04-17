# WP-6 Packaging + Smoke Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Final plan path:** after plan approval, move this file to `docs/superpowers/plans/2026-04-18-wp6-packaging-smoke.md` so it lives alongside WP-2/3/5 plans in the repo. (Plan mode forbids writing there right now.)

**Context:** WP-1 through WP-5 are merged (baseline, GoalAllocator, StaticMap, Theta*, PyQt GUI). Before the group demo we still owe the cross-cutting WP-6 per `docs/REQUIREMENTS.md:26` — packaging and smoke tests. Today a fresh `pip install -e .` silently produces a broken env (no `gym` pin, no `python_requires`, vendored Python-RVO2 not installed), students have to remember `python -m crowd_nav.gui` vs `bash scripts/run_baseline.sh`, and CI only smoke-tests R1 — WP-3/4/5 have no end-to-end gate. WP-6 closes those three gaps so the project ships cleanly.

**Goal:** Make `pip install -e .` pin the historically-silent deps, give students three one-word console commands (`crowdnav-preflight`, `crowdnav-baseline`, `crowdnav-gui`), and extend `pytest -m smoke` so it exercises every demo feature (R1 baseline + WP-3 static map + WP-4 Theta\* + WP-5 GUI), gated in CI.

**Architecture:** Pin the three historically-silent deps (`gym==0.15.7`, `cython<3`, `python_requires=">=3.10,<3.11"`) in `setup.py`. Register `[console_scripts]` entry points that shell out to the existing `scripts/run_baseline.sh` and `python -m crowd_nav.gui` paths, plus a new `crowd_nav/utils/preflight.py` importability / ffmpeg / model-file verifier. Add one smoke test per remaining workpackage (WP-3, WP-4, WP-5) that drives the same `test.py` / GUI entry points used manually in `docs/R1_VERIFICATION.md` — the tests, not new scripts, are the one-command smoke driver. Extend `.github/workflows/smoke.yml` to install PyQt5 and run `pytest -m smoke` end-to-end under `QT_QPA_PLATFORM=offscreen`.

**Tech Stack:** `setuptools` (already in use), `pytest`, `pytest-qt` (WP-5), `argparse` for the preflight CLI, `importlib`/`shutil.which` for dependency probes. No new runtime deps.

**Assumptions carried forward:**
- `scripts/run_baseline.sh` and `python -m crowd_nav.gui` are the canonical entry points (WP-1, WP-5).
- `tests/conftest.py::repo_root` and `exports_dir` fixtures exist and are usable by new smoke tests.
- `tests/gui/conftest.py` forces `QT_QPA_PLATFORM=offscreen` for every GUI test (WP-5).
- `crowd_nav/data/output_trained/rl_model.pth` is committed and loadable (R7).
- Python-RVO2 stays a separate install step (`scripts/setup_env.sh` step 2); WP-6 only adds a preflight check that `import rvo2` succeeds.
- `crowd_nav/test.py` already parses `--static_map` (WP-3 commit `b581827`) and `--planner` (WP-4 commit `8f5a656`); Task 3/4 verify this before relying on it.

---

## File Structure

### New files

```
crowd_nav/utils/preflight.py              # Preflight: rvo2 import, ffmpeg, model, gym pin
crowd_nav/cli/__init__.py                  # (empty) package marker for console-script module
crowd_nav/cli/baseline.py                  # Entry point: crowdnav-baseline -> runs scripts/run_baseline.sh
crowd_nav/cli/gui.py                       # Entry point: crowdnav-gui -> calls crowd_nav.gui.__main__
crowd_nav/cli/preflight.py                 # Entry point: crowdnav-preflight -> argparse wrapper
tests/test_preflight.py                    # Unit tests for preflight checks
tests/test_console_scripts.py              # Integration: each console script is importable + dispatches correctly
tests/test_setup_py_pins.py                # Unit: setup.py declares gym==0.15.7, cython<3, python_requires
tests/test_ci_workflow.py                  # Unit: .github/workflows/smoke.yml installs PyQt5 + runs smoke marker
tests/test_smoke_wp3.py                    # Smoke: static map loaded -> rollout mp4 written
tests/test_smoke_wp4.py                    # Smoke: Theta* enabled -> rollout mp4 + no fallback warning
tests/test_smoke_wp5.py                    # Smoke: GUI end-to-end (re-uses tests/gui/test_end_to_end driver)
```

### Files modified

| File | Change |
|------|--------|
| `setup.py` | add `python_requires`, pin `gym==0.15.7` + `cython<3` in `install_requires`, add `crowd_nav.cli` to `packages`, register `console_scripts` entry points |
| `pyproject.toml:3-9` | no new markers — existing `smoke`, `gui`, `unit`, `integration` markers cover every new test |
| `.github/workflows/smoke.yml:25-50` | install `.[test]` (PyQt5 + pytest-qt); run `pytest -m smoke` with `QT_QPA_PLATFORM=offscreen` in place of the single-marker R1 step; upload all four smoke artifacts |
| `README.md:36-86` | show `crowdnav-{preflight,baseline,gui}` as the first-class commands; keep the `python test.py ...` block as the advanced path |
| `scripts/setup_env.sh:54` | append `crowdnav-preflight` as the final step so `setup_env.sh` fails loudly on broken envs |
| `tests/gui/test_end_to_end.py` | extract body of `test_gui_full_voyage` into `_run_full_gui_voyage(...)` so WP-5 smoke can reuse it (behaviour-preserving) |
| `docs/REQUIREMENTS.md:26` | flip WP-6 status to "Implemented 2026-04-18" |

### Reuse (do NOT re-implement)

- `scripts/run_baseline.sh` — the `crowdnav-baseline` entry point shells out to this; no logic duplication.
- `crowd_nav.gui.__main__` — `crowdnav-gui` calls `main()` from there.
- `tests/conftest.py::repo_root, exports_dir` — reuse in new smoke tests.
- `tests/gui/conftest.py` — already forces offscreen Qt (no change).
- `tests/gui/test_end_to_end.py` — WP-5 smoke imports the extracted driver; does not rewrite.
- `crowd_nav/test.py` CLI flags (`--static_map`, `--planner`, `--seed`, `--video_file`) — WP-3/4 smokes drive the existing CLI, no new flags.

---

## Task 0: Pin deps + `python_requires` in `setup.py`

**Files:**
- Modify: `setup.py`
- Create: `tests/test_setup_py_pins.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_setup_py_pins.py
"""WP-6 packaging pins: setup.py must declare the three historically-silent
deps (gym version, cython ceiling, python_requires) so fresh installs fail
loudly instead of importing broken code paths."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def setup_src() -> str:
    return (REPO / "setup.py").read_text()


def test_python_requires_pinned_to_310(setup_src: str):
    assert "python_requires='>=3.10,<3.11'" in setup_src \
        or 'python_requires=">=3.10,<3.11"' in setup_src, \
        "setup.py must declare python_requires='>=3.10,<3.11'"


def test_gym_pinned_to_0_15_7(setup_src: str):
    assert "'gym==0.15.7'" in setup_src or '"gym==0.15.7"' in setup_src, \
        "gym==0.15.7 must appear in install_requires"


def test_cython_capped_below_3(setup_src: str):
    assert "'cython<3'" in setup_src or '"cython<3"' in setup_src, \
        "cython<3 must appear in install_requires"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_setup_py_pins.py -v`
Expected: 3 failures — each required substring missing.

- [ ] **Step 3: Modify `setup.py`**

```python
from setuptools import setup


setup(
    name='crowdnav',
    version='0.0.1',
    python_requires='>=3.10,<3.11',
    packages=[
        'crowd_nav',
        'crowd_nav.cli',
        'crowd_nav.configs',
        'crowd_nav.gui',
        'crowd_nav.gui.controllers',
        'crowd_nav.gui.workers',
        'crowd_nav.planner',
        'crowd_nav.policy',
        'crowd_nav.utils',
        'crowd_sim',
        'crowd_sim.envs',
        'crowd_sim.envs.policy',
        'crowd_sim.envs.utils',
    ],
    install_requires=[
        'cython<3',
        'gitpython',
        'gym==0.15.7',
        'matplotlib',
        'numpy',
        'scipy',
        'torch',
        'torchvision',
    ],
    extras_require={
        'gui': [
            'Pillow',
            'PyQt5',
        ],
        'test': [
            'Pillow',
            'PyQt5',
            'pylint',
            'pytest',
            'pytest-qt',
        ],
    },
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_setup_py_pins.py -v`
Expected: 3 passes.

- [ ] **Step 5: Commit**

```bash
git add setup.py tests/test_setup_py_pins.py
git commit -m "feat(wp6): pin gym==0.15.7, cython<3, python_requires=>=3.10,<3.11"
```

---

## Task 1: Add `crowdnav-preflight` CLI + module

**Files:**
- Create: `crowd_nav/utils/preflight.py`
- Create: `crowd_nav/cli/__init__.py`
- Create: `crowd_nav/cli/preflight.py`
- Create: `tests/test_preflight.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preflight.py
"""Preflight verifies every runtime dep students silently miss: rvo2 built from
the vendored tree, ffmpeg on PATH, gym pinned to 0.15.7, trained model present."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from crowd_nav.utils.preflight import (  # noqa: E402 — imported after marker
    CheckResult,
    check_ffmpeg,
    check_gym_version,
    check_model,
    check_rvo2,
    run_all,
)


def test_check_rvo2_ok_when_module_importable(monkeypatch):
    import sys
    import types

    monkeypatch.setitem(sys.modules, "rvo2", types.ModuleType("rvo2"))
    result = check_rvo2()
    assert isinstance(result, CheckResult)
    assert result.ok is True


def test_check_rvo2_fail_when_import_errors(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rvo2":
            raise ImportError("not built")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = check_rvo2()
    assert result.ok is False
    assert "Python-RVO2" in result.hint


def test_check_ffmpeg_uses_shutil_which(monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
    assert check_ffmpeg().ok is True
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert check_ffmpeg().ok is False


def test_check_gym_version_exact_pin(monkeypatch):
    import importlib.metadata as md

    monkeypatch.setattr(md, "version", lambda name: "0.15.7" if name == "gym" else "0.0")
    assert check_gym_version().ok is True
    monkeypatch.setattr(md, "version", lambda name: "0.21.0" if name == "gym" else "0.0")
    assert check_gym_version().ok is False


def test_check_model_exists_under_repo_root(tmp_path, monkeypatch):
    model = tmp_path / "crowd_nav" / "data" / "output_trained" / "rl_model.pth"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"\x00")
    monkeypatch.setenv("CROWDNAV_ROOT", str(tmp_path))
    assert check_model().ok is True
    model.unlink()
    assert check_model().ok is False


def test_run_all_returns_nonzero_on_any_failure(monkeypatch):
    ok = CheckResult(True, "ok", "")
    bad = CheckResult(False, "rvo2 fail", "build it")
    monkeypatch.setattr("crowd_nav.utils.preflight.check_rvo2", lambda: bad)
    monkeypatch.setattr("crowd_nav.utils.preflight.check_ffmpeg", lambda: ok)
    monkeypatch.setattr("crowd_nav.utils.preflight.check_gym_version", lambda: ok)
    monkeypatch.setattr("crowd_nav.utils.preflight.check_model", lambda: ok)
    assert run_all() == 1


def test_run_all_returns_zero_when_all_pass(monkeypatch):
    ok = CheckResult(True, "ok", "")
    monkeypatch.setattr("crowd_nav.utils.preflight.check_rvo2", lambda: ok)
    monkeypatch.setattr("crowd_nav.utils.preflight.check_ffmpeg", lambda: ok)
    monkeypatch.setattr("crowd_nav.utils.preflight.check_gym_version", lambda: ok)
    monkeypatch.setattr("crowd_nav.utils.preflight.check_model", lambda: ok)
    assert run_all() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preflight.py -v`
Expected: `ModuleNotFoundError: No module named 'crowd_nav.utils.preflight'`.

- [ ] **Step 3: Create `crowd_nav/utils/preflight.py`**

```python
"""WP-6 preflight checks: verify runtime deps students commonly miss.

Each check is a pure function returning ``CheckResult`` so unit tests can
monkeypatch probes in isolation. ``run_all`` orchestrates them and returns
an exit code suitable for ``sys.exit``.
"""

from __future__ import annotations

import importlib
import importlib.metadata as _md
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    message: str
    hint: str


def _repo_root() -> Path:
    override = os.environ.get("CROWDNAV_ROOT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2]


def check_rvo2() -> CheckResult:
    try:
        importlib.import_module("rvo2")
    except ImportError as exc:
        return CheckResult(
            False,
            f"cannot import rvo2: {exc}",
            "build the vendored Python-RVO2: "
            "(cd Python-RVO2-main && python setup.py build && python setup.py install)",
        )
    return CheckResult(True, "rvo2 importable", "")


def check_ffmpeg() -> CheckResult:
    if shutil.which("ffmpeg"):
        return CheckResult(True, "ffmpeg on PATH", "")
    return CheckResult(
        False,
        "ffmpeg not found on PATH",
        "install ffmpeg (linux: apt install ffmpeg; macOS: brew install ffmpeg)",
    )


def check_gym_version() -> CheckResult:
    try:
        version = _md.version("gym")
    except _md.PackageNotFoundError:
        return CheckResult(False, "gym not installed", "pip install 'gym==0.15.7'")
    if version != "0.15.7":
        return CheckResult(
            False,
            f"gym=={version} installed, expected 0.15.7",
            "pip install 'gym==0.15.7'",
        )
    return CheckResult(True, "gym==0.15.7", "")


def check_model() -> CheckResult:
    path = _repo_root() / "crowd_nav" / "data" / "output_trained" / "rl_model.pth"
    if not path.exists():
        return CheckResult(
            False,
            f"trained model missing at {path}",
            "fetch rl_model.pth from the project release or re-clone with LFS enabled",
        )
    return CheckResult(True, f"model present at {path}", "")


def run_all() -> int:
    checks = {
        "rvo2": check_rvo2(),
        "ffmpeg": check_ffmpeg(),
        "gym": check_gym_version(),
        "model": check_model(),
    }
    bad = [name for name, r in checks.items() if not r.ok]
    for name, r in checks.items():
        status = "OK " if r.ok else "FAIL"
        print(f"[{status}] {name}: {r.message}")
        if not r.ok and r.hint:
            print(f"       hint: {r.hint}")
    return 0 if not bad else 1
```

- [ ] **Step 4: Create `crowd_nav/cli/__init__.py` and `crowd_nav/cli/preflight.py`**

```python
# crowd_nav/cli/__init__.py
"""Console-script entry points registered in setup.py."""
```

```python
# crowd_nav/cli/preflight.py
"""`crowdnav-preflight` — verify runtime deps and exit non-zero on first problem."""

from __future__ import annotations

import sys

from crowd_nav.utils.preflight import run_all


def main() -> int:
    rc = run_all()
    sys.exit(rc)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_preflight.py -v`
Expected: 7 passes.

- [ ] **Step 6: Commit**

```bash
git add crowd_nav/utils/preflight.py crowd_nav/cli/__init__.py crowd_nav/cli/preflight.py tests/test_preflight.py
git commit -m "feat(wp6): crowdnav-preflight checks rvo2, ffmpeg, gym pin, trained model"
```

---

## Task 2: Add `crowdnav-baseline` + `crowdnav-gui` entry points

**Files:**
- Create: `crowd_nav/cli/baseline.py`
- Create: `crowd_nav/cli/gui.py`
- Modify: `setup.py`
- Create: `tests/test_console_scripts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_console_scripts.py
"""WP-6 console scripts: after `pip install -e .` each entry point is
importable and dispatches to the expected module (shell-out for baseline,
QApplication for gui)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_crowdnav_preflight_is_importable():
    from crowd_nav.cli import preflight  # noqa: F401


def test_crowdnav_baseline_is_importable():
    from crowd_nav.cli import baseline  # noqa: F401


def test_crowdnav_gui_is_importable():
    from crowd_nav.cli import gui  # noqa: F401


def test_crowdnav_baseline_delegates_to_run_baseline_script(monkeypatch):
    from crowd_nav.cli import baseline

    called = {}

    def fake_run(cmd, **kwargs):
        called["cmd"] = cmd
        called["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "exit", lambda rc=0: None)
    baseline.main()
    assert called["cmd"][0] == "bash"
    assert called["cmd"][1].endswith("scripts/run_baseline.sh")


def test_crowdnav_gui_calls_into_crowd_nav_gui_main(monkeypatch):
    import crowd_nav.gui.__main__ as gui_main

    calls = {"n": 0}

    def _fake_main():
        calls["n"] += 1

    monkeypatch.setattr(gui_main, "main", _fake_main)

    from crowd_nav.cli import gui

    gui.main()
    assert calls["n"] == 1


def test_setup_py_declares_all_three_entry_points():
    src = (Path(__file__).resolve().parent.parent / "setup.py").read_text()
    for name, target in (
        ("crowdnav-preflight", "crowd_nav.cli.preflight:main"),
        ("crowdnav-baseline",  "crowd_nav.cli.baseline:main"),
        ("crowdnav-gui",       "crowd_nav.cli.gui:main"),
    ):
        assert f"{name} = {target}" in src or f"{name}={target}" in src, \
            f"entry_points missing '{name} = {target}'"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_console_scripts.py -v`
Expected: `ModuleNotFoundError: No module named 'crowd_nav.cli.baseline'` (and peers) + the `setup.py` entry-point assertion.

- [ ] **Step 3: Create `crowd_nav/cli/baseline.py` and `crowd_nav/cli/gui.py`**

```python
# crowd_nav/cli/baseline.py
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
```

```python
# crowd_nav/cli/gui.py
"""`crowdnav-gui` — launch the PyQt5 MainWindow without typing `python -m ...`."""

from __future__ import annotations

from crowd_nav.gui.__main__ import main as _gui_main


def main() -> None:
    _gui_main()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Register entry points in `setup.py`**

Add the `entry_points` kwarg to the `setup()` call (keep existing kwargs from Task 0):

```python
    entry_points={
        'console_scripts': [
            'crowdnav-preflight = crowd_nav.cli.preflight:main',
            'crowdnav-baseline = crowd_nav.cli.baseline:main',
            'crowdnav-gui = crowd_nav.cli.gui:main',
        ],
    },
```

- [ ] **Step 5: Reinstall so entry points are materialized**

Run: `pip install -e .`
Expected: `Successfully installed crowdnav-0.0.1`.

- [ ] **Step 6: Verify entry points exist**

Run: `command -v crowdnav-preflight && command -v crowdnav-baseline && command -v crowdnav-gui`
Expected: three paths in the active env's `bin/`.

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_console_scripts.py -v`
Expected: 6 passes.

- [ ] **Step 8: Run preflight end-to-end**

Run: `crowdnav-preflight`
Expected: `[OK ]` lines for rvo2, ffmpeg, gym, model. Exit 0 when env is complete.

- [ ] **Step 9: Commit**

```bash
git add crowd_nav/cli/baseline.py crowd_nav/cli/gui.py setup.py tests/test_console_scripts.py
git commit -m "feat(wp6): console_scripts crowdnav-{preflight,baseline,gui}"
```

---

## Task 3: WP-3 smoke test (static map loaded → rollout MP4)

**Files:**
- Create: `tests/test_smoke_wp3.py`

- [ ] **Step 1: Confirm `test.py` accepts `--static_map`**

Run: `grep -nE "\\-\\-static_map|static_map" crowd_nav/test.py`
Expected: at least one argparse line from WP-3 commit `b581827`. If the flag is named differently, adjust the test's argv list in Step 2 accordingly — **do not add a new argparse flag in this task**.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_smoke_wp3.py
"""WP-3 smoke: with a static map loaded, a headless rollout writes a non-empty
mp4. Uses a tiny empty grid so the test runs fast; the point is to prove the
static-map code path wires through test.py -> env -> render -> ffmpeg."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.slow]


def _write_tiny_map(tmp_path: Path) -> Path:
    Image = pytest.importorskip("PIL.Image")
    grid = np.zeros((30, 30), dtype=np.uint8)
    png = tmp_path / "empty.png"
    Image.fromarray(grid, mode="L").save(png)
    (tmp_path / "empty.json").write_text(
        json.dumps({"resolution": 1.0, "origin": [-15.0, -15.0]})
    )
    return png


def test_static_map_rollout_produces_video(repo_root: Path, exports_dir: Path, tmp_path: Path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")

    png = _write_tiny_map(tmp_path)
    out = exports_dir / "wp3_smoke.mp4"
    if out.exists():
        out.unlink()

    cmd = [
        "python", "test.py",
        "--policy", "sarl",
        "--model_dir", "data/output_trained",
        "--phase", "test",
        "--test_case", "0",
        "--seed", "42",
        "--visualize",
        "--static_map", str(png),
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
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest tests/test_smoke_wp3.py -v -m smoke`
Expected: PASS, `exports/wp3_smoke.mp4` > 10 KB.

- [ ] **Step 4: Commit**

```bash
git add tests/test_smoke_wp3.py
git commit -m "test(wp6): smoke for WP-3 static-map rollout"
```

---

## Task 4: WP-4 smoke test (Theta\* planner → rollout MP4)

**Files:**
- Create: `tests/test_smoke_wp4.py`

- [ ] **Step 1: Confirm `test.py` accepts `--planner`**

Run: `grep -nE "\\-\\-planner|planner" crowd_nav/test.py`
Expected: argparse line from WP-4 commit `8f5a656`. Adjust argv in Step 2 if the flag is named differently.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_smoke_wp4.py
"""WP-4 smoke: Theta* planner produces a rollout mp4 AND the stdout shows
the planner was actually consulted (not the straight-line fallback that WP-4
commit 91be229 prints a WARNING for)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.slow]


def _write_obstacle_map(tmp_path: Path) -> Path:
    Image = pytest.importorskip("PIL.Image")
    # 30x30 grid, 1m resolution, origin (-15,-15). Single 4x4-cell "island"
    # placed so a straight start->goal route through world (~-5,0)->(~5,0)
    # would clip it, forcing Theta* to route around.
    grid = np.zeros((30, 30), dtype=np.uint8)
    grid[13:17, 13:17] = 255
    png = tmp_path / "island.png"
    Image.fromarray(grid, mode="L").save(png)
    (tmp_path / "island.json").write_text(
        json.dumps({"resolution": 1.0, "origin": [-15.0, -15.0]})
    )
    return png


def test_theta_star_rollout_produces_video_and_waypoints_skirt_island(
    repo_root: Path, exports_dir: Path, tmp_path: Path
):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")

    png = _write_obstacle_map(tmp_path)
    out = exports_dir / "wp4_smoke.mp4"
    if out.exists():
        out.unlink()

    cmd = [
        "python", "test.py",
        "--policy", "sarl",
        "--model_dir", "data/output_trained",
        "--phase", "test",
        "--test_case", "0",
        "--seed", "42",
        "--visualize",
        "--static_map", str(png),
        "--planner", "theta_star",
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
    assert out.exists() and out.stat().st_size > 10_000

    combined = (result.stdout + "\n" + result.stderr).lower()
    assert "theta_star" in combined, "Theta* branch was not taken"
    assert "fallback" not in combined, "planner degraded to straight-line fallback"
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest tests/test_smoke_wp4.py -v -m smoke`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_smoke_wp4.py
git commit -m "test(wp6): smoke for WP-4 Theta* rollout"
```

---

## Task 5: WP-5 GUI smoke (promote existing end-to-end test)

**Files:**
- Modify: `tests/gui/test_end_to_end.py`
- Create: `tests/test_smoke_wp5.py`

Note: the full click-through driver already lives at `tests/gui/test_end_to_end.py`. Duplicating it here under a `smoke` marker would be double-maintained logic, so instead we extract the body into a helper that both tests call. The existing behaviour is preserved.

- [ ] **Step 1: Extract the test body into a helper**

Edit `tests/gui/test_end_to_end.py` — rename the body of `test_gui_full_voyage(qtbot, repo_root, tmp_path, monkeypatch)` into a module-level function `_run_full_gui_voyage(...)`, then make the existing test just call it:

```python
def _run_full_gui_voyage(qtbot, repo_root: Path, tmp_path: Path, monkeypatch) -> None:
    Image = pytest.importorskip("PIL.Image")

    from crowd_nav.gui.controllers.sim_controller import SimController
    from crowd_nav.gui.main_window import MainWindow

    grid = np.zeros((30, 30), dtype=np.uint8)
    png = tmp_path / "empty.png"
    Image.fromarray(grid, mode="L").save(png)
    (tmp_path / "empty.json").write_text(
        json.dumps({"resolution": 1.0, "origin": [-15.0, -15.0]})
    )

    cfg = repo_root / "crowd_nav" / "data" / "output_trained" / "env.config"
    controller = SimController(env_config_path=cfg)
    w = MainWindow(controller=controller)
    qtbot.addWidget(w)

    model_pth = repo_root / "crowd_nav" / "data" / "output_trained" / "rl_model.pth"

    def _fake_dialog(parent, caption, directory, filter_):
        if caption.startswith("Select map"):
            return (str(png), "")
        if caption.startswith("Select DRL model"):
            return (str(model_pth), "")
        return ("", "")

    monkeypatch.setattr(
        "crowd_nav.gui.main_window.QFileDialog.getOpenFileName", _fake_dialog
    )

    w.on_load_map()
    assert controller.static_map is not None
    w.on_load_model()
    assert controller.policy is not None
    w.on_set_start()
    w.canvas.clicked.emit(-3.0, -3.0)
    assert controller.start == (-3.0, -3.0)
    w.on_set_goal()
    w.canvas.clicked.emit(3.0, 3.0)
    assert controller.goal == (3.0, 3.0)

    w._last_states = controller.run_episode()
    assert len(w._last_states) > 0

    monkeypatch.setattr(
        "crowd_nav.gui.main_window.QFileDialog.getExistingDirectory",
        lambda *a, **kw: str(tmp_path),
    )
    w.on_export()

    assert (tmp_path / "run.csv").exists()
    assert (tmp_path / "trajectory.png").exists()


def test_gui_full_voyage(qtbot, repo_root: Path, tmp_path: Path, monkeypatch) -> None:
    _run_full_gui_voyage(qtbot, repo_root, tmp_path, monkeypatch)
```

- [ ] **Step 2: Confirm the refactor is behaviour-preserving**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/gui/test_end_to_end.py -v`
Expected: PASS (same assertions, same path).

- [ ] **Step 3: Write the failing smoke test**

```python
# tests/test_smoke_wp5.py
"""WP-5 GUI smoke: the full click-through (load map, load model, set start/goal,
visualize, export) succeeds under offscreen Qt. Re-uses the e2e driver so the
GUI stays single-sourced."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.gui, pytest.mark.slow]


def test_gui_full_voyage_smoke(qtbot, repo_root: Path, tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("PyQt5")
    from tests.gui.test_end_to_end import _run_full_gui_voyage

    _run_full_gui_voyage(qtbot, repo_root, tmp_path, monkeypatch)
```

- [ ] **Step 4: Run smoke test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_smoke_wp5.py -v -m smoke`
Expected: PASS — `run.csv` + `trajectory.png` written to tmp_path.

- [ ] **Step 5: Commit**

```bash
git add tests/gui/test_end_to_end.py tests/test_smoke_wp5.py
git commit -m "test(wp6): smoke for WP-5 GUI end-to-end voyage"
```

---

## Task 6: Update `.github/workflows/smoke.yml` to run every smoke test

**Files:**
- Create: `tests/test_ci_workflow.py`
- Modify: `.github/workflows/smoke.yml`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ci_workflow.py
"""WP-6: CI installs PyQt5 and runs `pytest -m smoke` under offscreen Qt."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

WF = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "smoke.yml"


@pytest.fixture(scope="module")
def workflow() -> str:
    return WF.read_text()


def test_ci_installs_pyqt5_and_pytest_qt(workflow: str):
    assert "PyQt5" in workflow or "pyqt5" in workflow.lower(), "CI must install PyQt5"
    assert "pytest-qt" in workflow, "CI must install pytest-qt"


def test_ci_runs_smoke_marker(workflow: str):
    assert '-m "smoke"' in workflow or "-m smoke" in workflow


def test_ci_sets_offscreen_qt_for_smoke(workflow: str):
    assert "QT_QPA_PLATFORM" in workflow and "offscreen" in workflow


def test_ci_runs_preflight(workflow: str):
    assert "crowdnav-preflight" in workflow
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ci_workflow.py -v`
Expected: 4 failures.

- [ ] **Step 3: Edit `.github/workflows/smoke.yml`**

Replace the three test-related steps (Install, Fast tests, R1 baseline smoke, Upload) with:

```yaml
      - name: Install Python deps
        env:
          CMAKE_POLICY_VERSION_MINIMUM: "3.5"
        run: |
          python -m pip install --upgrade pip
          pip install "cython<3"
          (cd Python-RVO2-main && python setup.py build && python setup.py install)
          pip install -e ".[test]"
          pip install "gym==0.15.7"

      - name: Preflight
        run: crowdnav-preflight

      - name: Fast tests
        run: pytest -v -m "not slow and not smoke"

      - name: All smoke tests (R1 + WP-3 + WP-4 + WP-5 GUI, headless)
        env:
          MPLBACKEND: Agg
          QT_QPA_PLATFORM: offscreen
        run: pytest -v -m "smoke"

      - name: Upload smoke artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: smoke-outputs
          path: |
            exports/baseline.mp4
            exports/wp3_smoke.mp4
            exports/wp4_smoke.mp4
          if-no-files-found: warn
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ci_workflow.py -v`
Expected: 4 passes.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/smoke.yml tests/test_ci_workflow.py
git commit -m "ci(wp6): smoke.yml runs preflight + pytest -m smoke with PyQt5 offscreen"
```

---

## Task 7: Wire preflight into `scripts/setup_env.sh` + README polish

**Files:**
- Modify: `scripts/setup_env.sh`
- Modify: `README.md`

- [ ] **Step 1: Append preflight to `setup_env.sh`**

Before the final `log "done. Activate with: ..."` line, insert:

```bash
log "running preflight checks"
if ! crowdnav-preflight; then
  echo "preflight failed — see hints above" >&2
  exit 1
fi
```

- [ ] **Step 2: Add a console-scripts block to README "Getting Started"**

Insert immediately after README.md line 68 (`### Visualize a test case`):

```markdown
### One-line commands

```bash
crowdnav-preflight        # verify rvo2 / ffmpeg / gym / trained model
crowdnav-gui              # launch the PyQt GUI (slide-14 demo)
crowdnav-baseline         # reproduce R1 -> exports/baseline.mp4
```

Advanced flag-driven usage with `python test.py ...` is documented below.
```

Leave the existing `python test.py ...` example and its flag table untouched.

- [ ] **Step 3: Dry-run the setup script**

Run: `bash scripts/setup_env.sh`
Expected: idempotent reuse of conda env; preflight prints 4 `[OK ]` lines; exit 0.

- [ ] **Step 4: Commit**

```bash
git add scripts/setup_env.sh README.md
git commit -m "chore(wp6): setup_env.sh runs preflight; README advertises console scripts"
```

---

## Task 8: Flip WP-6 status in `docs/REQUIREMENTS.md`

**Files:**
- Modify: `docs/REQUIREMENTS.md`

- [ ] **Step 1: Edit the WP-6 row (line 26)**

Change the cell for WP-6 from:
```
| **WP-6** Packaging + smoke tests | cross-cutting | WP-1…5 | Group C |
```
to:
```
| **WP-6** Packaging + smoke tests | cross-cutting (Implemented 2026-04-18) | WP-1…5 | Group C |
```

- [ ] **Step 2: Run the full unit suite**

Run: `pytest -m "not slow and not smoke" -q`
Expected: all PASS.

- [ ] **Step 3: Run the full smoke suite locally**

Run: `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg pytest -m smoke -v`
Expected: 4 passes — baseline, wp3, wp4, wp5.

- [ ] **Step 4: Commit**

```bash
git add docs/REQUIREMENTS.md
git commit -m "docs(wp6): flip WP-6 status to Implemented 2026-04-18"
```

---

## Verification (end-to-end)

After all tasks are merged, an operator should be able to run:

1. **Fresh-env install (simulate a new student):**
   ```bash
   conda deactivate
   conda env remove -n navigate-wp6 -y || true
   ENV_NAME=navigate-wp6 ./scripts/setup_env.sh
   conda activate navigate-wp6
   crowdnav-preflight
   ```
   Expected: 4 `[OK ]` lines; exit 0.

2. **One-command demo:**
   ```bash
   crowdnav-baseline
   ls -lh exports/baseline.mp4
   ```
   Expected: MP4 > 10 KB, exit 0.

3. **Headless GUI smoke:**
   ```bash
   QT_QPA_PLATFORM=offscreen pytest tests/test_smoke_wp5.py -v -m smoke
   ```
   Expected: PASS.

4. **Full smoke suite locally:**
   ```bash
   QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg pytest -m smoke -v
   ```
   Expected: 4 tests (baseline, wp3, wp4, wp5) all PASS; artifacts under `exports/`.

5. **CI green on push:** open a PR containing these commits, confirm the `smoke` workflow goes green on GitHub Actions with all four artifacts uploaded under `smoke-outputs`.

If any of steps 1–4 fails locally, fix before merging; CI (step 5) is a belt-and-braces check, not a substitute.

---

## Self-Review Notes

- **Spec coverage:** WP-6 description in `docs/REQUIREMENTS.md:26` says "Packaging + smoke tests, cross-cutting, depends on WP-1…5". Tasks 0–2 cover packaging (pins + entry points + preflight); Tasks 3–5 cover per-WP smoke tests; Task 6 gates them in CI; Tasks 7–8 polish docs and status.
- **Reuse verified:** `scripts/run_baseline.sh`, `python -m crowd_nav.gui`, `tests/gui/test_end_to_end.py`, `tests/conftest.py` fixtures, and the existing `smoke`/`gui` pytest markers are all reused without modification (except the single extract-body refactor in Task 5, which is behaviour-preserving and gated by the existing test still passing).
- **Type/signature consistency:** `CheckResult` used identically across `check_rvo2`, `check_ffmpeg`, `check_gym_version`, `check_model`, `run_all`. Console-script `main()` functions are consistent — `preflight` and `baseline` call `sys.exit(rc)`, `gui` delegates to `crowd_nav.gui.__main__.main()` which already exits via Qt's `app.exec_()`.
- **Non-goals:** no lock file, no wheel/sdist publish pipeline, no Windows support, no Python-RVO2 rewrite, no `INSTALL.md` split (README edits are minimal and additive). These are future work if the student group grows.
