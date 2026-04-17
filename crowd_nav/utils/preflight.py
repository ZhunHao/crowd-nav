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
