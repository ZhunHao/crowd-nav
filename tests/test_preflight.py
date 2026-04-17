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
    from crowd_nav.utils import preflight

    def fake_import_module(name):
        if name == "rvo2":
            raise ImportError("not built")
        raise RuntimeError(f"unexpected import: {name}")

    monkeypatch.setattr(preflight.importlib, "import_module", fake_import_module)
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
