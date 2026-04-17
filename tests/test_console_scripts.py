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
