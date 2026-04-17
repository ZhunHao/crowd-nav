"""Entry point: python -m crowd_nav.gui constructs MainWindow without error."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.gui


def test_entry_constructs_mainwindow(qtbot, repo_root: Path, monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv(
        "CROWDNAV_ENV_CONFIG",
        str(repo_root / "crowd_nav" / "configs" / "env.config"),
    )
    from crowd_nav.gui.__main__ import build_window

    w = build_window()
    qtbot.addWidget(w)
    assert w.windowTitle().startswith("CrowdNav")
    assert len(w.toolbar.actions()) == 7
