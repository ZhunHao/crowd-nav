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
    import sys
    _tests_dir = str(Path(__file__).parent)
    if _tests_dir not in sys.path:
        sys.path.insert(0, _tests_dir)
    from gui.test_end_to_end import _run_full_gui_voyage

    _run_full_gui_voyage(qtbot, repo_root, tmp_path, monkeypatch)
