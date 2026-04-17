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
