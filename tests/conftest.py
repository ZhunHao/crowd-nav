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
