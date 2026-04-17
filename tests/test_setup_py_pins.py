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
