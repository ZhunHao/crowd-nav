"""GUI test fixtures - force offscreen Qt platform so CI and headless devs can run.

pytest-qt supplies the ``qtbot`` fixture; we just set ``QT_QPA_PLATFORM=offscreen``
before the QApplication spins up so no X/Wayland/Cocoa is required.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session", autouse=True)
def _qt_offscreen_platform() -> None:
    # Override any shell-set QT_QPA_PLATFORM (e.g. 'xcb' on a Linux dev
    # workstation) so GUI tests always run headless. The module-level
    # setdefault above only handles the unset case.
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
