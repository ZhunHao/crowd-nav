"""SimCanvas behaviour - click to world coord conversion."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gui


def test_canvas_click_emits_world_coordinates(qtbot) -> None:
    from crowd_nav.gui.canvas import SimCanvas

    canvas = SimCanvas(world_bounds=(-15.0, 15.0, -15.0, 15.0))
    qtbot.addWidget(canvas)

    captured: list[tuple[float, float]] = []
    canvas.clicked.connect(lambda x, y: captured.append((x, y)))

    # Fire a synthetic pick by calling the handler the canvas binds.
    canvas._on_button_press(_fake_event(xdata=0.0, ydata=0.0))
    canvas._on_button_press(_fake_event(xdata=7.5, ydata=-3.0))

    assert captured == [(0.0, 0.0), (7.5, -3.0)]


def test_canvas_ignores_clicks_outside_axes(qtbot) -> None:
    from crowd_nav.gui.canvas import SimCanvas

    canvas = SimCanvas(world_bounds=(-15.0, 15.0, -15.0, 15.0))
    qtbot.addWidget(canvas)

    captured: list[tuple[float, float]] = []
    canvas.clicked.connect(lambda x, y: captured.append((x, y)))

    canvas._on_button_press(_fake_event(xdata=None, ydata=None))

    assert captured == []


def test_canvas_click_emits_real_pyqtsignal(qtbot) -> None:
    """Exercise the actual Qt signal plumbing, not just the Python callback."""
    from crowd_nav.gui.canvas import SimCanvas

    canvas = SimCanvas(world_bounds=(-15.0, 15.0, -15.0, 15.0))
    qtbot.addWidget(canvas)

    with qtbot.waitSignal(canvas.clicked, timeout=500) as blocker:
        canvas._on_button_press(_fake_event(xdata=1.5, ydata=-2.5))
    assert blocker.args == [1.5, -2.5]


def test_canvas_ignores_non_left_button(qtbot) -> None:
    from crowd_nav.gui.canvas import SimCanvas

    canvas = SimCanvas(world_bounds=(-15.0, 15.0, -15.0, 15.0))
    qtbot.addWidget(canvas)

    captured: list[tuple[float, float]] = []
    canvas.clicked.connect(lambda x, y: captured.append((x, y)))

    event = _fake_event(xdata=1.0, ydata=1.0)
    event.button = 3  # right-click
    canvas._on_button_press(event)

    assert captured == []


class _fake_event:
    def __init__(self, xdata, ydata):
        self.xdata = xdata
        self.ydata = ydata
        self.button = 1
        self.inaxes = object() if xdata is not None else None
