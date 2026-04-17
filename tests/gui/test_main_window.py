"""MainWindow: toolbar has 7 buttons, each wired to the controller."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.gui


def test_toolbar_exposes_seven_named_actions(qtbot) -> None:
    from crowd_nav.gui.main_window import MainWindow

    controller = MagicMock()
    w = MainWindow(controller=controller)
    qtbot.addWidget(w)

    names = [a.text() for a in w.toolbar.actions() if a.text()]
    assert names == [
        "Load static map",
        "Load DRL model",
        "Set start",
        "Set goal",
        "Add static obstacles",
        "Visualize",
        "Export",
    ]


def test_load_map_action_invokes_controller(qtbot, tmp_path, monkeypatch) -> None:
    from crowd_nav.gui.main_window import MainWindow

    controller = MagicMock()
    controller.static_map = None
    w = MainWindow(controller=controller)
    qtbot.addWidget(w)

    fake_png = tmp_path / "m.png"
    fake_png.touch()
    monkeypatch.setattr(
        "crowd_nav.gui.main_window.QFileDialog.getOpenFileName",
        lambda *a, **kw: (str(fake_png), ""),
    )
    w.on_load_map()
    controller.load_map.assert_called_once_with(fake_png)


def test_set_start_arms_canvas_next_click(qtbot) -> None:
    from crowd_nav.gui.main_window import MainWindow

    controller = MagicMock()
    w = MainWindow(controller=controller)
    qtbot.addWidget(w)

    w.on_set_start()
    w.canvas.clicked.emit(1.5, -2.5)
    controller.set_start.assert_called_once_with((1.5, -2.5))


def test_set_start_shows_error_when_controller_raises(qtbot, monkeypatch) -> None:
    from crowd_nav.gui.main_window import MainWindow
    import crowd_nav.gui.main_window as mw_mod

    controller = MagicMock()
    controller.set_start.side_effect = ValueError("inside obstacle")
    errors: list[str] = []
    monkeypatch.setattr(mw_mod, "show_error", lambda parent, msg: errors.append(msg))
    w = MainWindow(controller=controller)
    qtbot.addWidget(w)
    w.on_set_start()
    w.canvas.clicked.emit(1.5, -2.5)
    assert errors == ["start: inside obstacle"]


def test_add_obstacles_click_sequence_emits_exactly_one_obstacle(qtbot) -> None:
    """Regression: two consecutive clicks after Add-obstacles should create
    exactly one rect (press then release), not zero or two."""
    from crowd_nav.gui.main_window import MainWindow

    controller = MagicMock()
    w = MainWindow(controller=controller)
    qtbot.addWidget(w)

    w.on_add_obstacles()
    w.canvas.clicked.emit(0.0, 0.0)      # press
    w.canvas.clicked.emit(2.0, 4.0)      # release
    assert controller.add_obstacle.call_count == 1
    obs = controller.add_obstacle.call_args.args[0]
    assert obs["type"] == "rect"
    assert obs["w"] == 2.0
    assert obs["h"] == 4.0


def test_add_obstacles_button_twice_does_not_double_emit(qtbot) -> None:
    """Regression: pressing Add-obstacles twice must not leak signal connections."""
    from crowd_nav.gui.main_window import MainWindow

    controller = MagicMock()
    w = MainWindow(controller=controller)
    qtbot.addWidget(w)

    w.on_add_obstacles()
    w.on_add_obstacles()
    w.canvas.clicked.emit(0.0, 0.0)
    w.canvas.clicked.emit(1.0, 1.0)
    assert controller.add_obstacle.call_count == 1
