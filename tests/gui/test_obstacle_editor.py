"""ObstacleEditor: translate mouse drags into obstacle dicts."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gui


def test_drag_rect_emits_rect_obstacle(qtbot) -> None:
    from crowd_nav.gui.obstacle_editor import ObstacleEditor

    editor = ObstacleEditor(mode="rect")
    captured: list[dict] = []
    editor.obstacle_created.connect(captured.append)

    editor.on_press((0.0, 0.0))
    editor.on_release((2.0, 4.0))

    assert len(captured) == 1
    obs = captured[0]
    assert obs["type"] == "rect"
    assert obs["cx"] == pytest.approx(1.0)
    assert obs["cy"] == pytest.approx(2.0)
    assert obs["w"] == pytest.approx(2.0)
    assert obs["h"] == pytest.approx(4.0)


def test_drag_circle_emits_circle_obstacle(qtbot) -> None:
    from crowd_nav.gui.obstacle_editor import ObstacleEditor

    editor = ObstacleEditor(mode="circle")
    captured: list[dict] = []
    editor.obstacle_created.connect(captured.append)

    editor.on_press((0.0, 0.0))
    editor.on_release((3.0, 4.0))

    assert len(captured) == 1
    obs = captured[0]
    assert obs["type"] == "circle"
    assert obs["cx"] == pytest.approx(0.0)
    assert obs["cy"] == pytest.approx(0.0)
    assert obs["r"] == pytest.approx(5.0)  # hypot(3, 4)


def test_release_without_press_is_noop(qtbot) -> None:
    from crowd_nav.gui.obstacle_editor import ObstacleEditor

    editor = ObstacleEditor(mode="rect")
    captured: list[dict] = []
    editor.obstacle_created.connect(captured.append)

    editor.on_release((1.0, 1.0))
    assert captured == []
