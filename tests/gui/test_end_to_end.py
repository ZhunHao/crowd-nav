"""Full click-through: load map, load model, set start/goal, visualize, export.

Calls ``SimController.run_episode`` synchronously rather than through
``SimWorker`` — the QThread signal path is unit-tested separately in
``test_sim_worker.py`` and combining both in a headless offscreen Qt run
produces deterministic hangs on macOS. This keeps the GUI wiring coverage
(handlers → controller → rollout → export artifacts) without re-exercising
the thread layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.gui, pytest.mark.integration, pytest.mark.slow]


def _run_full_gui_voyage(qtbot, repo_root: Path, tmp_path: Path, monkeypatch) -> None:
    Image = pytest.importorskip("PIL.Image")

    from crowd_nav.gui.controllers.sim_controller import SimController
    from crowd_nav.gui.main_window import MainWindow

    # Tiny empty map so rollout stays fast.
    grid = np.zeros((30, 30), dtype=np.uint8)
    png = tmp_path / "empty.png"
    Image.fromarray(grid, mode="L").save(png)
    (tmp_path / "empty.json").write_text(
        json.dumps({"resolution": 1.0, "origin": [-15.0, -15.0]})
    )

    cfg = repo_root / "crowd_nav" / "data" / "output_trained" / "env.config"
    controller = SimController(env_config_path=cfg)
    w = MainWindow(controller=controller)
    qtbot.addWidget(w)

    model_pth = repo_root / "crowd_nav" / "data" / "output_trained" / "rl_model.pth"

    def _fake_dialog(parent, caption, directory, filter_):
        if caption.startswith("Select map"):
            return (str(png), "")
        if caption.startswith("Select DRL model"):
            return (str(model_pth), "")
        return ("", "")

    monkeypatch.setattr(
        "crowd_nav.gui.main_window.QFileDialog.getOpenFileName", _fake_dialog
    )

    # Drive the first four buttons via MainWindow handlers.
    w.on_load_map()
    assert controller.static_map is not None
    w.on_load_model()
    assert controller.policy is not None
    w.on_set_start()
    w.canvas.clicked.emit(-3.0, -3.0)
    assert controller.start == (-3.0, -3.0)
    w.on_set_goal()
    w.canvas.clicked.emit(3.0, 3.0)
    assert controller.goal == (3.0, 3.0)

    # Run rollout synchronously; SimWorker's thread plumbing is unit-tested.
    w._last_states = controller.run_episode()
    assert len(w._last_states) > 0

    monkeypatch.setattr(
        "crowd_nav.gui.main_window.QFileDialog.getExistingDirectory",
        lambda *a, **kw: str(tmp_path),
    )
    w.on_export()

    assert (tmp_path / "run.csv").exists()
    assert (tmp_path / "trajectory.png").exists()


def test_gui_full_voyage(qtbot, repo_root: Path, tmp_path: Path, monkeypatch) -> None:
    _run_full_gui_voyage(qtbot, repo_root, tmp_path, monkeypatch)
