"""Unit tests for SimController (WP-5).

Pure Python - no QApplication needed. These tests verify the headless
facade that MainWindow later drives.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_controller_initialises_with_env_config(repo_root: Path) -> None:
    from crowd_nav.gui.controllers.sim_controller import SimController

    cfg = repo_root / "crowd_nav" / "configs" / "env.config"
    sc = SimController(env_config_path=cfg)
    assert sc.env_config_path == cfg
    assert sc.static_map is None
    assert sc.policy is None
    assert sc.start is None
    assert sc.goal is None


@pytest.mark.unit
def test_controller_load_map(repo_root: Path, tmp_path: Path) -> None:
    import json

    import numpy as np
    Image = pytest.importorskip("PIL.Image")

    from crowd_nav.gui.controllers.sim_controller import SimController

    grid = np.zeros((4, 4), dtype=np.uint8)
    grid[1:3, 1:3] = 255
    png = tmp_path / "blob.png"
    Image.fromarray(grid, mode="L").save(png)
    (tmp_path / "blob.json").write_text(json.dumps({"resolution": 1.0, "origin": [-2.0, -2.0]}))

    sc = SimController(env_config_path=repo_root / "crowd_nav" / "configs" / "env.config")
    sc.load_map(png)
    assert sc.static_map is not None
    # blob covers world roughly -1..1
    assert sc.static_map.is_free(0.0, 0.0) is False
    assert sc.static_map.is_free(-1.8, -1.8) is True


@pytest.mark.unit
def test_controller_set_start_rejects_blocked_point(
    repo_root: Path, tmp_path: Path
) -> None:
    import json

    import numpy as np
    Image = pytest.importorskip("PIL.Image")

    from crowd_nav.gui.controllers.sim_controller import SimController

    grid = np.full((2, 2), 255, dtype=np.uint8)
    png = tmp_path / "all_land.png"
    Image.fromarray(grid, mode="L").save(png)
    (tmp_path / "all_land.json").write_text(
        json.dumps({"resolution": 1.0, "origin": [-1.0, -1.0]})
    )
    sc = SimController(env_config_path=repo_root / "crowd_nav" / "configs" / "env.config")
    sc.load_map(png)
    with pytest.raises(ValueError, match="start .* inside obstacle"):
        sc.set_start((0.0, 0.0))


@pytest.mark.unit
def test_controller_add_obstacle_does_not_double_count(
    repo_root: Path, tmp_path: Path
) -> None:
    """Regression: add_obstacle used to duplicate prior user shapes into
    static_map.obstacles on every call because it rebuilt from the already-
    baked static_map rather than from the pristine base map.
    """
    import json

    import numpy as np
    Image = pytest.importorskip("PIL.Image")

    from crowd_nav.gui.controllers.sim_controller import SimController

    # Empty 4x4 map so the base has zero obstacles.
    grid = np.zeros((4, 4), dtype=np.uint8)
    png = tmp_path / "empty.png"
    Image.fromarray(grid, mode="L").save(png)
    (tmp_path / "empty.json").write_text(
        json.dumps({"resolution": 1.0, "origin": [-2.0, -2.0]})
    )

    sc = SimController(
        env_config_path=repo_root / "crowd_nav" / "configs" / "env.config"
    )
    sc.load_map(png)
    assert len(sc.static_map.obstacles) == 0

    sc.add_obstacle({"type": "rect", "cx": 0.0, "cy": 0.0, "w": 1.0, "h": 1.0})
    sc.add_obstacle({"type": "rect", "cx": 1.0, "cy": 1.0, "w": 1.0, "h": 1.0})
    sc.add_obstacle({"type": "circle", "cx": -1.0, "cy": -1.0, "r": 0.5})

    # Three distinct user shapes, zero base obstacles -> exactly three total.
    assert len(sc.static_map.obstacles) == 3


@pytest.mark.unit
def test_controller_rejects_missing_env_config(tmp_path: Path) -> None:
    from crowd_nav.gui.controllers.sim_controller import SimController

    bogus = tmp_path / "does_not_exist.config"
    with pytest.raises(FileNotFoundError, match="env config not found"):
        SimController(env_config_path=bogus)
