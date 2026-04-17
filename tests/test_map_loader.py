"""Unit tests for map_loader (WP-5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.unit
def test_load_png_map_with_json_sidecar(tmp_path: Path) -> None:
    from crowd_sim.envs.utils.map_loader import load_static_map

    Image = pytest.importorskip("PIL.Image")
    # 4x4 bitmap: 255 = land, 0 = water. Ring of land with water centre.
    grid = np.array(
        [
            [255, 255, 255, 255],
            [255, 0, 0, 255],
            [255, 0, 0, 255],
            [255, 255, 255, 255],
        ],
        dtype=np.uint8,
    )
    png = tmp_path / "ring.png"
    Image.fromarray(grid, mode="L").save(png)
    (tmp_path / "ring.json").write_text(
        json.dumps({"resolution": 1.0, "origin": [-2.0, -2.0]})
    )

    sm = load_static_map(png)
    # (0, 0) is water (centre) -> free.
    assert sm.is_free(0.0, 0.0) is True
    # (-2, -2) is land (corner) -> blocked.
    assert sm.is_free(-1.5, -1.5) is False


@pytest.mark.unit
def test_load_npy_map(tmp_path: Path) -> None:
    from crowd_sim.envs.utils.map_loader import load_static_map

    grid = np.zeros((3, 3), dtype=np.uint8)
    grid[1, 1] = 1  # single blocked cell at centre
    npy = tmp_path / "point.npy"
    np.save(npy, grid)
    meta = tmp_path / "point.json"
    meta.write_text(json.dumps({"resolution": 0.5, "origin": [-0.75, -0.75]}))

    sm = load_static_map(npy)
    # Centre cell covers world [-0.25, 0.25] x [-0.25, 0.25].
    assert sm.is_free(0.0, 0.0) is False
    assert sm.is_free(-0.6, -0.6) is True


@pytest.mark.unit
def test_load_static_map_raises_for_unknown_suffix(tmp_path: Path) -> None:
    from crowd_sim.envs.utils.map_loader import load_static_map

    bogus = tmp_path / "x.bmp"
    bogus.write_bytes(b"not a real map")
    with pytest.raises(ValueError, match="unsupported map format"):
        load_static_map(bogus)


@pytest.mark.unit
def test_load_static_map_raises_when_sidecar_missing(tmp_path: Path) -> None:
    from crowd_sim.envs.utils.map_loader import load_static_map

    Image = pytest.importorskip("PIL.Image")
    png = tmp_path / "lonely.png"
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8), mode="L").save(png)
    with pytest.raises(FileNotFoundError, match="sidecar .* missing"):
        load_static_map(png)
