"""Decode land maps into StaticMap obstacles (WP-5).

Supports two on-disk formats:

* ``.png`` (+ ``.json`` sidecar): 8-bit greyscale, non-zero = land, zero = water.
* ``.npy`` (+ ``.json`` sidecar): any integer/bool array, truthy = land.

Sidecar schema::

    {"resolution": <metres per cell>, "origin": [xmin, ymin]}

The loader rasterises blocked cells into 1x1 rect obstacles anchored at their
cell centres; downstream ``StaticMap.is_free`` already handles rect obstacles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np

from crowd_sim.envs.utils.static_map import StaticMap

PathLike = Union[str, Path]


def load_static_map(path: PathLike, margin: float = 0.0) -> StaticMap:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"map file not found: {p}")

    if p.suffix.lower() == ".png":
        grid = _decode_png(p)
    elif p.suffix.lower() == ".npy":
        grid = np.load(p, allow_pickle=False)
    else:
        raise ValueError(
            f"unsupported map format {p.suffix!r}; expected .png or .npy"
        )

    meta_path = p.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"map sidecar {meta_path.name!r} missing next to {p.name!r}"
        )
    meta = json.loads(meta_path.read_text())
    if "resolution" not in meta or "origin" not in meta:
        raise ValueError(
            f"map sidecar {meta_path.name!r} must define 'resolution' and 'origin'"
        )
    origin = meta["origin"]
    if not (isinstance(origin, (list, tuple)) and len(origin) == 2):
        raise ValueError(
            f"'origin' in {meta_path.name!r} must be a 2-element list, got {origin!r}"
        )
    resolution = float(meta["resolution"])
    ox, oy = float(origin[0]), float(origin[1])

    if grid.ndim != 2:
        raise ValueError(
            f"map array in {p.name!r} must be 2-D, got shape {grid.shape}"
        )
    obs: list[dict] = []
    blocked = np.transpose(np.nonzero(grid))
    for i, j in blocked:
        cx = ox + (j + 0.5) * resolution
        cy = oy + (i + 0.5) * resolution
        obs.append(
            {"type": "rect", "cx": cx, "cy": cy, "w": resolution, "h": resolution}
        )
    return StaticMap.from_static_obstacles(obs, margin=margin)


def _decode_png(path: Path) -> np.ndarray:
    from PIL import Image  # imported here so test skip on missing Pillow is clean

    with Image.open(path) as im:
        arr = np.array(im.convert("L"), dtype=np.uint8)
    return arr
