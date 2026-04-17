"""Goal allocator — sample non-overlapping waypoints + per-phase human positions.

WP-2 / R2. Pure sampling math; no I/O, no gym, no matplotlib. Consumes primitives
so ``crowd_sim.py`` and ``test.py`` can wire it in without circular imports. Seams:

* ``is_free``: defaults to always-True; WP-3 will inject ``StaticMap.is_free``.
* ``waypoint_source``: defaults to straight-line interpolation; WP-4 will inject
  ``ThetaStar.plan`` without rewriting this module.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence

Point = tuple[float, float]
Bounds = tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)
IsFreeFn = Callable[[float, float], bool]


def _always_free(_x: float, _y: float) -> bool:
    return True


@dataclass(frozen=True)
class GoalAllocator:
    """Stateless allocator for waypoints and dynamic-obstacle start/goal pairs.

    ``max_tries`` caps rejection sampling per call; raises ``RuntimeError`` on
    exhaustion so callers notice infeasible configs instead of looping silently.
    """

    max_tries: int = 500

    def sample_unused_position(
        self,
        used_regions: Sequence[Point],
        bounds: Bounds,
        min_dist: float,
        is_free: IsFreeFn | None = None,
    ) -> Point:
        """Return a point in ``bounds`` at least ``min_dist`` from every used region
        and satisfying ``is_free(x, y)``.

        Uses ``random.uniform`` so WP-1's ``seed_everything`` controls determinism.
        """
        if min_dist < 0:
            raise ValueError(f"min_dist must be >= 0, got {min_dist}")
        xmin, xmax, ymin, ymax = bounds
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(f"degenerate bounds: {bounds}")
        free = is_free if is_free is not None else _always_free

        for _ in range(self.max_tries):
            x = random.uniform(xmin, xmax)
            y = random.uniform(ymin, ymax)
            if not free(x, y):
                continue
            if all(math.hypot(x - ux, y - uy) >= min_dist for ux, uy in used_regions):
                return (x, y)
        raise RuntimeError(
            f"could not sample free position in {self.max_tries} tries "
            f"(bounds={bounds}, min_dist={min_dist}, "
            f"n_used={len(used_regions)})"
        )
