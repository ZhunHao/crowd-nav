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


WaypointSourceFn = Callable[[Point, Point, int], Sequence[Point]]


def _straight_line_source(start: Point, goal: Point, n: int) -> list[Point]:
    """Interpolate ``n`` points between ``start`` and ``goal`` (excluding start).

    The last point is always exactly ``goal`` so :meth:`allocate_waypoints`
    terminates on the global goal. WP-4 will replace this with Theta*.
    """
    if n <= 0:
        raise ValueError(f"num_waypoints must be >= 1, got {n}")
    return [
        (
            start[0] + (goal[0] - start[0]) * (i / n),
            start[1] + (goal[1] - start[1]) * (i / n),
        )
        for i in range(1, n + 1)
    ]


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

    def allocate_waypoints(
        self,
        start: Point,
        goal: Point,
        num_waypoints: int,
        min_inter_dist: float,
        is_free: IsFreeFn | None = None,
        waypoint_source: WaypointSourceFn | None = None,
    ) -> list[Point]:
        """Produce ``num_waypoints`` non-overlapping waypoints from ``start`` to ``goal``.

        The last element of the returned list is always ``goal``. Intermediate
        waypoints come from ``waypoint_source`` (straight-line interpolation by
        default); any that collide with an earlier waypoint (< ``min_inter_dist``)
        or fail ``is_free`` are re-sampled via :meth:`sample_unused_position`
        within a perturbation bubble around the original point.
        """
        if num_waypoints <= 0:
            raise ValueError(f"num_waypoints must be >= 1, got {num_waypoints}")

        source = waypoint_source if waypoint_source is not None else _straight_line_source
        raw = list(source(start, goal, num_waypoints))
        if len(raw) != num_waypoints:
            raise ValueError(
                f"waypoint_source returned {len(raw)} points, expected {num_waypoints}"
            )

        # Domain bounds derived from start/goal with a 2 m slack — enough for
        # perturbation but keeps the allocator inside the env's -15..15 box.
        pad = 2.0
        bounds: Bounds = (
            min(start[0], goal[0]) - pad,
            max(start[0], goal[0]) + pad,
            min(start[1], goal[1]) - pad,
            max(start[1], goal[1]) + pad,
        )

        accepted: list[Point] = []
        used: list[Point] = [start]
        for idx, candidate in enumerate(raw):
            is_last = idx == num_waypoints - 1
            # Always snap the final waypoint to the global goal — callers rely on it.
            if is_last:
                accepted.append(goal)
                used.append(goal)
                continue

            cand = candidate
            if (
                all(math.hypot(cand[0] - u[0], cand[1] - u[1]) >= min_inter_dist for u in used)
                and (is_free is None or is_free(*cand))
            ):
                accepted.append(cand)
                used.append(cand)
                continue

            # Re-sample inside a small bubble centred on the original candidate.
            bubble: Bounds = (
                max(cand[0] - 1.5, bounds[0]),
                min(cand[0] + 1.5, bounds[1]),
                max(cand[1] - 1.5, bounds[2]),
                min(cand[1] + 1.5, bounds[3]),
            )
            replacement = self.sample_unused_position(
                used_regions=used,
                bounds=bubble,
                min_dist=min_inter_dist,
                is_free=is_free,
            )
            accepted.append(replacement)
            used.append(replacement)

        return accepted

    def allocate_human_positions(
        self,
        robot_start: Point,
        robot_goal: Point,
        occupied: Sequence[Point],
        human_num: int,
        min_dist: float,
        is_free: IsFreeFn | None = None,
    ) -> list[tuple[Point, Point]]:
        """Return ``human_num`` (start, goal) pairs for dynamic obstacles.

        Starts are sampled non-overlapping with ``occupied`` (prior humans + robot
        positions). Goals are placed on the opposite side of the midpoint so the
        human crosses the robot's corridor — preserves the "circle_crossing"
        intent from :meth:`CrowdSim.generate_random_human_position` while
        eliminating the ``radius / 1`` bug and the overlap artefact.
        """
        if human_num < 0:
            raise ValueError(f"human_num must be >= 0, got {human_num}")

        mid = (
            (robot_start[0] + robot_goal[0]) / 2.0,
            (robot_start[1] + robot_goal[1]) / 2.0,
        )
        # Axis-aligned bounds around the robot's corridor + 2 m pad so humans
        # don't spawn miles away.
        pad = 2.0
        bounds: Bounds = (
            min(robot_start[0], robot_goal[0]) - pad,
            max(robot_start[0], robot_goal[0]) + pad,
            min(robot_start[1], robot_goal[1]) - pad,
            max(robot_start[1], robot_goal[1]) + pad,
        )

        used: list[Point] = list(occupied)
        pairs: list[tuple[Point, Point]] = []
        for _ in range(human_num):
            start = self.sample_unused_position(used, bounds, min_dist, is_free)
            # Reflect through midpoint so the human walks across the corridor.
            reflected: Point = (2 * mid[0] - start[0], 2 * mid[1] - start[1])
            if all(
                math.hypot(reflected[0] - u[0], reflected[1] - u[1]) >= min_dist
                for u in used
            ) and (is_free is None or is_free(*reflected)):
                goal = reflected
            else:
                goal = self.sample_unused_position(used, bounds, min_dist, is_free)
            pairs.append((start, goal))
            used.append(start)
            used.append(goal)
        return pairs
