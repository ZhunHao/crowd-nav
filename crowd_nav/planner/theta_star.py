"""Theta* any-angle global path planner (R4 / WP-4).

Classic Theta* on an 8-connected grid rasterised from :class:`StaticMap`.
``line_of_sight`` samples the world segment through ``StaticMap.is_free``
so the planner never disagrees with the allocator / collision checker.
``plan`` returns a list of world-coordinate vertices (excluding the start,
always terminating at the goal).

No I/O, no gym, no configparser.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from crowd_sim.envs.utils.static_map import StaticMap

Point = tuple[float, float]
Bounds = tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)
WaypointSourceFn = Callable[[Point, Point, int], list[Point]]


class NoPathFound(RuntimeError):
    """Raised when Theta* cannot connect start to goal on the current map."""


# 8-connected neighbour offsets: (drow, dcol, step_cost).
_NEIGHBOURS: tuple[tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


@dataclass(frozen=True)
class ThetaStar:
    """Frozen Theta* planner. Rasterises ``static_map`` once into a blocked-cell
    bitmap on construction; subsequent ``plan`` calls reuse the grid.
    """

    static_map: StaticMap
    inflation: float = 0.5
    grid_resolution: float = 0.25
    bounds: Bounds = (-15.0, 15.0, -15.0, 15.0)
    simplify: bool = True

    _grid: np.ndarray = field(init=False, repr=False, compare=False)
    _nrows: int = field(init=False, repr=False, compare=False)
    _ncols: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        xmin, xmax, ymin, ymax = self.bounds
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(f"degenerate bounds: {self.bounds}")
        if self.grid_resolution <= 0:
            raise ValueError(
                f"grid_resolution must be > 0, got {self.grid_resolution}"
            )
        ncols = max(1, int(math.ceil((xmax - xmin) / self.grid_resolution)))
        nrows = max(1, int(math.ceil((ymax - ymin) / self.grid_resolution)))
        grid = np.zeros((nrows, ncols), dtype=np.bool_)
        for i in range(nrows):
            cy = ymin + (i + 0.5) * self.grid_resolution
            for j in range(ncols):
                cx = xmin + (j + 0.5) * self.grid_resolution
                if not self.static_map.is_free(cx, cy, margin=self.inflation):
                    grid[i, j] = True
        object.__setattr__(self, "_grid", grid)
        object.__setattr__(self, "_nrows", nrows)
        object.__setattr__(self, "_ncols", ncols)

    # ------------------------------------------------------------------
    # Grid <-> world helpers.
    # ------------------------------------------------------------------
    def _world_to_cell(self, p: Point) -> tuple[int, int]:
        xmin, _xmax, ymin, _ymax = self.bounds
        j = int((p[0] - xmin) / self.grid_resolution)
        i = int((p[1] - ymin) / self.grid_resolution)
        j = max(0, min(self._ncols - 1, j))
        i = max(0, min(self._nrows - 1, i))
        return (i, j)

    def _cell_to_world(self, cell: tuple[int, int]) -> Point:
        xmin, _xmax, ymin, _ymax = self.bounds
        i, j = cell
        return (
            xmin + (j + 0.5) * self.grid_resolution,
            ymin + (i + 0.5) * self.grid_resolution,
        )

    def _is_blocked(self, cell: tuple[int, int]) -> bool:
        i, j = cell
        if not (0 <= i < self._nrows and 0 <= j < self._ncols):
            return True
        return bool(self._grid[i, j])

    # ------------------------------------------------------------------
    # Line-of-sight.
    # ------------------------------------------------------------------
    def line_of_sight(self, a: Point, b: Point) -> bool:
        """True iff every sample along ``a`` to ``b`` passes
        ``static_map.is_free(x, y, margin=self.inflation)``.
        Sampling step = ``grid_resolution / 2``.
        """
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = math.hypot(dx, dy)
        if length < 1e-12:
            return self.static_map.is_free(a[0], a[1], margin=self.inflation)
        step = self.grid_resolution / 2.0
        n = max(2, int(math.ceil(length / step)) + 1)
        for k in range(n + 1):
            t = k / n
            x = a[0] + dx * t
            y = a[1] + dy * t
            if not self.static_map.is_free(x, y, margin=self.inflation):
                return False
        return True

    # ------------------------------------------------------------------
    # Theta* main loop.
    # ------------------------------------------------------------------
    def plan(self, start: Point, goal: Point) -> list[Point]:
        if start == goal:
            raise ValueError(f"start == goal ({start!r}); nothing to plan")

        if self.line_of_sight(start, goal):
            return [goal]

        start_cell = self._world_to_cell(start)
        goal_cell = self._world_to_cell(goal)
        if self._is_blocked(goal_cell):
            raise NoPathFound(
                f"goal cell {goal_cell} (world {goal!r}) is blocked under "
                f"inflation={self.inflation}"
            )
        if self._is_blocked(start_cell):
            raise NoPathFound(
                f"start cell {start_cell} (world {start!r}) is blocked under "
                f"inflation={self.inflation}"
            )

        def h(cell: tuple[int, int]) -> float:
            return math.hypot(cell[0] - goal_cell[0], cell[1] - goal_cell[1])

        g: dict[tuple[int, int], float] = {start_cell: 0.0}
        parent: dict[tuple[int, int], tuple[int, int]] = {start_cell: start_cell}
        counter = 0
        open_heap: list[tuple[float, int, tuple[int, int]]] = [
            (h(start_cell), counter, start_cell)
        ]
        closed: set[tuple[int, int]] = set()

        while open_heap:
            _f, _ctr, s = heapq.heappop(open_heap)
            if s in closed:
                continue
            if s == goal_cell:
                return self._reconstruct(parent, goal_cell, goal)
            closed.add(s)

            for di, dj, step_cost in _NEIGHBOURS:
                n_cell = (s[0] + di, s[1] + dj)
                if self._is_blocked(n_cell) or n_cell in closed:
                    continue
                # No corner-cutting across blocked diagonals.
                if di != 0 and dj != 0 and (
                    self._is_blocked((s[0] + di, s[1]))
                    or self._is_blocked((s[0], s[1] + dj))
                ):
                    continue
                p = parent[s]
                p_world = self._cell_to_world(p)
                n_world = self._cell_to_world(n_cell)
                if self.line_of_sight(p_world, n_world):
                    cand_g = g[p] + math.hypot(
                        p[0] - n_cell[0], p[1] - n_cell[1]
                    )
                    cand_parent = p
                else:
                    cand_g = g[s] + step_cost
                    cand_parent = s
                if cand_g < g.get(n_cell, math.inf):
                    g[n_cell] = cand_g
                    parent[n_cell] = cand_parent
                    counter += 1
                    heapq.heappush(open_heap, (cand_g + h(n_cell), counter, n_cell))

        raise NoPathFound(
            f"no path from {start!r} to {goal!r} on grid "
            f"{self._nrows}x{self._ncols} (inflation={self.inflation})"
        )

    def _reconstruct(
        self,
        parent: dict[tuple[int, int], tuple[int, int]],
        goal_cell: tuple[int, int],
        goal_world: Point,
    ) -> list[Point]:
        cells: list[tuple[int, int]] = []
        c = goal_cell
        for _ in range(self._nrows * self._ncols + 1):
            cells.append(c)
            if parent[c] == c:
                break
            c = parent[c]
        else:  # pragma: no cover
            raise RuntimeError("parent chain did not terminate")
        cells.reverse()
        path = [self._cell_to_world(cell) for cell in cells[1:]]
        if path:
            path[-1] = goal_world
        else:
            path = [goal_world]
        if self.simplify:
            path = _rdp_simplify(path, epsilon=self.grid_resolution)
        return path

    # ------------------------------------------------------------------
    # WaypointSourceFn adapter.
    # ------------------------------------------------------------------
    def as_waypoint_source(self, n: int) -> WaypointSourceFn:
        """Wrap :meth:`plan` in a closure matching
        :data:`GoalAllocator.WaypointSourceFn` - callers get exactly ``n``
        points, last == goal, resampled by arc length.
        """
        if n <= 0:
            raise ValueError(f"n must be >= 1, got {n}")

        def _source(start: Point, goal: Point, num: int) -> list[Point]:
            if num != n:
                raise ValueError(
                    f"as_waypoint_source bound to n={n} but called with num={num}"
                )
            path = self.plan(start=start, goal=goal)
            return _arc_length_resample(start, path, num)

        return _source


# ----------------------------------------------------------------------
# Private helpers.
# ----------------------------------------------------------------------
def _point_segment_distance(p: Point, a: Point, b: Point) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    if dx == 0.0 and dy == 0.0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(
        0.0,
        min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)),
    )
    proj = (a[0] + t * dx, a[1] + t * dy)
    return math.hypot(p[0] - proj[0], p[1] - proj[1])


def _rdp_simplify(path: list[Point], epsilon: float) -> list[Point]:
    """Ramer-Douglas-Peucker, preserving first + last vertex."""
    if len(path) < 3:
        return list(path)

    def _rdp(points: Sequence[Point]) -> list[Point]:
        if len(points) < 3:
            return list(points)
        a, b = points[0], points[-1]
        max_d = -1.0
        idx = -1
        for i in range(1, len(points) - 1):
            d = _point_segment_distance(points[i], a, b)
            if d > max_d:
                max_d = d
                idx = i
        if max_d <= epsilon or idx < 0:
            return [a, b]
        left = _rdp(points[: idx + 1])
        right = _rdp(points[idx:])
        return left[:-1] + right

    return _rdp(path)


def _arc_length_resample(start: Point, path: list[Point], n: int) -> list[Point]:
    """Return ``n`` points along start to path[0] to ... to path[-1] equally spaced
    by arc length. Last element is ``path[-1]`` (== goal).
    """
    if not path:
        raise ValueError("path must be non-empty")
    vertices: list[Point] = [start] + list(path)
    seg_lengths = [
        math.hypot(
            vertices[i + 1][0] - vertices[i][0],
            vertices[i + 1][1] - vertices[i][1],
        )
        for i in range(len(vertices) - 1)
    ]
    total = sum(seg_lengths)
    if total < 1e-12:
        return [path[-1]] * n
    cum = [0.0]
    for s in seg_lengths:
        cum.append(cum[-1] + s)
    out: list[Point] = []
    for k in range(1, n + 1):
        target = total * k / n
        i = 0
        while i < len(cum) - 2 and cum[i + 1] < target:
            i += 1
        seg_len = seg_lengths[i]
        if seg_len < 1e-12:
            out.append(vertices[i + 1])
            continue
        t = (target - cum[i]) / seg_len
        a = vertices[i]
        b = vertices[i + 1]
        out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    out[-1] = path[-1]
    return out
