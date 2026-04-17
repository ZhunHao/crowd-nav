"""StaticMap geometry primitives (R3 / WP-3).

Pure geometry — no I/O, no gym, no rvo2. Exposes ``is_free(x, y, margin)`` so the
allocator, the env's Tier-B collision branch, and Tier-C ORCA wiring can share a
single source of truth. WP-4's Theta* planner will reuse the same predicate.

The env still owns the hardcoded obstacle layout (see
``crowd_sim.envs.crowd_sim.CrowdSim.reset``); :meth:`StaticMap.from_static_obstacles`
adapts its ``list[dict]`` representation into frozen :class:`Obstacle` tuples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping

ObstacleKind = Literal["rect", "circle"]


@dataclass(frozen=True)
class Obstacle:
    """One static obstacle. Unused geometry fields (``r`` for rects, ``w``/``h``
    for circles) stay at ``0.0`` so the struct is frozen-safe and serialisable.
    """

    kind: ObstacleKind
    cx: float
    cy: float
    w: float = 0.0
    h: float = 0.0
    r: float = 0.0


@dataclass(frozen=True)
class StaticMap:
    """Frozen bundle of obstacles + default margin.

    ``is_free(x, y, margin=None)`` returns ``True`` iff the point is outside every
    obstacle inflated by ``margin`` (or ``self.margin`` when ``None``). Boundary
    points count as occupied (``<=``) so the robot cannot hug a wall.
    """

    obstacles: tuple[Obstacle, ...] = field(default_factory=tuple)
    margin: float = 0.0

    def is_free(self, x: float, y: float, margin: float | None = None) -> bool:
        m = self.margin if margin is None else margin
        for obs in self.obstacles:
            if obs.kind == "rect":
                if abs(x - obs.cx) <= obs.w / 2.0 + m and abs(y - obs.cy) <= obs.h / 2.0 + m:
                    return False
            elif obs.kind == "circle":
                if math.hypot(x - obs.cx, y - obs.cy) <= obs.r + m:
                    return False
            else:  # pragma: no cover — frozen dataclass + Literal prevents this
                raise ValueError(f"unknown obstacle kind: {obs.kind!r}")
        return True

    def project_to_free(
        self,
        x: float,
        y: float,
        margin: float | None = None,
        step: float = 0.1,
        max_radius: float = 10.0,
    ) -> tuple[float, float]:
        """Return ``(x, y)`` when already free, else the nearest free point
        found via ring-spiral search outward. Raises ``RuntimeError`` if no
        free point exists within ``max_radius``.
        """
        if self.is_free(x, y, margin=margin):
            return (float(x), float(y))
        r = step
        while r <= max_radius + 1e-9:
            n = max(8, int(2.0 * math.pi * r / step))
            for i in range(n):
                angle = 2.0 * math.pi * i / n
                px = x + r * math.cos(angle)
                py = y + r * math.sin(angle)
                if self.is_free(px, py, margin=margin):
                    return (float(px), float(py))
            r += step
        raise RuntimeError(
            f"could not project ({x}, {y}) to free point within r={max_radius}"
        )

    @classmethod
    def from_static_obstacles(
        cls,
        obs_list: Iterable[Mapping[str, object]],
        margin: float = 0.0,
    ) -> "StaticMap":
        """Convert env's ``list[dict]`` representation into a :class:`StaticMap`.

        Expected dict shapes::

            {"type": "rect",   "cx": float, "cy": float, "w": float, "h": float}
            {"type": "circle", "cx": float, "cy": float, "r": float}
        """
        obstacles: list[Obstacle] = []
        for raw in obs_list:
            kind = raw.get("type")
            if kind == "rect":
                obstacles.append(
                    Obstacle(
                        kind="rect",
                        cx=float(raw["cx"]),  # type: ignore[arg-type]
                        cy=float(raw["cy"]),  # type: ignore[arg-type]
                        w=float(raw["w"]),  # type: ignore[arg-type]
                        h=float(raw["h"]),  # type: ignore[arg-type]
                    )
                )
            elif kind == "circle":
                obstacles.append(
                    Obstacle(
                        kind="circle",
                        cx=float(raw["cx"]),  # type: ignore[arg-type]
                        cy=float(raw["cy"]),  # type: ignore[arg-type]
                        r=float(raw["r"]),  # type: ignore[arg-type]
                    )
                )
            else:
                raise ValueError(f"unknown obstacle type: {kind!r}")
        return cls(obstacles=tuple(obstacles), margin=float(margin))


def rect_vertices_ccw(obs: Obstacle) -> list[tuple[float, float]]:
    """Return a rect's 4 corners in counter-clockwise order (RVO2 convention)."""
    if obs.kind != "rect":
        raise ValueError(f"rect_vertices_ccw expects a rect, got kind={obs.kind!r}")
    hw = obs.w / 2.0
    hh = obs.h / 2.0
    return [
        (obs.cx - hw, obs.cy - hh),
        (obs.cx + hw, obs.cy - hh),
        (obs.cx + hw, obs.cy + hh),
        (obs.cx - hw, obs.cy + hh),
    ]
