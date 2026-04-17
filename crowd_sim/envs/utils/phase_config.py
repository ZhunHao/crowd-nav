"""Per-phase allocator params resolved from ``configparser`` (R2 / WP-2).

Central place for the ``[goal_allocator]`` section — keeps ``test.py`` and
``crowd_sim.py`` free of ``cp.getint`` / ``cp.getfloat`` boilerplate and of the
fallback logic (phase-specific override → ``[sim] human_num``).
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from typing import Literal

Phase = Literal["train", "val", "test"]
_VALID_PHASES: tuple[Phase, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class AllocatorParams:
    num_waypoints: int = 10
    min_inter_waypoint_dist: float = 1.0
    max_tries: int = 500


@dataclass(frozen=True)
class StaticMapParams:
    """[static_map] section — toggle + default margin for ``StaticMap.is_free``."""

    enabled: bool = True
    margin: float = 0.5


_VALID_PLANNER_ALGOS: tuple[str, ...] = ("theta_star",)


@dataclass(frozen=True)
class PlannerParams:
    """[planner] section — global path planner configuration (R4 / WP-4)."""

    enabled: bool = False
    algorithm: str = "theta_star"
    inflation_radius: float = 0.5
    grid_resolution: float = 0.25
    bounds: tuple[float, float, float, float] = (-15.0, 15.0, -15.0, 15.0)
    goal_tolerance: float = 0.3
    waypoint_simplify: bool = True


@dataclass(frozen=True)
class PhaseConfig:
    """Resolves allocator params and per-phase human counts."""

    params: AllocatorParams
    sim_human_num: int
    human_num_train: int | None
    human_num_val: int | None
    human_num_test: int | None
    static_map: StaticMapParams = StaticMapParams()
    planner: PlannerParams = PlannerParams()

    @classmethod
    def from_configparser(cls, cp: configparser.RawConfigParser) -> "PhaseConfig":
        sim_human_num = cp.getint("sim", "human_num", fallback=5)

        section = "goal_allocator"
        num_waypoints = cp.getint(section, "num_waypoints", fallback=10)
        min_inter = cp.getfloat(section, "min_inter_waypoint_dist", fallback=1.0)
        max_tries = cp.getint(section, "max_tries", fallback=500)

        def _opt(key: str) -> int | None:
            if cp.has_option(section, key):
                return cp.getint(section, key)
            return None

        sm_enabled = cp.getboolean("static_map", "enabled", fallback=True)
        sm_margin = cp.getfloat("static_map", "margin", fallback=0.5)

        pl_enabled = cp.getboolean("planner", "enabled", fallback=False)
        pl_algo = cp.get("planner", "algorithm", fallback="theta_star").strip()
        if pl_algo not in _VALID_PLANNER_ALGOS:
            raise ValueError(
                f"[planner] algorithm must be one of {_VALID_PLANNER_ALGOS}, got {pl_algo!r}"
            )
        pl_inflation = cp.getfloat("planner", "inflation_radius", fallback=0.5)
        pl_res = cp.getfloat("planner", "grid_resolution", fallback=0.25)
        pl_xmin = cp.getfloat("planner", "bounds_xmin", fallback=-15.0)
        pl_xmax = cp.getfloat("planner", "bounds_xmax", fallback=15.0)
        pl_ymin = cp.getfloat("planner", "bounds_ymin", fallback=-15.0)
        pl_ymax = cp.getfloat("planner", "bounds_ymax", fallback=15.0)
        pl_tol = cp.getfloat("planner", "goal_tolerance", fallback=0.3)
        pl_simplify = cp.getboolean("planner", "waypoint_simplify", fallback=True)

        return cls(
            params=AllocatorParams(
                num_waypoints=num_waypoints,
                min_inter_waypoint_dist=min_inter,
                max_tries=max_tries,
            ),
            sim_human_num=sim_human_num,
            human_num_train=_opt("human_num_train"),
            human_num_val=_opt("human_num_val"),
            human_num_test=_opt("human_num_test"),
            static_map=StaticMapParams(enabled=sm_enabled, margin=sm_margin),
            planner=PlannerParams(
                enabled=pl_enabled,
                algorithm=pl_algo,
                inflation_radius=pl_inflation,
                grid_resolution=pl_res,
                bounds=(pl_xmin, pl_xmax, pl_ymin, pl_ymax),
                goal_tolerance=pl_tol,
                waypoint_simplify=pl_simplify,
            ),
        )

    def human_num_for(self, phase: Phase) -> int:
        if phase not in _VALID_PHASES:
            raise ValueError(f"unknown phase {phase!r}; expected one of {_VALID_PHASES}")
        override = {
            "train": self.human_num_train,
            "val": self.human_num_val,
            "test": self.human_num_test,
        }[phase]
        return override if override is not None else self.sim_human_num
