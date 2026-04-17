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
class PhaseConfig:
    """Resolves allocator params and per-phase human counts."""

    params: AllocatorParams
    sim_human_num: int
    human_num_train: int | None
    human_num_val: int | None
    human_num_test: int | None

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
