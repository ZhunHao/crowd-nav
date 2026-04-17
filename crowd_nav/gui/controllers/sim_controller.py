"""Headless pipeline facade for the PyQt GUI (WP-5 / R5).

Wraps the WP-4 rollout logic from ``crowd_nav/test.py`` as a testable
QObject-less class. The Qt layer (MainWindow, SimWorker) composes this
facade; it never imports Qt directly so unit tests can exercise it
without a QApplication.
"""

from __future__ import annotations

import configparser
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

from crowd_sim.envs.utils.map_loader import load_static_map
from crowd_sim.envs.utils.phase_config import PhaseConfig
from crowd_sim.envs.utils.static_map import StaticMap

if TYPE_CHECKING:
    from crowd_sim.envs.policy.policy import Policy

Point = tuple[float, float]
_LOG = logging.getLogger(__name__)


@dataclass
class SimController:
    env_config_path: Path
    static_map: Optional[StaticMap] = None
    policy: Optional["Policy"] = None
    start: Optional[Point] = None
    goal: Optional[Point] = None
    user_obstacles: list[dict] = field(default_factory=list)
    last_waypoints: list[tuple[float, float]] = field(default_factory=list)
    _phase_cfg: Optional[PhaseConfig] = field(default=None, init=False, repr=False)
    _base_map: Optional[StaticMap] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not Path(self.env_config_path).exists():
            raise FileNotFoundError(
                f"env config not found: {self.env_config_path}"
            )
        cp = configparser.RawConfigParser()
        cp.read(self.env_config_path)
        self._phase_cfg = PhaseConfig.from_configparser(cp)

    # ---------- button handlers (pure python) ----------

    def load_map(self, path: Path) -> None:
        margin = self._phase_cfg.static_map.margin if self._phase_cfg else 0.0
        self._base_map = load_static_map(path, margin=margin)
        # Reset user overlay when a new map is loaded.
        self.user_obstacles = []
        self.static_map = self._base_map

    def load_model(self, path: Path, policy_name: str) -> None:
        from crowd_nav.policy.policy_factory import policy_factory
        import torch

        policy = policy_factory[policy_name]()
        cfg_path = self.env_config_path.parent / "policy.config"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"policy.config not found next to env.config at {cfg_path}"
            )
        cp = configparser.RawConfigParser()
        cp.read(cfg_path)
        policy.configure(cp)
        if policy.trainable:
            policy.get_model().load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True)
            )
        else:
            _LOG.warning(
                "load_model: policy %r is not trainable; weight file %s ignored",
                policy_name, path,
            )
        self.policy = policy
        self.policy.set_phase("test")
        self.policy.set_device(torch.device("cpu"))

    def set_start(self, xy: Point) -> None:
        self._reject_if_blocked(xy, label="start")
        self.start = xy

    def set_goal(self, xy: Point) -> None:
        self._reject_if_blocked(xy, label="goal")
        self.goal = xy

    def add_obstacle(self, shape: dict) -> None:
        self.user_obstacles.append(shape)
        # Always rebuild from the pristine base map so user shapes never
        # compound across successive clicks.
        if self._base_map is not None:
            base_dicts = [_obstacle_to_dict(o) for o in self._base_map.obstacles]
            self.static_map = StaticMap.from_static_obstacles(
                base_dicts + self.user_obstacles,
                margin=self._base_map.margin,
            )

    def run_episode(
        self,
        frame_callback: Optional[Callable[[int, dict], None]] = None,
    ) -> list[dict]:
        """Drive a full episode, reusing the test.py rollout logic.
        Returns a list of step dicts suitable for export_writer.write_exports.
        Raises ValueError if start/goal/policy/map are not set.
        """
        for name, val in (
            ("static_map", self.static_map),
            ("policy", self.policy),
            ("start", self.start),
            ("goal", self.goal),
        ):
            if val is None:
                raise ValueError(f"cannot run episode: {name} not set")
        from crowd_nav.gui.controllers._rollout import run_waypoint_rollout

        result = run_waypoint_rollout(
            env_config_path=self.env_config_path,
            static_map=self.static_map,
            policy=self.policy,
            start=self.start,
            goal=self.goal,
            user_obstacles=self.user_obstacles,
            frame_callback=frame_callback,
        )
        self.last_waypoints = result["waypoints"]
        return result["states"]

    # ---------- internals ----------

    def _reject_if_blocked(self, xy: Point, *, label: str) -> None:
        if self.static_map is None:
            return  # no map loaded yet; allow free-form placement
        margin = self._phase_cfg.static_map.margin if self._phase_cfg else 0.0
        if not self.static_map.is_free(*xy, margin=margin):
            raise ValueError(
                f"{label} {xy} is inside obstacle (margin={margin})"
            )


def _obstacle_to_dict(obs) -> dict:
    d: dict = {"type": obs.kind, "cx": obs.cx, "cy": obs.cy}
    if obs.kind == "rect":
        d["w"] = obs.w
        d["h"] = obs.h
    else:
        d["r"] = obs.r
    return d
