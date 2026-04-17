"""Rollout extraction (WP-5 / Task 3b).

Ports the waypoint loop from ``crowd_nav/test.py`` into a callable so both
the CLI (``crowd_nav/test.py``) and the PyQt GUI (``SimController``) share
a single source of truth. Behaviour is identical; only the I/O surface differs.

``frame_callback(step_counter, step)`` receives 0-indexed global step counts
across all waypoints in the episode (not waypoint-local).
"""

from __future__ import annotations

import configparser
import logging
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

import gym

from crowd_nav.planner.theta_star import NoPathFound, ThetaStar
from crowd_sim.envs.utils.goal_allocator import GoalAllocator
from crowd_sim.envs.utils.phase_config import PhaseConfig
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.static_map import StaticMap

if TYPE_CHECKING:
    from crowd_sim.envs.policy.policy import Policy

Point = tuple[float, float]
_LOG = logging.getLogger(__name__)


def run_waypoint_rollout(
    *,
    env_config_path: Path,
    static_map: StaticMap,
    policy: "Policy",
    start: Point,
    goal: Point,
    user_obstacles: list[dict],
    frame_callback: Optional[Callable[[int, dict], None]] = None,
) -> list[dict]:
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_path)
    phase_cfg = PhaseConfig.from_configparser(env_config)

    env = gym.make("CrowdSim-v0")
    try:
        env.local_goal = list(start)
        # Match test.py's sentinel so human-seeding bbox reproduces test.py exactly.
        env.curr_post = [-11, -11]
        env.configure(env_config)

        robot = Robot(env_config, "robot")
        robot.set_policy(policy)
        env.set_robot(robot)
        policy.set_env(env)

        env.reset("test", 0)

        # env.reset() wipes static_obstacles; append user shapes after.
        if user_obstacles:
            env.static_obstacles = list(env.static_obstacles) + list(user_obstacles)

        # Goal projection (mirror test.py, using the map's own margin as base).
        projection_margin = static_map.margin
        if phase_cfg.planner.enabled:
            projection_margin = max(
                projection_margin, phase_cfg.planner.inflation_radius
            )
        if not static_map.is_free(*goal, margin=projection_margin):
            goal = static_map.project_to_free(*goal, margin=projection_margin)
            env.robot_goalx, env.robot_goaly = goal

        allocator = GoalAllocator(max_tries=phase_cfg.params.max_tries)
        waypoint_source = None
        if phase_cfg.planner.enabled:
            planner = ThetaStar(
                static_map=static_map,
                inflation=phase_cfg.planner.inflation_radius,
                grid_resolution=phase_cfg.planner.grid_resolution,
                bounds=phase_cfg.planner.bounds,
                simplify=phase_cfg.planner.waypoint_simplify,
            )
            waypoint_source = planner.as_waypoint_source(phase_cfg.params.num_waypoints)

        try:
            waypoints = allocator.allocate_waypoints(
                start=start,
                goal=goal,
                num_waypoints=phase_cfg.params.num_waypoints,
                min_inter_dist=phase_cfg.params.min_inter_waypoint_dist,
                is_free=static_map.is_free,
                waypoint_source=waypoint_source,
            )
        except NoPathFound as exc:
            _LOG.warning(
                "Theta* found no path (%s); falling back to straight-line.", exc
            )
            waypoints = allocator.allocate_waypoints(
                start=start,
                goal=goal,
                num_waypoints=phase_cfg.params.num_waypoints,
                min_inter_dist=phase_cfg.params.min_inter_waypoint_dist,
                is_free=static_map.is_free,
                waypoint_source=None,
            )

        states: list[dict] = []
        step_counter = 0
        prev_x, prev_y = start
        for wp_idx, (gx, gy) in enumerate(waypoints):
            env.local_goal = [gx, gy]
            env.robot_initx, env.robot_inity = prev_x, prev_y
            env.robot_goalx, env.robot_goaly = gx, gy
            ob = env.light_reset("test", 0)
            done = False
            while not done:
                action = robot.act(ob)
                ob, reward, done, _info = env.step(action)
                step = {
                    "t": env.global_time,
                    "robot": (
                        float(robot.px),
                        float(robot.py),
                        float(robot.vx),
                        float(robot.vy),
                    ),
                    "humans": [
                        (float(h.px), float(h.py), float(h.vx), float(h.vy))
                        for h in env.humans
                    ],
                    "waypoint_idx": wp_idx,
                    "reward": float(reward),
                }
                states.append(step)
                if frame_callback is not None:
                    frame_callback(step_counter, step)
                step_counter += 1
            prev_x, prev_y = robot.px, robot.py
    finally:
        env.close()

    return states
