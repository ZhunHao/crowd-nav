"""Tier B regression: robot vs static obstacle collision in CrowdSim.step."""

from __future__ import annotations

import configparser
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parent.parent


def _build_env_with_rect_wall():
    """Build an env and install a single wall rect above the robot."""
    import gym

    from crowd_sim.envs.utils.static_map import Obstacle, StaticMap
    from crowd_sim.envs.utils.seeding import seed_everything
    from crowd_sim.envs.utils.robot import Robot
    from crowd_sim.envs.utils.action import ActionXY
    from crowd_sim.envs.policy.orca import ORCA

    seed_everything(42)
    env = gym.make("CrowdSim-v0")
    env.local_goal = [0, 0]
    env.curr_post = [-3, -3]
    cp = configparser.RawConfigParser()
    cp.read(REPO / "crowd_nav" / "configs" / "env.config")
    env.configure(cp)

    robot = Robot(cp, "robot")
    # ORCA gives us a non-trainable policy with multiagent_training attribute.
    robot.set_policy(ORCA())
    env.set_robot(robot)
    env.robot_initx, env.robot_inity = 0.0, 4.5
    env.robot_goalx, env.robot_goaly = 0.0, -4.5
    env.reset("test", 0)

    # Robot starts at y=4.0 — wall at y=5 (h=1) is at [4.5, 5.5]; inflated by
    # robot radius 0.3 gives [4.2, 5.8]. So y=4.0 is free; +y motion crosses.
    env.robot.set(0.0, 4.0, 0.0, -4.5, 0.0, 0.0, 0.0)
    # Install a 20x1 rect centred at y=5.
    wall = Obstacle(kind="rect", cx=0.0, cy=5.0, w=20.0, h=1.0)
    env.static_map = StaticMap(obstacles=(wall,), margin=0.0)
    # No dynamic humans for this isolated regression.
    env.humans = []

    return env, ActionXY


@pytest.mark.integration
def test_step_reports_collision_when_robot_enters_obstacle() -> None:
    from crowd_sim.envs.utils.info import Collision

    env, ActionXY = _build_env_with_rect_wall()
    # Velocity +y at 0.3 m/s, time_step ~0.25 => end_position ~(0, 4.575).
    # Robot radius 0.3 => inflated at y=4.575+0.3=4.875 < wall.cy-h/2=4.5,
    # so use a bigger step to cross the inflated wall.
    action = ActionXY(0.0, 1.5)  # 1.5 m/s +y.
    _ob, _reward, done, info = env.step(action, update=False)
    assert done is True, f"expected done=True on wall collision, got info={info!r}"
    assert isinstance(info, Collision), f"expected Collision, got {type(info).__name__}"


@pytest.mark.integration
def test_step_no_collision_when_robot_stays_clear() -> None:
    from crowd_sim.envs.utils.info import Collision

    env, ActionXY = _build_env_with_rect_wall()
    # Velocity -y so robot moves away from wall — no collision.
    action = ActionXY(0.0, -0.5)
    _ob, _reward, done, info = env.step(action, update=False)
    assert not isinstance(info, Collision), f"unexpected Collision: info={info!r}"
