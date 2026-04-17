"""Regression tests for human spawn logic after WP-2 rewiring."""

from __future__ import annotations

import configparser
from pathlib import Path

import pytest

from crowd_sim.envs.utils.seeding import seed_everything


REPO = Path(__file__).resolve().parent.parent


def _build_env():
    import gym

    seed_everything(42)
    env = gym.make("CrowdSim-v0")
    env.local_goal = [0, 0]
    env.curr_post = [-11, -11]
    cp = configparser.RawConfigParser()
    cp.read(REPO / "crowd_nav" / "configs" / "env.config")
    env.configure(cp)

    from crowd_sim.envs.utils.robot import Robot

    robot = Robot(cp, "robot")
    env.set_robot(robot)
    env.robot_initx, env.robot_inity = -3.0, -3.0
    env.robot_goalx, env.robot_goaly = 3.0, 3.0
    env.robot.set(-3.0, -3.0, 3.0, 3.0, 0, 0, 0)
    return env


@pytest.mark.integration
def test_generate_circle_crossing_human_new_has_no_print_statements(capsys) -> None:
    """generate_circle_crossing_human_new used to print (curr_post, local_goal)
    and (px, py, gx, gy) on every call — intolerable noise. Must be silent."""
    env = _build_env()
    env.humans = []
    env.generate_circle_crossing_human_new([-3.0, -3.0], [3.0, 3.0])
    captured = capsys.readouterr()
    assert captured.out == "", f"spurious stdout:\n{captured.out}"


@pytest.mark.integration
def test_generate_random_human_position_circle_crossing_matches_config_count() -> None:
    env = _build_env()
    env.generate_random_human_position(human_num=5, rule="circle_crossing")
    assert len(env.humans) == 5
    # Each human has distinct (px, py) — no overlapping spawns.
    positions = {(h.px, h.py) for h in env.humans}
    assert len(positions) == 5
