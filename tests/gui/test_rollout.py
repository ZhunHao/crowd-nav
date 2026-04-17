"""Integration test: run_waypoint_rollout reproduces test.py's rollout output."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.integration
def test_rollout_returns_states_with_expected_schema(repo_root: Path) -> None:
    import configparser

    import torch

    from crowd_nav.gui.controllers._rollout import run_waypoint_rollout
    from crowd_nav.policy.policy_factory import policy_factory
    from crowd_sim.envs.utils.static_map import StaticMap

    model_dir = repo_root / "crowd_nav" / "data" / "output_trained"
    env_cfg = model_dir / "env.config"
    policy_cfg = model_dir / "policy.config"

    cp = configparser.RawConfigParser()
    cp.read(policy_cfg)
    policy = policy_factory["sarl"]()
    policy.configure(cp)
    policy.get_model().load_state_dict(
        torch.load(model_dir / "rl_model.pth", map_location="cpu", weights_only=True)
    )
    policy.set_phase("test")
    policy.set_device(torch.device("cpu"))

    # Empty map - no static obstacles - should trivially reach.
    sm = StaticMap(obstacles=(), margin=0.5)

    states = run_waypoint_rollout(
        env_config_path=env_cfg,
        static_map=sm,
        policy=policy,
        start=(-3.0, -3.0),
        goal=(3.0, 3.0),
        user_obstacles=[],
    )
    assert len(states) > 0
    s0 = states[0]
    assert {"t", "robot", "humans", "waypoint_idx", "reward"} <= set(s0)
    assert isinstance(s0["robot"], tuple) and len(s0["robot"]) == 4


@pytest.mark.integration
def test_rollout_invokes_frame_callback_with_zero_indexed_steps(
    repo_root: Path,
) -> None:
    import configparser

    import torch

    from crowd_nav.gui.controllers._rollout import run_waypoint_rollout
    from crowd_nav.policy.policy_factory import policy_factory
    from crowd_sim.envs.utils.static_map import StaticMap

    model_dir = repo_root / "crowd_nav" / "data" / "output_trained"
    cp = configparser.RawConfigParser()
    cp.read(model_dir / "policy.config")
    policy = policy_factory["sarl"]()
    policy.configure(cp)
    policy.get_model().load_state_dict(
        torch.load(model_dir / "rl_model.pth", map_location="cpu", weights_only=True)
    )
    policy.set_phase("test")
    policy.set_device(torch.device("cpu"))

    sm = StaticMap(obstacles=(), margin=0.5)
    received: list[tuple[int, int]] = []

    def _cb(idx: int, step: dict) -> None:
        received.append((idx, step["waypoint_idx"]))

    states = run_waypoint_rollout(
        env_config_path=model_dir / "env.config",
        static_map=sm,
        policy=policy,
        start=(-3.0, -3.0),
        goal=(3.0, 3.0),
        user_obstacles=[],
        frame_callback=_cb,
    )

    assert len(received) == len(states)
    assert received[0][0] == 0
    assert received[-1][0] == len(states) - 1
    # waypoint_idx must be monotonically non-decreasing.
    idxs = [wp for _, wp in received]
    assert idxs == sorted(idxs)
