"""Unit tests for PhaseConfig / AllocatorParams (R2 / WP-2)."""

from __future__ import annotations

import configparser
import textwrap

import pytest


def _cfg(body: str) -> configparser.RawConfigParser:
    cp = configparser.RawConfigParser()
    cp.read_string(textwrap.dedent(body))
    return cp


@pytest.mark.unit
def test_phase_config_reads_defaults_when_section_missing() -> None:
    from crowd_sim.envs.utils.phase_config import PhaseConfig

    cp = _cfg(
        """
        [sim]
        human_num = 5
        """
    )
    pc = PhaseConfig.from_configparser(cp)
    assert pc.params.num_waypoints == 10
    assert pc.params.min_inter_waypoint_dist == 1.0
    assert pc.human_num_for("test") == 5
    assert pc.human_num_for("train") == 5
    assert pc.human_num_for("val") == 5


@pytest.mark.unit
def test_phase_config_reads_explicit_allocator_section() -> None:
    from crowd_sim.envs.utils.phase_config import PhaseConfig

    cp = _cfg(
        """
        [sim]
        human_num = 5

        [goal_allocator]
        num_waypoints = 6
        min_inter_waypoint_dist = 1.5
        human_num_train = 3
        human_num_val = 4
        human_num_test = 7
        """
    )
    pc = PhaseConfig.from_configparser(cp)
    assert pc.params.num_waypoints == 6
    assert pc.params.min_inter_waypoint_dist == 1.5
    assert pc.human_num_for("train") == 3
    assert pc.human_num_for("val") == 4
    assert pc.human_num_for("test") == 7


@pytest.mark.unit
def test_phase_config_falls_back_to_sim_human_num_per_phase() -> None:
    from crowd_sim.envs.utils.phase_config import PhaseConfig

    cp = _cfg(
        """
        [sim]
        human_num = 5

        [goal_allocator]
        num_waypoints = 4
        min_inter_waypoint_dist = 0.5
        human_num_test = 9
        """
    )
    pc = PhaseConfig.from_configparser(cp)
    assert pc.human_num_for("test") == 9
    # train/val missing — fall back to [sim].
    assert pc.human_num_for("train") == 5
    assert pc.human_num_for("val") == 5


@pytest.mark.unit
def test_phase_config_rejects_unknown_phase() -> None:
    from crowd_sim.envs.utils.phase_config import PhaseConfig

    cp = _cfg(
        """
        [sim]
        human_num = 5
        """
    )
    pc = PhaseConfig.from_configparser(cp)
    with pytest.raises(ValueError, match="unknown phase"):
        pc.human_num_for("imitation")  # type: ignore[arg-type]


@pytest.mark.unit
def test_phase_config_parses_checked_in_env_config() -> None:
    """Guard against drift between the checked-in env.config and PhaseConfig."""
    from pathlib import Path

    from crowd_sim.envs.utils.phase_config import PhaseConfig

    repo = Path(__file__).resolve().parent.parent
    cfg_path = repo / "crowd_nav" / "configs" / "env.config"
    cp = configparser.RawConfigParser()
    cp.read(cfg_path)
    pc = PhaseConfig.from_configparser(cp)
    # Values match the committed env.config (see Task 5 step 2).
    assert pc.params.num_waypoints == 10
    assert pc.params.min_inter_waypoint_dist == 1.0
    # train/val not overridden -> fall back to [sim].
    assert pc.human_num_for("train") == pc.sim_human_num
    assert pc.human_num_for("val") == pc.sim_human_num
    # WP-3: [static_map] section must round-trip too.
    assert pc.static_map.enabled is True
    assert pc.static_map.margin == 0.5


@pytest.mark.unit
def test_phase_config_static_map_defaults_when_section_missing() -> None:
    from crowd_sim.envs.utils.phase_config import PhaseConfig

    cp = _cfg(
        """
        [sim]
        human_num = 5
        """
    )
    pc = PhaseConfig.from_configparser(cp)
    assert pc.static_map.enabled is True
    assert pc.static_map.margin == 0.5


@pytest.mark.unit
def test_phase_config_reads_explicit_static_map_section() -> None:
    from crowd_sim.envs.utils.phase_config import PhaseConfig

    cp = _cfg(
        """
        [sim]
        human_num = 5

        [static_map]
        enabled = false
        margin = 0.8
        """
    )
    pc = PhaseConfig.from_configparser(cp)
    assert pc.static_map.enabled is False
    assert pc.static_map.margin == 0.8


@pytest.mark.unit
def test_phase_config_planner_defaults_when_section_missing() -> None:
    from crowd_sim.envs.utils.phase_config import PhaseConfig

    cp = _cfg(
        """
        [sim]
        human_num = 5
        """
    )
    pc = PhaseConfig.from_configparser(cp)
    assert pc.planner.enabled is False
    assert pc.planner.algorithm == "theta_star"
    assert pc.planner.inflation_radius == 0.5
    assert pc.planner.grid_resolution == 0.25
    assert pc.planner.bounds == (-15.0, 15.0, -15.0, 15.0)
    assert pc.planner.goal_tolerance == 0.3
    assert pc.planner.waypoint_simplify is True


@pytest.mark.unit
def test_phase_config_reads_explicit_planner_section() -> None:
    from crowd_sim.envs.utils.phase_config import PhaseConfig

    cp = _cfg(
        """
        [sim]
        human_num = 5

        [planner]
        enabled = true
        algorithm = theta_star
        inflation_radius = 0.4
        grid_resolution = 0.2
        bounds_xmin = -10.0
        bounds_xmax = 10.0
        bounds_ymin = -10.0
        bounds_ymax = 10.0
        goal_tolerance = 0.25
        waypoint_simplify = false
        """
    )
    pc = PhaseConfig.from_configparser(cp)
    assert pc.planner.enabled is True
    assert pc.planner.inflation_radius == 0.4
    assert pc.planner.grid_resolution == 0.2
    assert pc.planner.bounds == (-10.0, 10.0, -10.0, 10.0)
    assert pc.planner.goal_tolerance == 0.25
    assert pc.planner.waypoint_simplify is False


@pytest.mark.unit
def test_phase_config_rejects_unknown_planner_algorithm() -> None:
    from crowd_sim.envs.utils.phase_config import PhaseConfig

    cp = _cfg(
        """
        [sim]
        human_num = 5

        [planner]
        enabled = true
        algorithm = dijkstra
        """
    )
    with pytest.raises(ValueError, match="algorithm"):
        PhaseConfig.from_configparser(cp)


@pytest.mark.unit
def test_phase_config_parses_checked_in_env_config_enables_planner() -> None:
    from pathlib import Path

    from crowd_sim.envs.utils.phase_config import PhaseConfig

    repo = Path(__file__).resolve().parent.parent
    cfg_path = repo / "crowd_nav" / "configs" / "env.config"
    cp = configparser.RawConfigParser()
    cp.read(cfg_path)
    pc = PhaseConfig.from_configparser(cp)
    assert pc.planner.enabled is True
    assert pc.planner.algorithm == "theta_star"
    # Buffer must exceed robot.radius (0.5) — see test_env_config_planner_buffer.
    assert pc.planner.inflation_radius > 0.5


@pytest.mark.unit
def test_phase_config_parses_output_trained_env_config_enables_planner() -> None:
    from pathlib import Path

    from crowd_sim.envs.utils.phase_config import PhaseConfig

    repo = Path(__file__).resolve().parent.parent
    cfg_path = repo / "crowd_nav" / "data" / "output_trained" / "env.config"
    cp = configparser.RawConfigParser()
    cp.read(cfg_path)
    pc = PhaseConfig.from_configparser(cp)
    assert pc.planner.enabled is True
    assert pc.planner.algorithm == "theta_star"
