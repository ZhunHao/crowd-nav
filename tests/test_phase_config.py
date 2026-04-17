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
