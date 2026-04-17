"""Guard rail: every shipped env.config must give the Theta* planner a
safety buffer strictly larger than the robot's radius. When
``inflation_radius <= robot.radius``, Theta* vertices sit on the env's
collision surface (CrowdSim.step checks is_free with ``margin=robot.radius``)
and any reactive SARL drift triggers immediate static_collision or
oscillation. This regression bit us on test_case 0 after WP-4 landed.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parent.parent
_ENV_CONFIGS = [
    _REPO / "crowd_nav" / "configs" / "env.config",
    _REPO / "crowd_nav" / "data" / "output_trained" / "env.config",
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "cfg_path",
    _ENV_CONFIGS,
    ids=lambda p: str(p.relative_to(_REPO)),
)
def test_planner_inflation_exceeds_robot_radius(cfg_path: Path) -> None:
    assert cfg_path.exists(), f"missing env.config: {cfg_path}"
    cp = configparser.RawConfigParser()
    cp.read(cfg_path)

    if not cp.has_section("planner") or not cp.getboolean(
        "planner", "enabled", fallback=False
    ):
        pytest.skip(f"{cfg_path.name} has no enabled [planner] section")

    robot_radius = cp.getfloat("robot", "radius")
    inflation = cp.getfloat("planner", "inflation_radius")
    assert inflation > robot_radius, (
        f"{cfg_path}: inflation_radius={inflation} must exceed "
        f"robot.radius={robot_radius} so Theta* vertices sit off the "
        f"collision surface. See test_case 0 regression (2026-04-17)."
    )
