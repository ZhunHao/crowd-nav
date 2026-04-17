"""Guard that test.py reads [goal_allocator] and hands waypoints to the env."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
TEST_PY = REPO / "crowd_nav" / "test.py"


@pytest.mark.unit
def test_test_py_no_longer_contains_hardcoded_goal_list() -> None:
    body = TEST_PY.read_text()
    # Exact literal from the deleted block.
    assert "[[-5,-9], [0, -10], [6,-9], [10,-5], [5,0]" not in body, (
        "hardcoded 10-waypoint list must be replaced by GoalAllocator"
    )


@pytest.mark.unit
def test_test_py_imports_goal_allocator_and_phase_config() -> None:
    body = TEST_PY.read_text()
    tree = ast.parse(body)
    froms = {
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
    }
    assert ("crowd_sim.envs.utils.goal_allocator", "GoalAllocator") in froms
    assert ("crowd_sim.envs.utils.phase_config", "PhaseConfig") in froms


@pytest.mark.unit
def test_test_py_calls_allocate_waypoints() -> None:
    body = TEST_PY.read_text()
    assert "allocate_waypoints(" in body
