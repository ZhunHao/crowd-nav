"""AST guard - test.py wires ThetaStar into GoalAllocator.allocate_waypoints."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
TEST_PY = REPO / "crowd_nav" / "test.py"


@pytest.mark.unit
def test_test_py_imports_theta_star() -> None:
    body = TEST_PY.read_text()
    tree = ast.parse(body)
    froms = {
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
    }
    assert ("crowd_nav.planner.theta_star", "ThetaStar") in froms
    assert ("crowd_nav.planner.theta_star", "NoPathFound") in froms


@pytest.mark.unit
def test_test_py_invokes_as_waypoint_source() -> None:
    body = TEST_PY.read_text()
    assert "as_waypoint_source(" in body, (
        "test.py must build waypoint_source via ThetaStar.as_waypoint_source(...)"
    )


@pytest.mark.unit
def test_test_py_handles_no_path_found() -> None:
    body = TEST_PY.read_text()
    assert "except NoPathFound" in body, (
        "test.py must catch NoPathFound and fall back to straight-line"
    )
