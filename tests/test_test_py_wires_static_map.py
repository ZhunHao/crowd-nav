"""Guard that test.py wires StaticMap.is_free into the allocator (R3 / WP-3)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
TEST_PY = REPO / "crowd_nav" / "test.py"


@pytest.mark.unit
def test_test_py_imports_static_map() -> None:
    body = TEST_PY.read_text()
    tree = ast.parse(body)
    froms = {
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
    }
    assert ("crowd_sim.envs.utils.static_map", "StaticMap") in froms


@pytest.mark.unit
def test_test_py_allocate_waypoints_call_passes_is_free() -> None:
    """The `allocate_waypoints(...)` call must pass `is_free=...`."""
    body = TEST_PY.read_text()
    # Permissive regex across the call arguments (newlines allowed).
    pattern = re.compile(r"allocate_waypoints\s*\((?:[^()]|\([^()]*\))*?is_free\s*=", re.DOTALL)
    assert pattern.search(body), (
        "allocate_waypoints(...) must receive an is_free= argument in test.py"
    )


@pytest.mark.unit
def test_test_py_projects_infeasible_global_goal() -> None:
    """test.py must call StaticMap.project_to_free so an infeasible config
    goal (e.g., one inside the cordon) is snapped before allocate_waypoints."""
    body = TEST_PY.read_text()
    assert "project_to_free(" in body, (
        "test.py must call static_map.project_to_free(...) for an infeasible goal"
    )
