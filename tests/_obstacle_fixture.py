"""Mirror of the hardcoded static_obstacles layout built in
``CrowdSim.reset()`` / ``light_reset()``. Integration tests rebuild the same
StaticMap without instantiating a full gym env.

Keep this file in lockstep with ``crowd_sim/envs/crowd_sim.py`` lines ~455-488.
If the loops there change, update this fixture in the same commit.
"""

from __future__ import annotations


def hardcoded_static_obstacles() -> list[dict]:
    obs: list[dict] = []
    # Left + right walls: 30 stacked 1x1 cells at x = +/-15, y in -15..14.
    for obs_idx in range(30):
        obs.append(
            {"type": "rect", "cx": -15.0, "cy": -15.0 + obs_idx * 1.0, "w": 1.0, "h": 1.0}
        )
    for obs_idx in range(30):
        obs.append(
            {"type": "rect", "cx": 15.0, "cy": -15.0 + obs_idx * 1.0, "w": 1.0, "h": 1.0}
        )
    # Bottom + top walls: 30 cells at y = +/-15, x in -15..14.
    for obs_idx in range(30):
        obs.append(
            {"type": "rect", "cx": -15.0 + obs_idx * 1.0, "cy": -15.0, "w": 1.0, "h": 1.0}
        )
    for obs_idx in range(30):
        obs.append(
            {"type": "rect", "cx": -15.0 + obs_idx * 1.0, "cy": 15.0, "w": 1.0, "h": 1.0}
        )
    # Interior row at y=-5: 20 cells at x in -15..4.
    for obs_idx in range(20):
        obs.append(
            {"type": "rect", "cx": -15.0 + obs_idx * 1.0, "cy": -5.0, "w": 1.0, "h": 1.0}
        )
    # Interior row at y=5: 20 cells at x in -4..15.
    for obs_idx in range(20):
        obs.append(
            {"type": "rect", "cx": 15.0 - obs_idx * 1.0, "cy": 5.0, "w": 1.0, "h": 1.0}
        )
    return obs
