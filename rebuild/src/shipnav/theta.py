"""Basic Theta* entry point.

This module does not implement its own search. Line-of-sight parent
relaxation happens inside the shared grid search in ``shipnav.planning``
while the search runs, not as a post-processing/smoothing pass over an
A*-style raw path.
"""
from shipnav.planning import plan


def theta_star(sea, start, goal, clearance=.7, resolution=1.):
    return plan(sea, start, goal, planner='theta', clearance=clearance, resolution=resolution).points
