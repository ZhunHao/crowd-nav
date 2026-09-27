"""Velocity-reference tracker used by the marine motion model.

`advance` is the tracker: given a vessel state and a desired velocity reference, it
returns the next reachable state under bounded acceleration/yaw-rate. This module is
the intended extension point if a richer tracker (e.g. line-of-sight guidance, a PID
heading autopilot) replaces the direct pass-through below.
"""
from shipnav.dynamics import advance
