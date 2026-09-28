"""Finite-control-set receding-horizon MPC comparator.

This is NOT the RMPC-CBF research controller and makes no nonlinear-solver
guarantee. It ranks a fixed grid of constant-velocity command candidates over a
short rollout horizon (the tracker integrates bounded acceleration/yaw rate when a
`Vessel` context is set, otherwise an instantaneous-velocity rollout is used) and
returns only the first step of the winning candidate, as is standard for
receding-horizon control. `solver_failed` records that no candidate cleared the
safety margin for the current step; the caller must not treat the returned command
as safe when that flag is set.
"""
from math import sin, cos, pi, dist

from shipnav.dynamics import motion_from
from shipnav.horizon import horizon_steps
from shipnav.safety import assess, rollout


class MPC:
    def __init__(self):
        self.sea = None
        self.vessel = None
        self.predictions = []
        self.steps = None
        self.solver_failed = False

    def set_context(self, sea, vessel, predictions, steps=None):
        self.sea, self.vessel, self.predictions, self.steps = sea, vessel, predictions, steps

    def __call__(self, p, v, goal, neighbours, radius, speed, dt):
        if self.sea is None:
            raise RuntimeError('MPC.set_context must be called before use')
        # Finite-control-set MPC: constant velocity references over a finite horizon.
        # The tracker integrates bounded acceleration and yaw rate, then only step 1 executes.
        motion = motion_from(self.vessel, speed) if self.vessel is not None else None
        steps = self.steps or horizon_steps(dt, 'holonomic' if self.vessel is None else 'marine', speed)
        commands = [(0., 0.)] + [(s*cos(k*pi/12), s*sin(k*pi/12)) for s in (speed*.5, speed) for k in range(24)]
        ranked = []
        for command in commands:
            path = rollout(p, command, dt, steps, motion)
            clearance = assess(self.sea, path, self.predictions, radius)
            cost = dist(path[-1], goal) + .1*dist(command, v)
            ranked.append((clearance, cost, command))
        feasible = [r for r in ranked if r[0] > 0]
        self.solver_failed = not bool(feasible)
        if feasible:
            return min(feasible, key=lambda r: r[1])[2]
        return max(ranked, key=lambda r: (r[0], -r[1]))[2]
