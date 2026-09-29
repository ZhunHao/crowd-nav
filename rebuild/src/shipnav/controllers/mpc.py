"""Finite-control-set receding-horizon MPC comparator.

This is NOT the RMPC-CBF research controller and makes no nonlinear-solver
guarantee. It ranks a fixed set of constant-velocity command candidates over the
shared lookahead horizon (`shipnav.horizon`; the tracker integrates bounded
acceleration/yaw rate when a `Vessel` context is set, otherwise an
instantaneous-velocity rollout is used) and returns only the first step of the
winning candidate, as is standard for receding-horizon control.

Candidates: stop, a 24-heading grid at half and full speed, and goal-scaled
commands pointing at the goal with speed ``min(speed, d/(k*dt))`` for several
``k`` (so a candidate can land on a nearby goal instead of overshooting it).

Cost (progress within the horizon, not endpoint distance):
``min_k [d_k + TIME_WEIGHT*k*dt] + EFFORT_WEIGHT*|command - velocity|`` where
``d_k`` is the rollout's distance to the goal at step k, and is 0 from the first
step (k >= 1) the rollout is within the ego radius of the goal (arrival, as in
`run_episode`). Arriving candidates therefore win and the earliest arrival is
preferred; an overshooting candidate is not penalised for where it would be
after arriving. For the route's FINAL goal the rollout itself also holds at the
goal after arrival (the same `safety.rollout` helper the filter uses), so motion
past the final goal is not assessed for safety either.

`solver_failed` records that no candidate cleared the safety margin for the
current step; the caller must not treat the returned command as safe when that
flag is set.
"""
from math import sin, cos, pi, dist

from shipnav.dynamics import motion_from
from shipnav.horizon import horizon_steps
from shipnav.safety import assess, local_sea, rollout

TIME_WEIGHT = .05
EFFORT_WEIGHT = .1
GOAL_SCALED_STEPS = (1, 2, 4, 8)
HEADINGS = 24


def progress_cost(path, goal, arrival, dt):
    """min over k of (distance to goal, 0 once arrived) + a small time term."""
    best, arrived = float('inf'), False
    for k, point in enumerate(path):
        arrived = arrived or (k > 0 and dist(point, goal) < arrival)
        best = min(best, (0. if arrived else dist(point, goal)) + TIME_WEIGHT*k*dt)
    return best


def candidates(position, goal, speed, dt):
    commands = [(0., 0.)]
    commands += [(s*cos(k*2*pi/HEADINGS), s*sin(k*2*pi/HEADINGS))
                 for s in (speed*.5, speed) for k in range(HEADINGS)]
    distance = dist(position, goal)
    if distance > 0:
        for k in GOAL_SCALED_STEPS:
            magnitude = min(speed, distance/(k*dt))
            commands.append(tuple((b-a)/distance*magnitude for a, b in zip(position, goal)))
    return commands


class MPC:
    def __init__(self):
        self.sea = None
        self.vessel = None
        self.predictions = []
        self.steps = None
        self.final = True
        self.solver_failed = False

    def set_context(self, sea, vessel, predictions, steps=None, final=True):
        """`final` marks `goal` in the next call as the route's final waypoint."""
        self.sea, self.vessel, self.predictions = sea, vessel, predictions
        self.steps, self.final = steps, final

    def __call__(self, p, v, goal, neighbours, radius, speed, dt):
        if self.sea is None:
            raise RuntimeError('MPC.set_context must be called before use')
        motion = motion_from(self.vessel, speed) if self.vessel is not None else None
        steps = self.steps or horizon_steps(dt, 'holonomic' if self.vessel is None else 'marine', speed)
        hold = goal if self.final else None
        sea = local_sea(self.sea, p, speed*dt*steps, radius)
        ranked = []
        for command in candidates(p, goal, speed, dt):
            path = rollout(p, command, dt, steps, motion, hold, radius)
            clearance = assess(sea, path, self.predictions, radius)
            cost = progress_cost(path, goal, radius, dt) + EFFORT_WEIGHT*dist(command, v)
            ranked.append((clearance, cost, command))
        feasible = [r for r in ranked if r[0] > 0]
        self.solver_failed = not bool(feasible)
        if feasible:
            return min(feasible, key=lambda r: r[1])[2]
        return max(ranked, key=lambda r: (r[0], -r[1]))[2]
