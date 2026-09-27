"""Reactive traffic: a handcrafted, collision-responsive heading change used
as a controlled test target. This is a simple avoidance rule for benchmark
purposes only -- it is NOT COLREGs-compliant and NOT an ORCA/reciprocal
velocity-obstacle implementation. Do not label it as either.

`ReactiveTraffic.advance` is called by `run_episode` once per step, after the
ego's control decision and before truth (swept-clearance) checks are made --
so the target's reaction is always to the ego's *realized* position for that
step, never to a value computed ahead of the ego's own decision. Its `at(t)`
interpolates the immutable history of realized `(time, position, velocity)`
samples recorded by `advance`; delayed observations therefore read back a
true past state, never a regenerated or resimulated trajectory.
"""
from math import dist, hypot
from shipnav.simulation import Traffic


class ReactiveTraffic:
    def __init__(self, ship, sea):
        self.start, self.goal, self.speed, self.radius = ship.start, ship.goal, ship.speed, ship.radius
        self.sea, self.history = sea, [(0., self.start, (0., 0.))]
        self.arrival = float('inf')

    def at(self, t):
        for (ta, pa, va), (tb, pb, vb) in zip(self.history, self.history[1:]):
            if ta <= t < tb:
                f = (t-ta)/(tb-ta)
                return tuple(a+f*(b-a) for a, b in zip(pa, pb)), vb, self.radius
        _, p, v = self.history[-1]
        return p, v, self.radius

    def breakpoints(self, t0, t1):
        return [ta for ta, _, _ in self.history if t0 < ta < t1]

    def advance(self, t, dt, ego):
        p = self.history[-1][1]
        d = dist(p, self.goal)
        v = tuple((b-a)/d*min(self.speed, d/dt) if d else 0. for a, b in zip(p, self.goal))
        if dist(p, ego) < 3.:
            away = (p[0]-ego[0], p[1]-ego[1]); norm = hypot(*away)
            if norm:
                v = tuple(self.speed*x/norm for x in away)
        q = tuple(x+dt*y for x, y in zip(p, v))
        if not self.sea.clear(p, q, self.radius):
            q, v = p, (0., 0.)
        self.history.append((t+dt, q, v))
