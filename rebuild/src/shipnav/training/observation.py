"""Observation schema 1: bounded, unit-scaled features shared by NavigationEnv and
the ModernLearned adapter so training and deployment encode identically."""
from math import cos, dist, hypot, pi, sin

import numpy as np

SCHEMA, ACTION_SCHEMA = 1, 1
RANGE, VELOCITY_BOUND, RADIUS_BOUND, AGE_BOUND = 20., 2., 2., 5.
MAX_TARGETS, TARGET_FEATURES, RAYS, RAY_ITERATIONS, EGO_FEATURES = 12, 8, 16, 10, 8
SIZE = EGO_FEATURES + MAX_TARGETS*TARGET_FEATURES + RAYS


def _clip(value, bound):
    return max(-1., min(1., value/bound))


def ray_lengths(sea, p, radius):
    """Clear distance along 16 bearings (monotone segment test, bisection)."""
    lengths = []
    for k in range(RAYS):
        c, s = cos(2*pi*k/RAYS), sin(2*pi*k/RAYS)
        if sea.clear(p, (p[0]+c*RANGE, p[1]+s*RANGE), radius):
            lengths.append(RANGE)
            continue
        lo, hi = 0., RANGE
        for _ in range(RAY_ITERATIONS):
            mid = (lo+hi)/2
            lo, hi = (mid, hi) if sea.clear(p, (p[0]+c*mid, p[1]+s*mid), radius) else (lo, mid)
        lengths.append(lo)
    return lengths


def encode(packet):
    vessel = packet['vessel']
    if vessel is None:
        raise ValueError('Observation schema 1 requires marine dynamics (vessel heading)')
    p, v, g, speed = packet['position'], packet['velocity'], packet['goal'], packet['speed']
    dx, dy = g[0]-p[0], g[1]-p[1]
    d = hypot(dx, dy)
    values = [dx/d if d else 0., dy/d if d else 0., min(d, RANGE)/RANGE,
              _clip(v[0], speed), _clip(v[1], speed), sin(vessel.heading), cos(vessel.heading),
              float(packet['final'])]
    targets = sorted(packet['observed'], key=lambda o: dist(o['position'], p))[:MAX_TARGETS]
    for o in targets:
        values += [_clip(o['position'][0]-p[0], RANGE), _clip(o['position'][1]-p[1], RANGE),
                   _clip(o['velocity'][0], VELOCITY_BOUND), _clip(o['velocity'][1], VELOCITY_BOUND),
                   min(o['radius'], RADIUS_BOUND)/RADIUS_BOUND, min(o['age'], AGE_BOUND)/AGE_BOUND,
                   min(o['margin'], AGE_BOUND)/AGE_BOUND, 1.]
    values += [0.]*(TARGET_FEATURES*(MAX_TARGETS-len(targets)))
    values += [r/RANGE for r in ray_lengths(packet['sea'], p, packet['radius'])]
    return np.asarray(values, dtype=np.float32)


def action_to_velocity(action, speed):
    """Normalised 2-D desired velocity -> model-unit command inside the speed disk.
    Zero requests braking through the marine tracker; it never zeroes velocity."""
    a = np.asarray(action, dtype=float)
    if a.shape != (2,) or not np.isfinite(a).all():
        raise ValueError('Action must be two finite numbers')
    a = np.clip(a, -1., 1.)
    norm = hypot(*a)
    a = a/norm if norm > 1 else a
    return float(speed*a[0]), float(speed*a[1])
