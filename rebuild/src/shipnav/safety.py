from math import cos, sin, pi, hypot, dist

from shipnav.horizon import horizon_steps


def relative_clearance(a, b, c, d, radius):
    x, y = a[0]-c[0], a[1]-c[1]
    u, v = b[0]-d[0]-x, b[1]-d[1]-y
    f = max(0., min(1., -(x*u+y*v)/(u*u+v*v))) if u*u+v*v else 0.
    return hypot(x+f*u, y+f*v)-radius


def rollout(position, desired, dt, steps, motion=None, goal=None, arrival=0.):
    """Roll a constant command out `steps` steps (bounded `motion` if given).

    With a `goal`, the path holds position from the first sample after the
    current one (k >= 1, where `run_episode` checks arrival) that lies within
    `arrival` of it: the episode ends on arrival at the final goal, so motion
    past it is not a real consequence of the command. Callers pass `goal` only
    for the FINAL waypoint; intermediate waypoints keep the full rollout.
    """
    if motion is not None:
        path = motion(desired, dt, steps)
    else:
        path = [tuple(x+k*dt*v for x, v in zip(position, desired)) for k in range(steps+1)]
    if goal is not None:
        for k, point in enumerate(path[1:], start=1):
            if dist(point, goal) < arrival:
                return path[:k] + [point]*(len(path)-k)
    return path


def assess(sea, path, predictions, radius):
    for target in predictions:
        if len(target['points']) != len(path) or len(target['margins']) != len(path):
            raise ValueError(f"Prediction horizon ({len(target['points'])} points) does not match "
                             f"rollout horizon ({len(path)} points)")
    margin = float('inf')
    for k, (a, b) in enumerate(zip(path, path[1:])):
        if not sea.clear(a, b, radius):
            margin = min(margin, -1e6)
        for target in predictions:
            extra = max(target['margins'][k:k+2])
            margin = min(margin, relative_clearance(a, b, *target['points'][k:k+2], radius+target['radius']+extra))
    return margin


def choose(sea, position, nominal, predictions, radius, speed, dt, steps=None, motion=None,
           goal=None, final=False):
    """Minimally modify `nominal`. When `final` is set, `goal` is the route's
    final waypoint and rollouts hold there after arrival (arrival radius =
    ego `radius`, matching `run_episode`)."""
    if steps is None:
        steps = horizon_steps(dt, 'holonomic' if motion is None else 'marine', speed)
    hold = goal if final else None
    candidates = [tuple(nominal), (0., 0.)]
    candidates += [(s*cos(k*pi/8), s*sin(k*pi/8)) for s in (speed*.5, speed) for k in range(16)]
    scored = []
    for command in candidates:
        path = rollout(position, command, dt, steps, motion, hold, radius)
        clearance = assess(sea, path, predictions, radius)
        scored.append((command, clearance, path))
    feasible = [x for x in scored if x[1] > 0]
    chosen = min(feasible, key=lambda x: dist(x[0], nominal)) if feasible else max(scored, key=lambda x: (x[1], -dist(x[0], nominal)))
    return {'executed': chosen[0], 'path': chosen[2],
            'override': dist(chosen[0], nominal) > 1e-9,
            'no_feasible_action': not feasible,
            'predicted_clearance': None if chosen[1] == float('inf') else chosen[1]}
