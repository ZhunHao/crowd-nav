from dataclasses import dataclass
from math import atan2, dist, hypot, isfinite
from time import perf_counter
from typing import Callable
from shipnav.maps import SeaMap

Point = tuple[float, float]
Neighbour = tuple[Point, Point, float]
Policy = Callable[[Point, Point, Point, list[Neighbour], float, float, float], Point]


@dataclass(frozen=True)
class Traffic:
    start: Point
    goal: Point
    speed: float = .3
    radius: float = .6

    def __post_init__(self):
        if (not all(isfinite(x) for x in (*self.start, *self.goal, self.speed, self.radius))
                or self.speed <= 0 or self.radius <= 0 or self.start == self.goal):
            raise ValueError('Traffic requires finite distinct endpoints, positive speed and radius')

    @property
    def arrival(self) -> float:
        return dist(self.start, self.goal) / self.speed

    def at(self, time: float) -> Neighbour:
        fraction = min(1.0, max(0.0, time / self.arrival))
        p = tuple(a + fraction*(b-a) for a, b in zip(self.start, self.goal))
        v = tuple((b-a)/self.arrival if fraction < 1 else 0.0
                  for a, b in zip(self.start, self.goal))
        return p, v, self.radius

    def breakpoints(self, t0: float, t1: float) -> list[float]:
        return [self.arrival] if t0 < self.arrival < t1 else []


@dataclass(frozen=True)
class CourseChangeTraffic:
    """Scripted, deterministic course-change traffic: linear interpolation
    between a fixed, immutable list of (time, position) waypoints. Velocity
    is the constant segment velocity between consecutive waypoints and zero
    after the last one. Not a reactive target -- see `ReactiveTraffic` in
    `shipnav.reactive` for collision-responsive heading changes.
    """
    waypoints: tuple[tuple[float, Point], ...]
    radius: float = .6

    def __post_init__(self):
        times = [t for t, _ in self.waypoints]
        if len(self.waypoints) < 2:
            raise ValueError('CourseChangeTraffic requires at least two waypoints')
        if not all(isfinite(t) for t in times) or times != sorted(set(times)) or len(set(times)) != len(times):
            raise ValueError('CourseChangeTraffic waypoint times must be finite and strictly increasing')
        if not all(len(p) == 2 and all(isfinite(v) for v in p) for _, p in self.waypoints):
            raise ValueError('CourseChangeTraffic waypoint positions must be finite 2D points')
        if not isfinite(self.radius) or self.radius <= 0:
            raise ValueError('CourseChangeTraffic requires a positive finite radius')

    @property
    def start(self) -> Point:
        return self.waypoints[0][1]

    @property
    def goal(self) -> Point:
        return self.waypoints[-1][1]

    @property
    def arrival(self) -> float:
        return self.waypoints[-1][0]

    def at(self, time: float) -> Neighbour:
        if time <= self.waypoints[0][0]:
            return self.waypoints[0][1], (0.0, 0.0), self.radius
        for (ta, pa), (tb, pb) in zip(self.waypoints, self.waypoints[1:]):
            if ta <= time < tb:
                fraction = (time-ta)/(tb-ta)
                p = tuple(a+fraction*(b-a) for a, b in zip(pa, pb))
                v = tuple((b-a)/(tb-ta) for a, b in zip(pa, pb))
                return p, v, self.radius
        return self.goal, (0.0, 0.0), self.radius

    def breakpoints(self, t0: float, t1: float) -> list[float]:
        return [t for t, _ in self.waypoints if t0 < t < t1]


def segment_distance(a: Point, b: Point) -> float:
    delta = (b[0]-a[0], b[1]-a[1])
    squared = delta[0]**2 + delta[1]**2
    u = min(1.0, max(0.0, -(a[0]*delta[0]+a[1]*delta[1])/squared)) if squared else 0.0
    return hypot(a[0]+u*delta[0], a[1]+u*delta[1])


def swept_clearance(a: Point, b: Point, ship: Traffic, time: float, dt: float, radius: float) -> float:
    end = time+dt
    clipped = (min(end, max(time, breakpoint)) for breakpoint in ship.breakpoints(time, end))
    cuts = sorted({time, end, *clipped})
    values = []
    for left, right in zip(cuts, cuts[1:]):
        relative = []
        for t in (left, right):
            ego = tuple(x+(t-time)/dt*(y-x) for x, y in zip(a, b))
            other = ship.at(t)[0]
            relative.append(tuple(x-y for x, y in zip(ego, other)))
        values.append(segment_distance(*relative)-radius-ship.radius)
    return min(values)


def run_episode(sea: SeaMap, route: list[Point], traffic: list[Traffic], policy: Policy,
                dt: float = .25, limit: float = 100, radius: float = .5,
                speed: float = 1.0, cancel: Callable[[], bool] = lambda: False, *, observer=None,
                filtered=False, uncertainty=True, dynamics='holonomic') -> dict:
    if not all(isfinite(v) and v > 0 for v in (dt, limit, radius, speed)):
        raise ValueError('Episode parameters must be positive and finite')
    if not route or any(not sea.clear(p, p, radius) for p in route):
        raise ValueError('Missing route or blocked waypoint')
    for i, ship in enumerate(traffic):
        if not sea.clear(ship.start, ship.goal, ship.radius):
            raise ValueError('Traffic crosses land')
        if dist(ship.start, route[0]) <= radius+ship.radius:
            raise ValueError('Traffic overlaps start')
        if any(dist(ship.start, other.start) <= ship.radius+other.radius for other in traffic[:i]):
            raise ValueError('Traffic overlaps traffic at start')
    from shipnav.observations import Observer
    from shipnav.prediction import predict
    from shipnav.safety import choose
    from shipnav.horizon import horizon_steps
    observer = observer or Observer()
    diagnostics = []
    vessel = None
    if dynamics == 'marine':
        from shipnav.dynamics import Vessel, advance, motion_from, runout
        # Start on the bearing of the first route leg (east if already arrived).
        heading = atan2(route[1][1]-route[0][1], route[1][0]-route[0][0]) if len(route) > 1 else 0.
        vessel = Vessel(*route[0], heading, 0.)
    elif dynamics != 'holonomic':
        raise ValueError('Unknown dynamics')
    position, velocity, time, index = route[0], (0.0, 0.0), 0.0, 1
    minimum, latencies, frames, travelled = float('inf'), [], [], 0.0

    def record():
        frames.append({'t': time, 'position': list(position), 'velocity': list(velocity),
                       'goal_index': min(index, len(route)-1),
                       'traffic': [list(ship.at(time)[0]) for ship in traffic]})

    record()
    status = 'success' if len(route) == 1 else 'running'
    while status == 'running':
        if cancel():
            status = 'cancelled'
            break
        if time >= limit - 1e-9:
            status = 'timeout'
            break
        step = min(dt, limit-time)
        began = perf_counter()
        perceived = observer.observe(traffic, time, len(frames)-1)
        steps = horizon_steps(step, dynamics, speed)
        predictions = predict(perceived, step, steps=steps, uncertainty=uncertainty)
        neighbours = [(s['position'], s['velocity'], s['radius']) for s in perceived]
        if hasattr(policy, 'set_context'):
            policy.set_context(sea, vessel, predictions, steps=steps, final=index == len(route)-1)
        inference_start = perf_counter()
        nominal = tuple(policy(position, velocity, route[index], neighbours, radius, speed, step))
        latencies.append((perf_counter()-inference_start)*1000)
        if len(nominal) != 2 or not all(isfinite(x) for x in nominal) or hypot(*nominal) > speed+1e-6:
            raise ValueError('Policy returned invalid velocity')
        motion = motion_from(vessel, speed) if vessel is not None else None
        # Unfiltered runs never evaluate feasibility: no_feasible_action is None
        # ("not evaluated"), not False ("a feasible action existed").
        decision = choose(sea, position, nominal, predictions, radius, speed, step, steps=steps, motion=motion,
                          goal=route[index], final=index == len(route)-1) if filtered else {
            'executed': nominal, 'override': False, 'no_feasible_action': None, 'path': [], 'predicted_clearance': None}
        velocity = tuple(decision['executed'])
        decision_ms = (perf_counter()-began)*1000
        # Soft wall-clock indicator only: a Python worker thread makes no hard real-time
        # guarantee, so this flags a missed step budget rather than enforcing one.
        diagnostics.append({'t': time, 'observed': perceived, 'predictions': predictions,
                            'nominal': nominal, **decision,
                            'decision_ms': decision_ms,
                            'deadline_miss': decision_ms > step*1000,
                            'solver_failed': bool(getattr(policy, 'solver_failed', False))})
        if len(velocity) != 2 or not all(isfinite(x) for x in velocity) or hypot(*velocity) > speed+1e-6:
            raise ValueError('Policy returned invalid velocity')
        next_position = tuple(x + step*v for x, v in zip(position, velocity))
        if vessel is not None:
            vessel = advance(vessel, velocity, step, max_speed=speed)
            next_position = (vessel.x, vessel.y)
            velocity = tuple((b-a)/step for a, b in zip(position, next_position))
            diagnostics[-1]['heading'] = vessel.heading
            diagnostics[-1]['speed'] = vessel.speed
        for ship in traffic:
            if hasattr(ship, 'advance'):
                ship.advance(time, step, position)
        clearance = min((swept_clearance(position, next_position, ship, time, step, radius)
                         for ship in traffic), default=float('inf'))
        minimum = min(minimum, clearance)
        land_hit = not sea.clear(position, next_position, radius)
        ship_hit = clearance <= 0
        collision = ship_hit or land_hit
        diagnostics[-1].update(land_collision=land_hit, ship_collision=ship_hit, actual_velocity=velocity,
                               clearance=clearance if isfinite(clearance) else None)
        # Target/target and target/land validity is scored separately from ego
        # performance: it reflects the scenario's traffic realism, not a
        # safety intervention or collision attributable to the ego's policy.
        target_states = [ship.at(time+step) for ship in traffic]
        target_positions = [p for p, _, _ in target_states]
        target_radii = [r for _, _, r in target_states]
        target_land_invalid = any(not sea.clear(p, p, r) for p, r in zip(target_positions, target_radii))
        pair_clearances = [dist(target_positions[i], target_positions[j])-target_radii[i]-target_radii[j]
                           for i in range(len(target_positions)) for j in range(i+1, len(target_positions))]
        diagnostics[-1].update(target_land_invalid=target_land_invalid,
                               target_target_min_clearance=min(pair_clearances) if pair_clearances else None)
        travelled += dist(position, next_position)
        position, time = next_position, time+step
        if collision:
            status = 'collision'
        elif dist(position, route[index]) < radius:
            index += 1
            if index == len(route):
                status = 'success'
        record()
    result = {'status': status, 'elapsed': time, 'distance': travelled,
              'min_dynamic_clearance': minimum if isfinite(minimum) else None,
              'inference_ms': latencies, 'frames': frames, 'diagnostics': diagnostics}
    if vessel is not None:
        # Marine runs end at arrival with way still on. Success semantics are
        # unchanged; instead report the terminal speed and the minimum land/edge
        # clearance (ego radius subtracted) swept by a zero-command run-out under
        # the same bounded dynamics. Negative means the vessel would ground after
        # "arriving"; SeaMap.minimum_clearance saturates at 0 inside land, so a
        # run-out that ends on land reads -radius. Only evaluated after final
        # arrival (None otherwise); holonomic runs carry neither field.
        swept = runout(vessel, dt, speed) if status == 'success' else None
        result['terminal_speed'] = vessel.speed
        result['runout_min_clearance'] = None if swept is None else min(
            sea.minimum_clearance(a, b) for a, b in zip(swept, swept[1:] or swept)) - radius
    return result
