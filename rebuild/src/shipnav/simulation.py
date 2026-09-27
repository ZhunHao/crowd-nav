from dataclasses import dataclass
from math import dist, hypot, isfinite
from random import Random
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


def traffic_for_route(sea: SeaMap, route: list[Point], count: int, seed: int) -> list[Traffic]:
    if count < 0:
        raise ValueError('Traffic count cannot be negative')
    if count == 0:
        return []
    if len(route) < 2:
        raise ValueError('Moving traffic requires a nonzero route')
    rng, ships = Random(seed), []
    for index in range(count):
        a, b = route[index % (len(route)-1):][:2]
        length = dist(a, b)
        if length == 0:
            raise ValueError('Repeated route points')
        normal = (-(b[1]-a[1])/length, (b[0]-a[0])/length)
        for _ in range(500):
            f, width = rng.uniform(.15, .85), rng.uniform(1.5, 4.0)
            center = tuple(x + f*(y-x) for x, y in zip(a, b))
            start = tuple(x + width*n for x, n in zip(center, normal))
            goal = tuple(x - width*n for x, n in zip(center, normal))
            if rng.random() < .5:
                start, goal = goal, start
            if not sea.clear(start, goal, .6):
                continue
            if dist(start, route[0]) <= 1.3 or dist(start, route[-1]) <= 1.3:
                continue
            if any(dist(start, other.start) <= 1.4 for other in ships):
                continue
            ships.append(Traffic(start, goal))
            break
        else:
            raise ValueError('Cannot place traffic on this route; lower count or change map')
    return ships


def segment_distance(a: Point, b: Point) -> float:
    delta = (b[0]-a[0], b[1]-a[1])
    squared = delta[0]**2 + delta[1]**2
    u = min(1.0, max(0.0, -(a[0]*delta[0]+a[1]*delta[1])/squared)) if squared else 0.0
    return hypot(a[0]+u*delta[0], a[1]+u*delta[1])


def swept_clearance(a: Point, b: Point, ship: Traffic, time: float, dt: float, radius: float) -> float:
    cuts = sorted({time, time+dt, min(time+dt, max(time, ship.arrival))})
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
                speed: float = 1.0, cancel: Callable[[], bool] = lambda: False) -> dict:
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
        neighbours = [ship.at(time) for ship in traffic]
        began = perf_counter()
        velocity = tuple(policy(position, velocity, route[index], neighbours, radius, speed, step))
        latencies.append((perf_counter()-began)*1000)
        if len(velocity) != 2 or not all(isfinite(x) for x in velocity) or hypot(*velocity) > speed+1e-6:
            raise ValueError('Policy returned invalid velocity')
        next_position = tuple(x + step*v for x, v in zip(position, velocity))
        clearance = min((swept_clearance(position, next_position, ship, time, step, radius)
                         for ship in traffic), default=float('inf'))
        minimum = min(minimum, clearance)
        collision = clearance <= 0 or not sea.clear(position, next_position, radius)
        travelled += dist(position, next_position)
        position, time = next_position, time+step
        if collision:
            status = 'collision'
        elif dist(position, route[index]) < radius:
            index += 1
            if index == len(route):
                status = 'success'
        record()
    return {'status': status, 'elapsed': time, 'distance': travelled,
            'min_dynamic_clearance': minimum if isfinite(minimum) else None,
            'inference_ms': latencies, 'frames': frames}
