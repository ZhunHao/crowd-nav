from dataclasses import asdict
from hashlib import sha256
from random import Random
from math import atan2, cos, dist, isfinite, pi, sin
import json
from shipnav.simulation import CourseChangeTraffic, Traffic


SCENARIO_KEYS = {'schema': int, 'seed': int, 'map': dict, 'start': list, 'goal': list,
                 'traffic': list, 'traffic_mode': str, 'family': str, 'split': str}
TRAFFIC_KEYS = {'start', 'goal', 'speed', 'radius'}
COURSE_CHANGE_KEYS = {'kind', 'waypoints', 'radius'}
MAP_KEYS = {'schema', 'type', 'coordinate_system', 'bounds', 'features', 'metadata'}
OBSERVATION_KEYS = {'noise', 'delay', 'dropout', 'stale_speed_bound'}


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f'{name} must be a finite number')


def _point(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f'{name} must be a 2D point [x, y]')
    for v in value:
        _number(v, name)


def _validate_traffic_entry(entry, i):
    where = f'traffic[{i}]'
    if not isinstance(entry, dict):
        raise ValueError(f'{where} must be an object')
    if 'kind' in entry and entry['kind'] != 'course_change':
        raise ValueError(f"{where}: unknown traffic kind {entry['kind']!r}")
    expected = COURSE_CHANGE_KEYS if 'kind' in entry else TRAFFIC_KEYS
    missing, unknown = expected-entry.keys(), entry.keys()-expected
    if missing:
        raise ValueError(f'{where} missing key(s): {", ".join(sorted(missing))}')
    if unknown:
        raise ValueError(f'{where} has unknown key(s): {", ".join(sorted(unknown))}')
    _number(entry['radius'], f'{where}.radius')
    if 'kind' in entry:
        waypoints = entry['waypoints']
        if not isinstance(waypoints, (list, tuple)):
            raise ValueError(f'{where}.waypoints must be a list of [time, [x, y]]')
        for j, waypoint in enumerate(waypoints):
            if not isinstance(waypoint, (list, tuple)) or len(waypoint) != 2:
                raise ValueError(f'{where}.waypoints[{j}] must be [time, [x, y]]')
            _number(waypoint[0], f'{where}.waypoints[{j}] time')
            _point(waypoint[1], f'{where}.waypoints[{j}] position')
    else:
        _point(entry['start'], f'{where}.start')
        _point(entry['goal'], f'{where}.goal')
        _number(entry['speed'], f'{where}.speed')


def validate_scenario(data):
    """Check a scenario dict's schema at the service boundary; raises
    ValueError naming the offending key. Values' physical validity (land,
    overlap, positive speed) is still checked by the constructors and
    `run_episode`."""
    if not isinstance(data, dict):
        raise ValueError('Scenario must be a JSON object')
    missing = SCENARIO_KEYS.keys()-data.keys()
    if missing:
        raise ValueError(f'Scenario missing key(s): {", ".join(sorted(missing))}')
    for key, kind in SCENARIO_KEYS.items():
        value = data[key]
        if key in ('start', 'goal'):
            _point(value, f'Scenario {key}')
        elif isinstance(value, bool) or not isinstance(value, (tuple, list) if kind is list else kind):
            raise ValueError(f'Scenario {key} must be of type {kind.__name__}')
    missing_map = MAP_KEYS-data['map'].keys()
    if missing_map:
        raise ValueError(f'Scenario map missing key(s): {", ".join(sorted(missing_map))}')
    for i, entry in enumerate(data['traffic']):
        _validate_traffic_entry(entry, i)


def validate_observation(observation):
    """Observation settings accept only the Observer's noise/delay/dropout/
    stale_speed_bound numbers; the seed always comes from the scenario."""
    if not isinstance(observation, dict):
        raise ValueError('Observation settings must be an object')
    unknown = observation.keys()-OBSERVATION_KEYS
    if unknown:
        raise ValueError(f'Unknown observation setting(s): {", ".join(sorted(map(str, unknown)))} '
                         f'(allowed: {", ".join(sorted(OBSERVATION_KEYS))}; the seed comes from the scenario)')
    for key, value in observation.items():
        _number(value, f'Observation {key}')


def scenario_hash(data):
    return sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def make_scenario(sea, start, goal, count, seed, *, corridor=None):
    """Random traffic. With `corridor` (a reference route) ships are timed to meet the
    nominal 1 m/s ego along it (see `corridor_traffic`); otherwise uniform over the map."""
    if count < 0:
        raise ValueError('Traffic count cannot be negative')
    if corridor is not None:
        ships = corridor_traffic(sea, corridor, count, Random(seed))
        return {'schema': 1, 'seed': seed, 'map': sea.to_dict(), 'start': list(start),
                'goal': list(goal), 'traffic': [traffic_to_dict(s) for s in ships],
                'traffic_mode': 'scripted', 'family': 'corridor', 'split': 'smoke'}
    rng, ships = Random(seed), []
    x0, y0, x1, y1 = sea.bounds
    for _ in range(count):
        for attempt in range(500):
            a = (rng.uniform(x0+.7, x1-.7), rng.uniform(y0+.7, y1-.7))
            b = (rng.uniform(x0+.7, x1-.7), rng.uniform(y0+.7, y1-.7))
            if dist(a, b) < 1 or not sea.clear(a, b, .6):
                continue
            if min(dist(a, start), dist(a, goal)) <= 1.3:
                continue
            if any(dist(a, s.start) <= 1.4 for s in ships):
                continue
            ships.append(Traffic(a, b)); break
        else:
            raise ValueError('Scenario placement failed; preserve this failed seed')
    return {'schema': 1, 'seed': seed, 'map': sea.to_dict(), 'start': list(start),
            'goal': list(goal), 'traffic': [asdict(s) for s in ships],
            'traffic_mode': 'scripted', 'family': 'random', 'split': 'smoke'}


def _along(route, s):
    """Point and unit tangent at arc length `s` of a polyline."""
    for a, b in zip(route, route[1:]):
        length = dist(a, b)
        if s <= length or b == route[-1]:
            f = min(1., s/length) if length else 0.
            return (a[0]+f*(b[0]-a[0]), a[1]+f*(b[1]-a[1])), ((b[0]-a[0])/length, (b[1]-a[1])/length)
        s -= length
    raise ValueError('Route needs at least two distinct points')


def corridor_traffic(sea, route, count, rng, radius=.6, speeds=(.3, 1.), run_up=(3., 15.)):
    """Scripted ships that reach a point of the reference route when a nominal 1 m/s ego
    does. Each waits at its start, departs `run_up` model units before the meeting point
    and continues the same distance beyond it, so encounters happen on maps far larger
    than the ships' voyages. Kinds: crossing (±60-120°), head-on and overtaking.
    The reference route is frozen into the timing, not into the scenario, so runs with
    another planner or controller face identical traffic."""
    total = sum(dist(a, b) for a, b in zip(route, route[1:]))
    if total <= 0:
        return []
    ships = []
    for _ in range(count):
        for attempt in range(500):
            arc = rng.uniform(.1, .95)*total
            point, (tx, ty) = _along(route, arc)
            heading = atan2(ty, tx)
            kind = rng.choice(('crossing', 'crossing', 'head_on', 'overtake'))
            speed = rng.uniform(*speeds)
            if kind == 'crossing':
                heading += rng.choice((1, -1))*rng.uniform(pi/3, 2*pi/3)
            elif kind == 'head_on':
                heading += pi + rng.uniform(-pi/12, pi/12)
            else:
                speed = rng.uniform(speeds[0], min(speeds[1], .6))
            d = (cos(heading), sin(heading))
            run = rng.uniform(*run_up)
            meet = arc  # nominal ego arrival time at this arc length (1 model unit/s)
            depart = max(0., meet-run/speed)
            a = tuple(p-c*speed*(meet-depart) for p, c in zip(point, d))
            b = tuple(p+c*run for p, c in zip(point, d))
            if not sea.clear(a, b, radius) or dist(a, route[0]) <= 1.3 or dist(a, route[-1]) <= 1.3:
                continue
            if any(dist(a, s.start) <= 1.4 for s in ships):
                continue
            waypoints = ((0., a),) + (((depart, a),) if depart > 0 else ()) + \
                        ((meet, point), (meet+run/speed, b))
            ships.append(CourseChangeTraffic(waypoints, radius))
            break
        else:
            raise ValueError('Scenario placement failed; preserve this failed seed')
    return ships


def traffic_to_dict(ship):
    """Serialise a Traffic-like object into a scenario JSON traffic entry.
    Plain fixed-voyage `Traffic` uses its dataclass fields as-is; scripted
    `CourseChangeTraffic` serialises its waypoint list under `kind:
    'course_change'` so `load_traffic` can tell them apart."""
    if isinstance(ship, CourseChangeTraffic):
        return {'kind': 'course_change',
                'waypoints': [[t, list(p)] for t, p in ship.waypoints],
                'radius': ship.radius}
    return asdict(ship)


def load_traffic(data):
    ships = []
    for entry in data['traffic']:
        if entry.get('kind') == 'course_change':
            waypoints = tuple((float(t), tuple(p)) for t, p in entry['waypoints'])
            ships.append(CourseChangeTraffic(waypoints, entry['radius']))
        else:
            ships.append(Traffic(tuple(entry['start']), tuple(entry['goal']), entry['speed'], entry['radius']))
    return ships
