from dataclasses import asdict
from hashlib import sha256
from random import Random
from math import dist
import json
from shipnav.simulation import Traffic


def scenario_hash(data):
    return sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def make_scenario(sea, start, goal, count, seed):
    if count < 0:
        raise ValueError('Traffic count cannot be negative')
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


def load_traffic(data):
    return [Traffic(tuple(s['start']), tuple(s['goal']), s['speed'], s['radius']) for s in data['traffic']]
