"""Procedural training and checkpoint-selection scenarios.

Seeds are disjoint from every frozen manifest; maps are random rectangular
islands, never harbour.json, Ubin or Southern Islands. 70% of scenarios use
corridor traffic timed to meet the ego on its reference route (crossing,
head-on, overtaking); the rest use uniform traffic, a fifth of it reactive.
"""
from hashlib import sha256
from math import dist
from pathlib import Path
from random import Random
import gzip
import json

from shipnav.episode import plan_route
from shipnav.maps import SeaMap
from shipnav.scenarios import make_scenario, scenario_hash, validate_scenario

TRAIN_SEEDS = range(100_000, 105_000)
SELECT_SEEDS = range(200_000, 200_200)
SIZES = (24., 36., 48.)
ENDPOINT_MARGIN, ISLAND_MARGIN = 2., 1.5


def _endpoints(rng, size):
    while True:
        a = (rng.uniform(ENDPOINT_MARGIN, size-ENDPOINT_MARGIN), rng.uniform(ENDPOINT_MARGIN, size-ENDPOINT_MARGIN))
        b = (rng.uniform(ENDPOINT_MARGIN, size-ENDPOINT_MARGIN), rng.uniform(ENDPOINT_MARGIN, size-ENDPOINT_MARGIN))
        if min(12., size/2) <= dist(a, b) <= 45.:
            return a, b


def _islands(rng, size, endpoints, count):
    land = []
    for _ in range(count):
        for _ in range(100):
            w, h = rng.uniform(1.5, size/5), rng.uniform(1.5, size/5)
            x, y = rng.uniform(1., size-1.-w), rng.uniform(1., size-1.-h)
            if all(not (x-ISLAND_MARGIN < px < x+w+ISLAND_MARGIN and y-ISLAND_MARGIN < py < y+h+ISLAND_MARGIN)
                   for px, py in endpoints):
                land.append((x, y, x+w, y+h))
                break
    return tuple(land)


def generate(seed, split):
    """Raises NoPath/PlanningLimit/InvalidRoute/ValueError for an unusable seed."""
    rng = Random(seed)
    size = rng.choice(SIZES)
    start, goal = _endpoints(rng, size)
    sea = SeaMap((0., 0., size, size), _islands(rng, size, (start, goal), rng.randint(0, 4)))
    corridor = rng.random() < .7
    route = plan_route(sea, start, goal) if corridor else None
    scenario = make_scenario(sea, start, goal, rng.randint(1, 5), seed, corridor=route)
    reactive = not corridor and rng.random() < .2
    scenario = {**scenario, 'split': split, 'family': f'generated_{scenario["family"]}',
                'traffic_mode': 'reactive' if reactive else 'scripted'}
    plan_route(sea, start, goal)  # every kept scenario must be reachable
    validate_scenario(scenario)
    return json.loads(json.dumps(scenario, allow_nan=False))  # JSON-normal form: lists, not tuples


def _encode(scenarios):
    lines = (json.dumps(s, sort_keys=True, separators=(',', ':'), allow_nan=False) for s in scenarios)
    return gzip.compress(('\n'.join(lines)+'\n').encode(), mtime=0)


def write_corpus(directory, splits, failures):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 1, 'generator': 'shipnav.training.scenarios', 'splits': {}, 'failures': failures}
    for name, scenarios in splits.items():
        data = _encode(scenarios)
        (directory/f'{name}.jsonl.gz').write_bytes(data)
        manifest['splits'][name] = {'file': f'{name}.jsonl.gz', 'sha256': sha256(data).hexdigest(),
                                    'count': len(scenarios)}
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    return manifest


def load_split(directory, split):
    directory = Path(directory)
    entry = json.loads((directory/'manifest.json').read_text())['splits'][split]
    data = (directory/entry['file']).read_bytes()
    if sha256(data).hexdigest() != entry['sha256']:
        raise ValueError(f'Corpus hash mismatch for split {split}')
    scenarios = [json.loads(line) for line in gzip.decompress(data).decode().splitlines() if line]
    if len(scenarios) != entry['count']:
        raise ValueError(f'Corpus count mismatch for split {split}')
    return scenarios


def frozen_identities(root):
    """Scenario hashes and seeds of every frozen evaluation/development manifest."""
    root = Path(root)
    hashes, seeds = set(), set(range(1000, 1010))  # historical train split, unused
    for path in (root/'scenarios/splits.json', root/'scenarios/baseline/splits.json'):
        for entries in json.loads(path.read_text())['splits'].values():
            hashes |= {e['scenario_hash'] for e in entries}
            seeds |= {e['seed'] for e in entries if 'seed' in e}
    return hashes, seeds
