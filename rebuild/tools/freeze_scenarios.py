"""Freeze canonical encounter scenarios and seeded train/dev/calibration/test
splits to deterministic JSON, independently of any planner or policy.

Each scenario file is a complete `make_scenario`-schema dict (full map,
start, goal, traffic list, `traffic_mode`, `family`, `split`, `seed`) -- not a
seed recipe -- so replaying a scenario never depends on this generator or on
future changes to the random placement algorithm.

Run as a script (`python tools/freeze_scenarios.py`) to (re)write
`scenarios/*.json` and `scenarios/splits.json` under the repository root.
`generate(output_dir)` is the reusable entry point: it is deterministic, so
calling it twice into different directories reproduces identical bytes.
"""
from dataclasses import asdict
from hashlib import sha256
from math import dist
from pathlib import Path

from shipnav.maps import SeaMap, canonical_json
from shipnav.scenarios import load_traffic, make_scenario, scenario_hash
from shipnav.simulation import Traffic

ROOT = Path(__file__).resolve().parents[1]

CANONICAL_FAMILIES = ('head_on', 'crossing', 'crossing_mirrored', 'overtake',
                      'narrow_passage', 'detour_harbour', 'unreachable',
                      'shore_goal', 'multi_conflict')

EGO_START, EGO_GOAL = (3., 12.), (21., 12.)

# split -> (map factory, ego start, ego goal, seed range, traffic count)
SEEDED_SPLITS = (
    ('train', lambda: SeaMap((0., 0., 24., 24.)),
     (2., 2.), (22., 22.), range(1000, 1010), 4),
    ('dev', lambda: SeaMap((0., 0., 24., 24.), ((9., 5., 14., 18.),)),
     (2., 2.), (22., 22.), range(2000, 2005), 3),
    ('calibration', lambda: SeaMap.load(ROOT / 'maps/harbour.json'),
     (2., 2.), (22., 22.), range(3000, 3005), 3),
)


def _scenario(sea, start, goal, traffic, family, split, seed=0):
    return {'schema': 1, 'seed': seed, 'map': sea.to_dict(), 'start': list(start),
            'goal': list(goal), 'traffic': [asdict(s) for s in traffic],
            'traffic_mode': 'scripted', 'family': family, 'split': split}


def validate(scenario, radius=.5):
    """Validate a scenario the way run_episode would before its first step.

    Raises ValueError on the same conditions run_episode raises for: blocked
    waypoints, traffic crossing land, traffic overlapping the start, or
    traffic overlapping other traffic at t=0. Never mutates or "fixes" the
    scenario -- callers record the failure instead.
    """
    sea = SeaMap.from_dict(scenario['map'])
    start, goal = tuple(scenario['start']), tuple(scenario['goal'])
    traffic = load_traffic(scenario)
    if not sea.clear(start, start, radius) or not sea.clear(goal, goal, radius):
        raise ValueError('Start or goal blocked by land')
    for i, ship in enumerate(traffic):
        if not sea.clear(ship.start, ship.goal, ship.radius):
            raise ValueError('Traffic crosses land')
        if dist(ship.start, start) <= radius + ship.radius:
            raise ValueError('Traffic overlaps start')
        if any(dist(ship.start, other.start) <= ship.radius + other.radius for other in traffic[:i]):
            raise ValueError('Traffic overlaps traffic at start')


def canonical_scenarios() -> dict:
    """Build the canonical encounter-family scenarios (all in the 'test' split).

    Endpoints documented inline where land makes clearance non-obvious.
    """
    open_map = SeaMap((0., 0., 24., 24.))
    scenarios = {
        'head_on': _scenario(open_map, EGO_START, EGO_GOAL,
            [Traffic(EGO_GOAL, EGO_START)], 'head_on', 'test'),
        'crossing': _scenario(open_map, EGO_START, EGO_GOAL,
            [Traffic((12., 3.), (12., 21.))], 'crossing', 'test'),
        'crossing_mirrored': _scenario(open_map, EGO_START, EGO_GOAL,
            [Traffic((12., 21.), (12., 3.))], 'crossing_mirrored', 'test'),
        'overtake': _scenario(open_map, EGO_START, EGO_GOAL,
            [Traffic((7., 12.), (21., 12.), speed=.3)], 'overtake', 'test'),
        'multi_conflict': _scenario(open_map, EGO_START, EGO_GOAL,
            [Traffic((8., 3.), (8., 21.)), Traffic((16., 21.), (16., 3.))],
            'multi_conflict', 'test'),
    }

    # Gap between the two land blocks spans y in [10,14]; the shared y=12
    # centreline keeps 2m clearance to each block (well clear of the .6m
    # default traffic radius). The target transits the same gap in the
    # opposite direction to the ego.
    narrow_map = SeaMap((0., 0., 24., 24.), ((8., 0., 12., 10.), (8., 14., 12., 24.)))
    scenarios['narrow_passage'] = _scenario(narrow_map, EGO_START, EGO_GOAL,
        [Traffic(EGO_GOAL, EGO_START)], 'narrow_passage', 'test')

    # Harbour land occupies x in [9,14], y in [5,18]. Both targets run
    # east-west clear of that block: y=20 gives 2m clearance to its north
    # edge (y=18); y=4 gives 1m clearance to its south edge (y=5). Both
    # exceed the default .6m traffic radius.
    harbour_map = SeaMap.load(ROOT / 'maps/harbour.json')
    scenarios['detour_harbour'] = _scenario(harbour_map, (2., 12.), (22., 12.),
        [Traffic((2., 20.), (22., 20.)), Traffic((22., 4.), (2., 4.))],
        'detour_harbour', 'test')

    # Land spans the full map height, splitting it in two; no traffic needed
    # -- the point is that the goal is unreachable, not that traffic is dense.
    unreachable_map = SeaMap((0., 0., 24., 24.), ((10., 0., 14., 24.),))
    scenarios['unreachable'] = _scenario(unreachable_map, EGO_START, EGO_GOAL, [],
                                         'unreachable', 'test')

    # Goal sits .6m off the shore strip (x>=23.5): clear at the .5m ego
    # radius (.6 > .5) but tight enough that a planner may legitimately
    # report NoPath. That is the point of this family; it is not tuned away.
    shore_map = SeaMap((0., 0., 24., 24.), ((23.5, 0., 24., 24.),))
    scenarios['shore_goal'] = _scenario(shore_map, EGO_START, (22.9, 12.), [],
                                        'shore_goal', 'test')

    assert set(scenarios) == set(CANONICAL_FAMILIES)
    return scenarios


def generate(output_dir: Path) -> dict:
    """Write every canonical and seeded scenario plus splits.json into
    output_dir, and return the splits.json structure. Deterministic: two
    calls into different empty directories produce byte-identical files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list] = {}
    failures: list = []

    def write(name, scenario):
        try:
            validate(scenario)
        except ValueError as error:
            failures.append({'seed': scenario['seed'], 'split': scenario['split'], 'error': str(error)})
            return
        body = canonical_json(scenario)
        (output_dir / f'{name}.json').write_text(body)
        splits.setdefault(scenario['split'], []).append({
            'file': f'{name}.json',
            'sha256': sha256(body.encode()).hexdigest(),
            'scenario_hash': scenario_hash(scenario),
        })

    for family, scenario in sorted(canonical_scenarios().items()):
        write(family, scenario)

    for split, map_factory, start, goal, seeds, count in SEEDED_SPLITS:
        for seed in seeds:
            sea = map_factory()
            try:
                scenario = make_scenario(sea, start, goal, count, seed)
            except ValueError as error:
                failures.append({'seed': seed, 'split': split, 'error': str(error)})
                continue
            scenario['split'] = split
            write(f'{split}_{seed}', scenario)

    result = {'schema': 1, 'splits': splits,
              'withheld_families': sorted(CANONICAL_FAMILIES), 'failures': failures}
    (output_dir / 'splits.json').write_text(canonical_json(result))
    return result


if __name__ == '__main__':
    summary = generate(ROOT / 'scenarios')
    print(canonical_json(summary))
