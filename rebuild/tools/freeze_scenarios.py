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
from hashlib import sha256
from math import dist
from pathlib import Path

from shipnav.maps import SeaMap, canonical_json
from shipnav.planning import NoPath, astar, smooth
from shipnav.policies import Direct
from shipnav.scenarios import load_traffic, make_scenario, scenario_hash, traffic_to_dict
from shipnav.simulation import CourseChangeTraffic, Traffic, run_episode

ROOT = Path(__file__).resolve().parents[1]

CANONICAL_FAMILIES = ('head_on', 'crossing', 'crossing_mirrored', 'overtake',
                      'narrow_passage', 'detour_harbour', 'unreachable',
                      'shore_goal', 'multi_conflict', 'course_change', 'reactive')

EGO_START, EGO_GOAL = (3., 12.), (21., 12.)

# Nominal ego used by the freeze-time encounter checks: a holonomic Direct ego
# following the astar_smooth route at the service's default speed and radius.
EGO_RADIUS, EGO_SPEED = .5, 1.
# An encounter family must bring the nominal ego and its target(s) within
# (ego radius + target radius) + CPA_MARGIN of each other (centre distance).
CPA_MARGIN = .5
CPA_SAMPLE_S = .01
NARROW_GAP_X = (8., 12.)
CROSSING_POINT = (12., 12.)
HARBOUR_X = (9., 14.)

# split -> (map factory, ego start, ego goal, seed range, traffic count)
SEEDED_SPLITS = (
    ('train', lambda: SeaMap((0., 0., 24., 24.)),
     (2., 2.), (22., 22.), range(1000, 1010), 4),
    ('dev', lambda: SeaMap((0., 0., 24., 24.), ((4., 14., 10., 20.), (14., 4., 20., 10.))),
     (2., 2.), (22., 22.), range(2000, 2005), 3),
    ('calibration', lambda: SeaMap.load(ROOT / 'maps/harbour.json'),
     (2., 2.), (22., 22.), range(3000, 3005), 3),
)


def _scenario(sea, start, goal, traffic, family, split, seed=0, traffic_mode='scripted'):
    return {'schema': 1, 'seed': seed, 'map': sea.to_dict(), 'start': list(start),
            'goal': list(goal), 'traffic': [traffic_to_dict(s) for s in traffic],
            'traffic_mode': traffic_mode, 'family': family, 'split': split}


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


def planned_route(scenario):
    """The route the service would plan (`astar_smooth`); raises NoPath."""
    sea = SeaMap.from_dict(scenario['map'])
    return smooth(sea, astar(sea, tuple(scenario['start']), tuple(scenario['goal'])))


def _ego_at(route, t):
    travelled = t*EGO_SPEED
    for a, b in zip(route, route[1:]):
        length = dist(a, b)
        if travelled <= length:
            f = travelled/length if length else 1.
            return tuple(x+f*(y-x) for x, y in zip(a, b))
        travelled -= length
    return tuple(route[-1])


def nominal_cpa(route, ship):
    """Closest point of approach between the nominal ego (straight segments of
    `route` at EGO_SPEED from t=0 until arrival) and a target's frozen voyage
    (`ship.at`, i.e. the scripted or pre-reaction path). Returns the centre
    distance, the time and both positions at the CPA."""
    duration = sum(dist(a, b) for a, b in zip(route, route[1:]))/EGO_SPEED
    best = None
    for i in range(int(duration/CPA_SAMPLE_S)+1):
        t = i*CPA_SAMPLE_S
        ego, other = _ego_at(route, t), ship.at(t)[0]
        gap = dist(ego, other)
        if best is None or gap < best['distance']:
            best = {'distance': gap, 't': t, 'ego': ego, 'target': tuple(other)}
    return best


def reaction_fires(scenario):
    """Run a Direct ego against the reactive wrapping of the frozen voyage and
    report whether the target's reaction rule ever overrides its nominal
    (toward-goal) velocity."""
    from shipnav.reactive import ReactiveTraffic
    sea = SeaMap.from_dict(scenario['map'])
    targets = [ReactiveTraffic(ship, sea) for ship in load_traffic(scenario)]
    run_episode(sea, planned_route(scenario), targets, Direct(), radius=EGO_RADIUS, speed=EGO_SPEED)
    dt = .25
    for target in targets:
        for (_, p, _), (_, _, v) in zip(target.history, target.history[1:]):
            d = dist(p, target.goal)
            nominal = tuple((b-a)/d*min(target.speed, d/dt) if d else 0. for a, b in zip(p, target.goal))
            if v != (0., 0.) and dist(v, nominal) > 1e-9:
                return True
    return False


def planning_errors(scenario):
    """Every family except `unreachable` must plan with astar_smooth;
    `unreachable` must raise NoPath."""
    try:
        planned_route(scenario)
    except NoPath as error:
        return [] if scenario['family'] == 'unreachable' else [f'astar_smooth found no route: {error}']
    return ['unreachable family unexpectedly planned a route'] if scenario['family'] == 'unreachable' else []


def _within(value, bounds):
    return bounds[0] <= value <= bounds[1]


def encounter_errors(scenario):
    """Freeze-time validity of an encounter family's timing (empty if valid or
    not an encounter family). Never adjusts the scenario."""
    family = scenario['family']
    ships = load_traffic(scenario)
    if family in ('unreachable', 'shore_goal', 'random'):
        return []
    route = planned_route(scenario)
    cpas = [nominal_cpa(route, ship) for ship in ships]
    near = [c['distance'] < EGO_RADIUS+ship.radius+CPA_MARGIN for c, ship in zip(cpas, ships)]
    errors = []
    if family == 'multi_conflict' and not all(near):
        errors.append('multi_conflict: not every target reaches the CPA threshold')
    if not any(near):
        closest = min(c['distance'] for c in cpas)
        errors.append(f'{family}: nominal CPA {closest:.2f} m is not below the encounter threshold')
    first = min(cpas, key=lambda c: c['distance'])
    if family in ('crossing', 'crossing_mirrored') and dist(first['ego'], CROSSING_POINT) > 1.:
        errors.append(f'{family}: CPA at {first["ego"]} is not at the crossing {CROSSING_POINT}')
    if family == 'narrow_passage' and not _within(first['ego'][0], NARROW_GAP_X):
        errors.append(f'narrow_passage: conflict x={first["ego"][0]:.2f} is outside the gap {NARROW_GAP_X}')
    if family == 'detour_harbour' and not _within(first['ego'][0], HARBOUR_X):
        errors.append(f'detour_harbour: conflict x={first["ego"][0]:.2f} is not on the detour past the harbour')
    if family == 'course_change' and first['t'] < ships[0].waypoints[1][0]:
        errors.append('course_change: CPA occurs before the turn')
    if family == 'reactive' and not reaction_fires(scenario):
        errors.append('reactive: the reaction never fires against a Direct ego')
    return errors


def canonical_scenarios() -> dict:
    """Build the canonical encounter-family scenarios (all in the 'test' split).

    Timing: the nominal ego (holonomic Direct on the astar_smooth route at
    1 m/s) is at x = 3 + t on the default (3,12)->(21,12) route. Where the plan
    fixes a family's endpoints (head_on, crossing, crossing_mirrored, overtake,
    multi_conflict) the conflict is timed by target speed; where it does not
    (narrow_passage, detour_harbour, course_change, reactive) it is timed by the
    target's start position. `encounter_errors` checks every family at freeze
    time and records a failure instead of adjusting anything.
    """
    open_map = SeaMap((0., 0., 24., 24.))
    scenarios = {
        # Same line, opposite directions: meets at x = 3+18/1.3 ~ 16.8 at .3 m/s.
        'head_on': _scenario(open_map, EGO_START, EGO_GOAL,
            [Traffic(EGO_GOAL, EGO_START)], 'head_on', 'test'),
        # Ego reaches x=12 at t=9; the target covers the 9 m to y=12 in 9 s.
        'crossing': _scenario(open_map, EGO_START, EGO_GOAL,
            [Traffic((12., 3.), (12., 21.), speed=1.)], 'crossing', 'test'),
        'crossing_mirrored': _scenario(open_map, EGO_START, EGO_GOAL,
            [Traffic((12., 21.), (12., 3.), speed=1.)], 'crossing_mirrored', 'test'),
        # Plan-specified .3 m/s: the ego catches the target at x ~ 8.7.
        'overtake': _scenario(open_map, EGO_START, EGO_GOAL,
            [Traffic((7., 12.), (21., 12.), speed=.3)], 'overtake', 'test'),
        # Ego at x=8 at t=5 (target covers 9 m: 1.8 m/s) and at x=16 at t=13
        # (9 m in 13 s: 9/13 m/s). Both crossings are on a collision course.
        'multi_conflict': _scenario(open_map, EGO_START, EGO_GOAL,
            [Traffic((8., 3.), (8., 21.), speed=1.8), Traffic((16., 21.), (16., 3.), speed=9/13)],
            'multi_conflict', 'test'),
    }

    # Gap between the two land blocks spans y in [10,14] for x in [8,12]; the
    # shared y=12 centreline keeps 2m clearance to each block. The target
    # starts just east of the gap at .3 m/s so both meet at x=10 (t=7), inside
    # the passage.
    narrow_map = SeaMap((0., 0., 24., 24.), ((8., 0., 12., 10.), (8., 14., 12., 24.)))
    scenarios['narrow_passage'] = _scenario(narrow_map, EGO_START, EGO_GOAL,
        [Traffic((12.1, 12.), EGO_START)], 'narrow_passage', 'test')

    # Harbour land occupies x in [9,14], y in [5,18]; the planned ego route
    # detours north along y=19.5 for x in [9.5,13.5] (t ~ 10.6..14.6). The
    # northern target runs east along y=20 (2m clear of the block) at .3 m/s
    # from x=7.7, so the ego comes up on it alongside the harbour (x ~ 11.5).
    # The southern target (y=4, 1m clear of the block) is background traffic.
    harbour_map = SeaMap.load(ROOT / 'maps/harbour.json')
    scenarios['detour_harbour'] = _scenario(harbour_map, (2., 12.), (22., 12.),
        [Traffic((7.7, 20.), (22., 20.)), Traffic((22., 4.), (2., 4.))],
        'detour_harbour', 'test')

    # Land spans the full map height, splitting it in two; no traffic needed
    # -- the point is that the goal is unreachable, not that traffic is dense.
    unreachable_map = SeaMap((0., 0., 24., 24.), ((10., 0., 14., 24.),))
    scenarios['unreachable'] = _scenario(unreachable_map, EGO_START, EGO_GOAL, [],
                                         'unreachable', 'test')

    # Goal (22.9,12) sits .8m off the shore strip (x>=23.7): above the
    # planner's .7m clearance, so astar_smooth plans and a genuine near-shore
    # approach runs (checked at freeze time), still tight for a vessel that
    # arrives carrying way.
    shore_map = SeaMap((0., 0., 24., 24.), ((23.7, 0., 24., 24.),))
    scenarios['shore_goal'] = _scenario(shore_map, EGO_START, (22.9, 12.), [],
                                        'shore_goal', 'test')

    # Scripted course-change target: north along x=12 at .3 m/s from
    # (12,10.2), turning at (12,12) at t=6 to head east into the ego's lane
    # (again .3 m/s) until (18,12) at t=26. The ego passes x=12 at t=9, after
    # the turn, and runs into the target ahead of it (CPA after the turn). This
    # is deterministic, waypoint-scripted motion (`CourseChangeTraffic`), not
    # the reactive rule below.
    scenarios['course_change'] = _scenario(open_map, EGO_START, EGO_GOAL,
        [CourseChangeTraffic(((0., (12., 10.2)), (6., (12., 12.)), (26., (18., 12.))))],
        'course_change', 'test')

    # Non-cooperative reactive target: a fixed initial voyage frozen exactly
    # like any other `Traffic`, but tagged `traffic_mode: 'reactive'` so the
    # service wraps it in `ReactiveTraffic` at run time. Timed by start
    # position: at .3 m/s from (14,8.7) the nominal voyage reaches y=12 at
    # t=11, when the ego passes x=14; the freeze-time check confirms the
    # reaction actually fires against a Direct ego. The frozen scenario never
    # realizes the reactive path -- only the initial conditions.
    scenarios['noncooperative_reactive'] = _scenario(open_map, EGO_START, EGO_GOAL,
        [Traffic((14., 8.7), (14., 21.))], 'reactive', 'test', traffic_mode='reactive')

    assert {scenario['family'] for scenario in scenarios.values()} == set(CANONICAL_FAMILIES)
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
        if scenario['split'] == 'test':
            errors = planning_errors(scenario) or encounter_errors(scenario)
            if errors:
                failures.append({'name': name, 'seed': scenario['seed'], 'split': scenario['split'],
                                 'error': '; '.join(errors)})
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
