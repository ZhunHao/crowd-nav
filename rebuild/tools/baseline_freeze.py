"""Extend historical frozen splits without rewriting their bytes or hash identities.

References are geometric A*-smoothed routes frozen before any policy runs; each
variant receives identical serialized scenarios. A null reference explicitly
represents the deliberately disconnected family, never a fabricated safe route.
"""
from hashlib import sha256
from math import dist
from pathlib import Path
import json

from baseline_evidence import archive, perturb
from freeze_scenarios import validate, encounter_errors
from shipnav.maps import SeaMap, canonical_json
from shipnav.planning import NoPath, astar, smooth
from shipnav.scale import PROFILES, map_options, to_model
from shipnav.scenarios import make_scenario, scenario_hash, _along

ROOT = Path(__file__).resolve().parents[1]


def reference(scenario):
    sea = SeaMap.from_dict(scenario['map'])
    opts = map_options(scenario['map'])
    clearance, resolution = opts.get('clearance', .7), opts.get('resolution', 1.)
    try:
        return smooth(sea, astar(sea, tuple(scenario['start']), tuple(scenario['goal']), clearance, resolution), clearance)
    except NoPath:
        if scenario['family'] != 'unreachable':
            raise
        return None


def real_cases(map_name, case_id, n, split, seed_base):
    from shipnav.maps import LocalFrame
    case = next(c for c in json.loads((ROOT/'maps/benchmarks.json').read_text())['cases'] if c['id'] == case_id)
    original = SeaMap.load(ROOT/'maps'/map_name)
    frame = LocalFrame(*original.to_dict()['metadata']['frame']['origin_lonlat'])
    sea = to_model(original, PROFILES['harbour_craft'])
    start, goal = [tuple(v/10 for v in frame.project(*case[k])) for k in ('start', 'goal')]
    whole = reference({'map': sea.to_dict(), 'start': start, 'goal': goal, 'family': 'corridor'})
    length = sum(dist(a, b) for a, b in zip(whole, whole[1:]))
    # Short 350–550 m local voyages sampled along the geographic reference.
    # Endpoints are fixed by arc length, without querying any controller outcome.
    for i in range(n):
        distance = 35+5*(i % 5)
        offset = (length-distance)*(i+1)/(n+1)
        a, _ = _along(whole, offset)
        b, _ = _along(whole, offset+distance)
        route = reference({'map': sea.to_dict(), 'start': a, 'goal': b, 'family': 'corridor'})
        scenario = make_scenario(sea, a, b, 1+i % 3, seed_base+i, corridor=route)
        scenario.update(split=split, family='southern_corridor' if split == 'test' else 'ubin_corridor')
        yield f'{split}_{scenario["family"]}_{seed_base+i}', scenario


def generate(output, real_maps=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    old = json.loads((ROOT/'scenarios/splits.json').read_text())
    result = {'schema': 1, 'original_splits_sha256': sha256((ROOT/'scenarios/splits.json').read_bytes()).hexdigest(),
              'splits': {'dev': [], 'calibration': [], 'test': []}, 'failures': []}
    seen = set()

    def write(name, scenario, historical=False):
        validate(scenario)
        if scenario['family'] not in ('random', 'ubin_corridor', 'southern_corridor'):
            errors = encounter_errors(scenario)
            if errors:
                raise ValueError(f'{name}: {errors}')
        physical = {k: scenario[k] for k in ('map', 'start', 'goal', 'traffic')}
        identity = sha256(canonical_json(physical).encode()).hexdigest()
        if identity in seen:
            raise ValueError('Duplicate physical scenario: '+name)
        seen.add(identity)
        path = output/(name+'.json')
        path.write_text(canonical_json(scenario))
        entry = archive(path)
        entry.update(id=name, scenario_hash=scenario_hash(scenario), physical_sha256=identity,
                     family=scenario['family'], seed=scenario['seed'], traffic_count=len(scenario['traffic']),
                     reference_route=reference(scenario), historical_regression=historical)
        result['splits'][scenario['split']].append(entry)

    for split in ('dev', 'calibration', 'test'):
        for entry in old['splits'][split]:
            scenario = json.loads((ROOT/'scenarios'/entry['file']).read_text())
            name = Path(entry['file']).stem
            write(name, scenario, True)
            if split == 'test':
                for i in range(1, 9):
                    altered = perturb(scenario, i)
                    # Disjoint seeded perturbation identities across families.
                    altered['seed'] += 100*len(result['splits']['test'])
                    write(f'{name}_variation_{i:02d}', altered)
    if real_maps:
        for args in [('singapore-ubin.json', 'ubin-detour', 2, 'dev', 6000),
                     ('singapore-ubin.json', 'ubin-detour', 2, 'calibration', 7000),
                     ('singapore-southern-islands.json', 'south-island-detour', 11, 'test', 8000)]:
            for name, scenario in real_cases(*args):
                write(name, scenario)
    (output/'splits.json').write_text(canonical_json(result))
    return result


if __name__ == '__main__':
    data = generate(ROOT/'scenarios/baseline')
    print({k: len(v) for k, v in data['splits'].items()})
