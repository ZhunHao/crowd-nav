"""Capture frozen scenario behavior across controllers, filters, and dynamics.

Timing and provenance are intentionally excluded: they reflect host load/Git state,
not simulation behavior. All frame and filter-decision values remain exact.
Run from rebuild: .venv-modern/bin/python tools/capture_scale_parity.py --output results/parity-before.json
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def capture(job):
    import torch
    torch.set_num_threads(1)
    from shipnav.planning import NoPath
    from shipnav.scenarios import scenario_hash
    from shipnav.service import execute
    filename, policy, filtered, dynamics = job
    scenario = json.loads(Path(filename).read_text())
    row = {'id': f'{Path(filename).stem}/{policy}/{filtered}/{dynamics}',
           'scenario_hash': scenario_hash(scenario), 'map_hash': digest(scenario['map'])}
    try:
        run = execute(scenario['map'], tuple(scenario['start']), tuple(scenario['goal']),
                      str(ROOT.parent/'CrowdNav-20250813-DIP/crowd_nav/data/output_trained'),
                      policy_name=policy, filtered=filtered, dynamics=dynamics, scenario=scenario)
    except NoPath as error:
        row['error'] = {'type': type(error).__name__, 'message': str(error)}
    else:
        row['result'] = {k: v for k, v in run.items() if k not in
                         ('inference_ms', 'provenance', 'settings', 'map', 'scenario', 'traffic_definitions')}
        row['result']['diagnostics'] = [{k: v for k, v in d.items() if k not in
                                        ('decision_ms', 'deadline_miss')} for d in run['diagnostics']]
    row['behavior_sha256'] = digest(row)
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    scenes = [str(p) for p in sorted((ROOT/'scenarios').glob('*.json')) if p.name != 'splits.json']
    jobs = list(product(scenes, ('direct', 'mpc', 'sarl', 'orca'), (False, True), ('holonomic', 'marine')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, args.output.open('w') as handle:
        handle.write('[\n')
        for i, row in enumerate(pool.map(capture, jobs)):
            if i:
                handle.write(',\n')
            handle.write(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False))
            handle.flush()
            print(f'{i+1}/{len(jobs)} {row["id"]} {row.get("result", {}).get("status", "no_path")}', flush=True)
        handle.write('\n]\n')
