"""Preregister training settings and the domain-cost budget before full training.

The budget comes from marine_mpc on the generated *selection* split, never the
held-out corpus. Commit the output before launching any full training run.
"""
from pathlib import Path
from statistics import mean
import argparse
import json
import subprocess

from shipnav.metrics import metrics
from shipnav.scale import map_options
from shipnav.service import execute
from shipnav.training.scenarios import load_split

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--chunks', type=int, required=True)
    p.add_argument('--envs', type=int, required=True)
    p.add_argument('--seeds', type=int, default=5)
    a = p.parse_args()
    if a.seeds < 3:
        raise SystemExit('At least three independent training seeds are required')
    if subprocess.check_output(['git', 'status', '--porcelain', 'src', 'scenarios/training'], cwd=ROOT).strip():
        raise SystemExit('Commit source and corpus before freezing the training configuration')
    exposures = []
    for scenario in load_split(ROOT/'scenarios/training', 'select'):
        run = execute(scenario['map'], scenario['start'], scenario['goal'], policy_name='mpc', scenario=scenario,
                      dynamics='marine', filtered=True, **map_options(scenario['map']))
        exposures.append(metrics(run)['domain_time_s'] or 0.)
    corpus = json.loads((ROOT/'scenarios/training/manifest.json').read_text())
    config = {
        'schema': 1, 'seeds': list(range(a.seeds)), 'variants': ['ppo', 'ppo_lagrangian'],
        'chunks': a.chunks, 'chunk_steps': 20480, 'envs': a.envs, 'filtered': True, 'observation': {},
        'budget_domain_s': round(mean(exposures), 3), 'budget_source': 'marine_mpc mean domain_time_s on select split',
        'selection': {'split': 'select', 'every_chunks': 5, 'include_last': True, 'collision_max': .05,
                      'rule': 'max success among collision<=max; else min collision then max success; '
                              'ties: lower mean domain_time_s, then earlier chunk'},
        'corpus_sha256': {k: v['sha256'] for k, v in corpus['splits'].items()},
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    out = ROOT/'evidence/retraining/training_config.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2, allow_nan=False)+'\n')
    print(json.dumps(config, indent=2))
