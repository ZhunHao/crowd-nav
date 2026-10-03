"""Preregister training settings and the domain-cost budget before full training.

The budget comes from marine_mpc on the generated *selection* split, never the
held-out corpus. Commit the output before launching any full training run.
"""
from collections import Counter
from pathlib import Path
from statistics import mean
import argparse
import json
import subprocess

from shipnav.metrics import metrics
from shipnav.scale import map_options
from shipnav.service import execute
from shipnav.training.scenarios import load_split
from shipnav.training.train import ROLLOUT
from training_identity import dirty_paths

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'evidence/retraining/training_config.json'
CHUNK_STEPS = 20480
# Everything the training runs execute or are checked against: tracked changes and untracked files alike.
FROZEN_INPUTS = ('src', 'tools', 'cloud', 'scenarios/training', 'pyproject.toml', 'uv.lock')


def check_arguments(chunks, envs, seeds):
    """Reject settings the trainer would refuse (same rule as shipnav.training.train.parse)."""
    if seeds < 3:
        raise SystemExit('At least three independent training seeds are required')
    if chunks < 1 or envs < 1:
        raise SystemExit('--chunks and --envs must be at least 1')
    rollout = min(ROLLOUT, CHUNK_STEPS)
    if rollout % envs or CHUNK_STEPS % rollout:
        raise SystemExit(f'--envs must divide the trainer rollout of {rollout} steps exactly')


def domain_budget(scenarios):
    """Mean marine_mpc domain time over `scenarios`, with the status counts that make it auditable.

    A run without domain-time data is an error: substituting zero would silently lower the frozen budget."""
    exposures, statuses = [], Counter()
    for scenario in scenarios:
        run = execute(scenario['map'], scenario['start'], scenario['goal'], policy_name='mpc', scenario=scenario,
                      dynamics='marine', filtered=True, **map_options(scenario['map']))
        exposure = metrics(run)['domain_time_s']
        if exposure is None:
            raise SystemExit(f"marine_mpc run without domain_time_s data (status {run['status']}); "
                             'refusing to freeze a budget that would count it as zero exposure')
        exposures.append(exposure)
        statuses[run['status']] += 1
    if not exposures:
        raise SystemExit('The select split is empty; cannot compute a budget')
    return round(mean(exposures), 3), len(exposures), dict(sorted(statuses.items()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--chunks', type=int, required=True)
    p.add_argument('--envs', type=int, required=True)
    p.add_argument('--seeds', type=int, default=5)
    a = p.parse_args()
    check_arguments(a.chunks, a.envs, a.seeds)
    if OUTPUT.exists():
        raise SystemExit(f'{OUTPUT.relative_to(ROOT)} already exists; a frozen configuration is never overwritten')
    dirty = dirty_paths(ROOT, FROZEN_INPUTS)
    if dirty:
        raise SystemExit('Commit source, tools, cloud scripts, corpus and lockfile before freezing the training '
                         'configuration (clean tree required):\n'+'\n'.join(dirty))
    budget, budget_n, mpc_statuses = domain_budget(load_split(ROOT/'scenarios/training', 'select'))
    corpus = json.loads((ROOT/'scenarios/training/manifest.json').read_text())
    config = {
        'schema': 1, 'seeds': list(range(a.seeds)), 'variants': ['ppo', 'ppo_lagrangian'],
        'chunks': a.chunks, 'chunk_steps': CHUNK_STEPS, 'envs': a.envs, 'filtered': True, 'observation': {},
        'budget_domain_s': budget, 'budget_n_scenarios': budget_n,
        'budget_mpc_status_counts': mpc_statuses, 'budget_source': 'marine_mpc mean domain_time_s on select split',
        'selection': {'split': 'select', 'every_chunks': 5, 'include_last': True, 'collision_max': .05,
                      'rule': 'max success among collision<=max; else min collision then max success; '
                              'ties: lower mean domain_time_s, then earlier chunk'},
        'corpus_sha256': {k: v['sha256'] for k, v in corpus['splits'].items()},
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(config, indent=2, allow_nan=False)+'\n')
    print(json.dumps(config, indent=2))
