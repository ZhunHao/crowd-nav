"""Write scenarios/training/ deterministically; failed seeds are recorded, never retried."""
from pathlib import Path

from shipnav.planning import InvalidRoute, NoPath, PlanningLimit
from shipnav.scenarios import scenario_hash
from shipnav.training.scenarios import SELECT_SEEDS, TRAIN_SEEDS, frozen_identities, generate, write_corpus

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    hashes, seeds = frozen_identities(ROOT)
    if (set(TRAIN_SEEDS) | set(SELECT_SEEDS)) & seeds:
        raise SystemExit('Generated seed range overlaps a frozen manifest')
    splits, failures = {'train': [], 'select': []}, []
    for split, seed_range in (('train', TRAIN_SEEDS), ('select', SELECT_SEEDS)):
        for seed in seed_range:
            try:
                scenario = generate(seed, split)
            except (NoPath, PlanningLimit, InvalidRoute, ValueError) as error:
                failures.append({'seed': seed, 'split': split, 'error': f'{type(error).__name__}: {error}'})
                continue
            if scenario_hash(scenario) in hashes:
                raise SystemExit(f'Seed {seed} reproduces a frozen scenario')
            splits[split].append(scenario)
    manifest = write_corpus(ROOT/'scenarios/training', splits, failures)
    print({k: v['count'] for k, v in manifest['splits'].items()}, 'failures', len(failures))
