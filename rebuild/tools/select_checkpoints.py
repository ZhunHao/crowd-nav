"""Pick one checkpoint per training run using only the generated selection split."""
from pathlib import Path
from statistics import mean
import argparse
import json
import os

from shipnav.metrics import metrics
from shipnav.scale import map_options
from shipnav.service import execute
from shipnav.training.checkpoints import IncompleteCheckpointError, verify_checkpoint
from shipnav.training.scenarios import load_split
from training_identity import final_chunk_problem, manifest_problems, mixed_identity, portable

ROOT = Path(__file__).resolve().parents[1]


def choose(rows, collision_max):
    if not rows:
        raise ValueError('No candidate checkpoints to choose from')
    safe = [r for r in rows if r['collision_rate'] <= collision_max]
    if safe:
        return min(safe, key=lambda r: (-r['success_rate'], r['mean_domain_s'], r['chunk']))
    return min(rows, key=lambda r: (r['collision_rate'], -r['success_rate'], r['mean_domain_s'], r['chunk']))


def load_manifests(directories, verify=verify_checkpoint):
    """Verify each folder; return ({path: manifest}, names of incomplete folders that were skipped).

    An incomplete (half-synced) folder is listed, not used. A hash mismatch propagates."""
    manifests, skipped = {}, []
    for directory in directories:
        try:
            manifests[directory] = verify(directory)
        except IncompleteCheckpointError:
            skipped.append(Path(directory).name)
    return manifests, skipped


def pick_checkpoints(manifests, every_chunks, include_last):
    """Complete checkpoints on every `every_chunks`-th chunk, plus the last complete one if requested.

    Ordered by the manifest's logical chunk; partial checkpoints are never picked."""
    if every_chunks < 1:
        raise ValueError('every_chunks must be at least 1')
    complete = sorted((p for p, m in manifests.items() if m['complete_chunk']), key=lambda p: manifests[p]['chunk'])
    picked = [p for p in complete if manifests[p]['chunk'] % every_chunks == every_chunks-1]
    if include_last and complete and complete[-1] not in picked:
        picked.append(complete[-1])
    return picked


def evaluate(path, scenarios, filtered):
    if not scenarios:
        raise ValueError('The selection split is empty; cannot evaluate a checkpoint')
    outcomes = []
    for scenario in scenarios:
        run = execute(scenario['map'], scenario['start'], scenario['goal'], policy_name='ppo',
                      learned_dir=str(path), scenario=scenario, dynamics='marine', filtered=filtered,
                      **map_options(scenario['map']))
        domain = metrics(run)['domain_time_s']
        if domain is None:
            raise ValueError(f'{path}: run without domain_time_s data (status {run["status"]}); '
                             'refusing to count it as zero exposure')
        outcomes.append((run['status'], domain))
    n = len(outcomes)
    return {'success_rate': sum(s == 'success' for s, _ in outcomes)/n,
            'collision_rate': sum(s == 'collision' for s, _ in outcomes)/n,
            'mean_domain_s': mean(d for _, d in outcomes)}


def candidate_dirs(run):
    return sorted(d for d in Path(run).iterdir()
                  if d.is_dir() and not d.name.startswith('.') and (d/'manifest.json').is_file())


def select(run, config, root=ROOT, load=load_split, assess=evaluate):
    """Choose one checkpoint of `run` on the selection split and write run/selection.json.

    Refuses a run directory whose checkpoints mix identities, that does not belong to the frozen training
    configuration, or that has not reached the configured final chunk. `path` is stored relative to `root`."""
    run = Path(run)
    rule = config['selection']
    manifests, skipped = load_manifests(candidate_dirs(run))
    if not manifests:
        raise ValueError(f'No checkpoints in {run}')
    mixed = mixed_identity(manifests)
    if mixed:
        raise ValueError(f'{run} mixes checkpoints of different runs: {", ".join(mixed)} differ')
    picked = pick_checkpoints(manifests, rule['every_chunks'], rule['include_last'])
    if not picked:
        raise ValueError(f'No complete checkpoints to select from in {run}')
    slot = Path(os.path.abspath(run)).parts[-2:]  # results/train/<variant>/seed<k>: the slot names the run
    named = slot[0] in ('ppo', 'ppo_lagrangian') and slot[1][4:].isdigit() and slot[1].startswith('seed')
    problems = [f'{d.name}: {problem}' for d in picked
                for problem in manifest_problems(manifests[d], config, slot[0] if named else manifests[d].get('variant'),
                                                 int(slot[1][4:]) if named else manifests[d].get('seed'))]
    final = final_chunk_problem(run, config)
    if final:
        problems.append(final)
    if problems:
        raise ValueError(f'{run} does not match the frozen training configuration: '+'; '.join(problems))
    scenarios = load(root/'scenarios/training', rule['split'])
    rows = [{'chunk': manifests[d]['chunk'], 'path': portable(os.path.abspath(d), root), 'files': manifests[d]['files'],
             **assess(d, scenarios, config['filtered'])} for d in picked]
    best = choose(rows, rule['collision_max'])
    (run/'selection.json').write_text(
        json.dumps({**best, 'candidates': rows, 'skipped_incomplete': skipped}, indent=2)+'\n')
    return best


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    a = p.parse_args()
    try:
        best = select(a.run, json.loads((ROOT/'evidence/retraining/training_config.json').read_text()))
    except ValueError as error:
        raise SystemExit(f'Refusing to select: {error}') from error
    print(json.dumps(best, indent=2))
