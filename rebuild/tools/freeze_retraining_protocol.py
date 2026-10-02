"""Preregister the held-out retraining evaluation: write evidence/retraining/protocol.json.

Run once, after every training run has a selection.json and before any held-out
evaluation, from a clean checkout. Commit the output, then pass its commit and SHA256
to retrain_eval.py. Selected checkpoint directories are stored relative to rebuild/
when they live under it (portable); a checkpoint elsewhere keeps its absolute path.
"""
from datetime import datetime, timezone
from hashlib import sha256
from math import isfinite
from pathlib import Path
import argparse
import json
import os
import subprocess

from retrain_eval import COMPARATORS, LEARNED, arm_name, learned_variants
from shipnav.training.checkpoints import verify_checkpoint

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'evidence/retraining/protocol.json'
BASELINE_PROTOCOL = ROOT/'evidence/baseline/benchmark_protocol.json'
TRAINING_CONFIG = ROOT/'evidence/retraining/training_config.json'
MANIFEST = 'scenarios/baseline/splits.json'
SARL_DIR = ROOT.parent/'CrowdNav-20250813-DIP/crowd_nav/data/output_trained'
TOOLS = ('baseline_run', 'baseline_evidence', 'retrain_eval')
COMPARISONS = [
    ['ppo_s{s}_filtered', 'marine_mpc', 'learned_vs_mpc'],
    ['ppo_lagrangian_s{s}_filtered', 'ppo_s{s}_filtered', 'constraint_penalty'],
    ['ppo_s{s}_filtered', 'ppo_s{s}_unfiltered', 'PPO_filter'],
    ['ppo_s{s}_filtered', 'marine_sarl', 'new_vs_pretrained'],
    ['marine_mpc', 'marine_mpc_unfiltered', 'MPC_filter'],
]
INFERENCE_NOTE = ('learned arms: hierarchical_interval over training seeds then scenarios; fixed comparators '
                  'replicated per seed key; pairs of independently trained variants are matched by seed index '
                  '(arbitrary pairing, exploratory)')
LIMITATIONS = ('Southern Islands transfer is zero-shot (no real-map training data); the 99 canonical-family cases '
               'were inspected during development; this corpus previously evaluated fixed policies.')


def file_sha256(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(root=ROOT):
    """SHA256 of pyproject.toml, every src/**/*.py and the evaluation tools, keyed by rebuild-relative path."""
    names = ['pyproject.toml', *sorted(p.relative_to(root).as_posix() for p in (root/'src').rglob('*.py')),
             *(f'tools/{tool}.py' for tool in TOOLS)]
    return {name: file_sha256(root/name) for name in names}


def portable(path, root=ROOT):
    """Rebuild-relative POSIX path when `path` lies under `root`, else the absolute path (no symlink resolution)."""
    path = Path(os.path.normpath(root/path))  # an absolute `path` replaces `root`
    try:
        return path.relative_to(os.path.normpath(root)).as_posix()
    except ValueError:
        return path.as_posix()


def load_selection(root, variant, seed, verify=verify_checkpoint):
    """One run's chosen checkpoint: {'path', 'files'}, refusing a missing or inconsistent selection."""
    run = f'{variant}/seed{seed}'
    file = Path(root)/'results/train'/variant/f'seed{seed}'/'selection.json'
    if not file.is_file():
        raise ValueError(f'Missing selection.json for {run}: run select_checkpoints.py on every training run')
    selection = json.loads(file.read_text())
    if not isinstance(selection.get('path'), str) or not isinstance(selection.get('files'), dict):
        raise ValueError(f'Malformed selection.json for {run}')
    try:
        manifest = verify(Path(root)/selection['path'])
    except ValueError as error:
        raise ValueError(f'Selected checkpoint for {run} failed verification: {error}') from error
    if manifest['files'] != selection['files']:
        raise ValueError(f'Selected checkpoint for {run} differs from its selection.json file hashes')
    return {'path': portable(selection['path'], Path(root)), 'files': selection['files']}


def load_selections(root, training, verify=verify_checkpoint):
    """{(variant, seed): {'path', 'files'}} for every learned variant and training seed."""
    if tuple(training['variants']) != LEARNED:
        raise ValueError(f'Training config variants {training["variants"]} differ from {list(LEARNED)}')
    return {(v, s): load_selection(root, v, s, verify) for v in LEARNED for s in training['seeds']}


def build_protocol(*, baseline, manifest, manifest_sha256, training, selections, source_sha256, lock_sha256,
                   model_hashes, source_commit, frozen_utc):
    """Assemble the preregistration from already-loaded inputs; no file or git access."""
    seeds = training['seeds']
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or not all(isinstance(s, int) for s in seeds):
        raise ValueError('Retraining needs at least three distinct integer training seeds')
    if {s for _, s in selections} != set(seeds):
        raise ValueError('Selected checkpoint seeds differ from the training configuration seeds')
    budget = training['budget_domain_s']
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or not isfinite(budget):
        raise ValueError('Training configuration lacks a finite domain-cost budget')
    if baseline['manifest'] != MANIFEST or baseline['manifest_sha256'] != manifest_sha256:
        raise ValueError('Baseline protocol does not describe the scenario manifest being frozen')
    entries = manifest['splits']['test']
    if {e['scenario_hash'] for e in entries} != set(baseline['shared_limits_model']):
        raise ValueError('Baseline shared limits do not cover exactly the test scenarios')
    if baseline['identities']['model'] != model_hashes:
        raise ValueError('SARL checkpoint hashes differ from the baseline protocol')
    variants = {**COMPARATORS, **learned_variants({k: v['path'] for k, v in selections.items()})}
    return {
        'schema': 1,
        'name': 'preregistered held-out evaluation of retrained PPO against re-run matched comparators',
        'frozen_utc': frozen_utc, 'source_commit': source_commit, 'source_sha256': source_sha256,
        'manifest': MANIFEST, 'manifest_sha256': manifest_sha256,
        'original_manifest': baseline['original_manifest'],
        'original_manifest_sha256': baseline['original_manifest_sha256'],
        'scenarios': {'test': entries},
        'shared_limits_model': baseline['shared_limits_model'],
        'variants': variants,
        'identities': {'lock_sha256': lock_sha256, 'model': model_hashes,
                       'learned': {arm_name(v, s, f): selections[v, s]['files']
                                   for v, s in sorted(selections) for f in (True, False)}},
        'training': {**training, 'seeds': list(seeds)},
        'comparisons': COMPARISONS,
        'targets': {'success_rate_minimum': .8, 'collision_rate_maximum': .05, 'constraint_cost_budget': budget},
        'inference': {**baseline['inference'], 'learned_arms': INFERENCE_NOTE},
        'multiplicity': 'exploratory, unadjusted',
        'limitations': LIMITATIONS,
    }


def dirty_paths(root=ROOT):
    """Tracked changes anywhere in rebuild/, and untracked files under the frozen inputs."""
    tracked = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no', '--', '.'],
                                      cwd=root, text=True)
    untracked = subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'tools', 'scenarios', 'evidence',
                                         'pyproject.toml', 'uv.lock'], cwd=root, text=True)
    return sorted({line for line in (tracked+untracked).splitlines() if line})


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if OUTPUT.exists():
        raise SystemExit(f'{OUTPUT.relative_to(ROOT)} already exists; a frozen protocol is never overwritten')
    if not TRAINING_CONFIG.is_file():
        raise SystemExit(f'{TRAINING_CONFIG.relative_to(ROOT)} is missing: freeze the training configuration '
                         'with tools/freeze_training_config.py and train before preregistering the evaluation')
    dirty = dirty_paths()
    if dirty:
        raise SystemExit('Commit or discard changes before freezing (clean tree required):\n'+'\n'.join(dirty))
    training = json.loads(TRAINING_CONFIG.read_text())
    baseline = json.loads(BASELINE_PROTOCOL.read_text())
    try:
        selections = load_selections(ROOT, training)
        model_hashes = {name: file_sha256(SARL_DIR/name) for name in baseline['identities']['model']
                        if (SARL_DIR/name).is_file()}
        protocol = build_protocol(
            baseline=baseline, manifest=json.loads((ROOT/MANIFEST).read_text()),
            manifest_sha256=file_sha256(ROOT/MANIFEST), training=training, selections=selections,
            source_sha256=source_hashes(), lock_sha256=file_sha256(ROOT/'uv.lock'),
            model_hashes=model_hashes,
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            frozen_utc=datetime.now(timezone.utc).isoformat())
    except ValueError as error:
        raise SystemExit(f'Refusing to freeze: {error}') from error
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(protocol, indent=2, allow_nan=False)+'\n')
    print(f'Wrote {OUTPUT.relative_to(ROOT)}; sha256 {file_sha256(OUTPUT)}')


if __name__ == '__main__':
    main()
