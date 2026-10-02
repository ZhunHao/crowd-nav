"""Held-out evaluation of selected PPO checkpoints against re-run matched comparators.

Never run heldout before committing evidence/retraining/protocol.json; its commit
and SHA256 are verified exactly as for the pre-retraining baseline.
"""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import subprocess

from baseline_run import (ROOT, bind_output, dump, execute_matrix, summary, validate_rows,
                          verify_frozen_identities)
from shipnav.benchmark import VARIANTS
from shipnav.metrics import hierarchical_interval

COMPARATORS = {name: VARIANTS[name] for name in ('marine_mpc', 'marine_mpc_unfiltered', 'marine_sarl')}
LEARNED = ('ppo', 'ppo_lagrangian')


def arm_name(variant, seed, filtered):
    return f'{variant}_s{seed}_{"filtered" if filtered else "unfiltered"}'


def learned_variants(selections):
    """selections: {(variant, seed): checkpoint directory} -> runner variant table."""
    seeds = {seed for _, seed in selections}
    if {(v, s) for v in LEARNED for s in seeds} != set(selections):
        raise ValueError('Every learned variant needs a selected checkpoint for every seed')
    return {arm_name(v, s, f): {'policy_name': 'ppo', 'learned_dir': selections[v, s], 'dynamics': 'marine',
                                'filtered': f}
            for v in LEARNED for s in sorted(seeds) for f in (True, False)}


def seed_matched(fixed, seeds):
    """A fixed comparator has no training-seed variance: replicate it under each seed key."""
    return {seed: fixed for seed in seeds}


def outcome(row, field):
    if field == 'success':
        return float(row['status'] == 'success')
    return float(bool(row['ship_collision'] or row['land_collision']))


def per_scenario(rows, variant, field):
    return {r['scenario_hash']: outcome(r, field) for r in rows if r['variant'] == variant}


def hierarchical(rows, data):
    seeds, out = data['training']['seeds'], {}
    for a, b, label in data['comparisons']:
        for field in ('success', 'any_collision'):
            def table(name):
                if name in COMPARATORS:
                    return seed_matched(per_scenario(rows, name, field), seeds)
                return {s: per_scenario(rows, name.format(s=s), field) for s in seeds}
            out[f'{label}/{field}'] = hierarchical_interval(table(a), table(b), seed=9127)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['heldout', 'audit', 'summarize'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='../CrowdNav-20250813-DIP/crowd_nav/data/output_trained')
    p.add_argument('--protocol', type=Path, default=ROOT/'evidence/retraining/protocol.json')
    p.add_argument('--protocol-commit', required=True)
    p.add_argument('--protocol-sha256', required=True)
    a = p.parse_args()
    raw = a.protocol.read_bytes()
    if sha256(raw).hexdigest() != a.protocol_sha256:
        raise ValueError('Protocol hash mismatch')
    path = a.protocol.resolve().relative_to(ROOT.parent).as_posix()
    if raw != subprocess.check_output(['git', 'show', f'{a.protocol_commit}:{path}'], cwd=ROOT):
        raise ValueError('Protocol is not the committed preregistration')
    data = json.loads(raw)
    variants = data['variants']
    verify_frozen_identities(data, a.model, variants=variants)
    entries = data['scenarios']['test']
    protocol = {'sha256': a.protocol_sha256, 'commit': a.protocol_commit}
    pairs = [tuple(x) for x in data['comparisons'] if '{s}' not in x[0] and '{s}' not in x[1]]
    if a.mode == 'heldout':
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        execute_matrix(entries, ROOT/'scenarios/baseline', a.model, a.output, protocol, data,
                       variants=variants, pairs=pairs)
    bind_output(a.output, protocol, entries, create=False, variants=variants)
    rows = [json.loads(line) for line in (a.output/'rows.jsonl').read_text().splitlines()]
    validate_rows(rows, entries, a.output, protocol, data, complete=True, variants=variants)
    if a.mode == 'audit':
        print(f'Verified {len(rows)} unique complete rows and all raw/stored archive hashes')
    else:
        dump(a.output/'summary.json', {**summary(rows, variants=variants, pairs=pairs),
                                       'hierarchical': hierarchical(rows, data), 'protocol': protocol})


if __name__ == '__main__':
    main()
