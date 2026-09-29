"""Serial bounded-disk baseline execution, auditing and paired summaries.

Never use the heldout command before committing benchmark_protocol.json. The
protocol SHA and its exact Git commit are mandatory and verified before execution.
"""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import mean
import argparse
import json
import random
import shutil
import subprocess
import time

from baseline_evidence import archive, compare, load_verified, reference_metrics, summarize_timing, timing_samples
from shipnav.benchmark import VARIANTS, benchmark
from shipnav.scale import units

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [('sarl_reference', 'sarl_theta', 'planner'), ('sarl_reference', 'sarl_no_goals', 'goals'),
         ('sarl_reference', 'sarl_filtered', 'SARL_filter'), ('orca_reference', 'orca_filtered', 'ORCA_filter'),
         ('marine_mpc', 'marine_mpc_unfiltered', 'MPC_filter'),
         ('sarl_filtered', 'sarl_cv_filter', 'prediction'), ('sarl_filtered', 'sarl_degraded', 'sensing'),
         ('marine_sarl', 'marine_mpc', 'matched_marine_controller')]


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run_unit(scenario, entry, name, model, output):
    output = Path(output)
    if shutil.disk_usage(output).free < 400*1024**2:
        raise RuntimeError('Below 400 MiB headroom; pause without discarding evidence')
    began = time.monotonic()
    [row] = benchmark([scenario], model, output, {name: VARIANTS[name]})
    run = json.loads((output/row['trace_file']).read_text())
    row.update(reference_metrics(run, entry['reference_route']))
    length_scale, time_scale = units(scenario['map'])
    row.update(distance_model=run.get('distance'), elapsed_model=run.get('elapsed'),
               reference_cross_track_model=(row['reference_cross_track_mean_m']/length_scale
                                            if row['reference_cross_track_mean_m'] is not None else None))
    row.update(scenario_id=entry['id'], family=scenario['family'], seed=scenario['seed'],
               traffic_count=len(scenario['traffic']), dynamics=VARIANTS[name].get('dynamics', 'holonomic'),
               deadline_ms=250*units(scenario['map'])[1], timing=timing_samples(run),
               wall_s=time.monotonic()-began)
    row['archive'] = archive(output/row['trace_file'])
    row['trace_file'] = row['archive']['file']
    row['scenario_archive'] = archive(output/f'{row["scenario_hash"]}-scenario.json')
    metric = output/f'{row["scenario_hash"]}-{name}-metrics.json'
    (output/'metrics.json').rename(metric)
    row['original_metrics_archive'] = archive(metric)
    return row


def summary(rows):
    result = {'rows': len(rows), 'variants': {}, 'comparisons': {}, 'timing': {}}
    for name in VARIANTS:
        subset = [r for r in rows if r['variant'] == name]
        if not subset:
            continue
        n = len(subset)
        statuses = Counter(r['status'] for r in subset)
        collision = sum(bool(r['ship_collision'] or r['land_collision']) for r in subset)
        result['variants'][name] = {'n': n, 'statuses': dict(statuses),
            'success_rate': statuses['success']/n, 'collision_rate': collision/n,
            'ship_collisions': sum(bool(r['ship_collision']) for r in subset),
            'land_collisions': sum(bool(r['land_collision']) for r in subset),
            'unevaluated_collision_count': sum(r['ship_collision'] is None for r in subset),
            'success_target_met': statuses['success']/n >= .8, 'collision_target_met': collision/n <= .05,
            'by_family': {family: dict(Counter(r['status'] for r in subset if r['family'] == family))
                          for family in sorted({r['family'] for r in subset})}}
        metric_names = ['elapsed_s', 'distance_m', 'min_ship_clearance', 'domain_time_s',
                        'reference_detour_ratio', 'reference_cross_track_mean_m', 'sampled_land_clearance_m',
                        'overrides', 'no_feasible_actions', 'solver_failures', 'dynamics_violations',
                        'elapsed_model', 'distance_model', 'reference_cross_track_model']
        result['variants'][name]['metrics'] = {}
        for metric in metric_names:
            values = [r[metric] for r in subset if r.get(metric) is not None
                      and (metric not in ('elapsed_s', 'elapsed_model') or r['status'] == 'success')]
            result['variants'][name]['metrics'][metric] = {'n': len(values), 'mean': mean(values) if values else None}
        result['variants'][name]['southern_physical_metrics'] = {}
        for metric in ('elapsed_s', 'distance_m', 'min_ship_clearance', 'reference_cross_track_mean_m', 'domain_time_s'):
            values = [r[metric] for r in subset if r['family'] == 'southern_corridor' and r.get(metric) is not None
                      and (metric != 'elapsed_s' or r['status'] == 'success')]
            result['variants'][name]['southern_physical_metrics'][metric] = {'n': len(values), 'mean': mean(values) if values else None}
    for a, b, label in PAIRS:
        result['comparisons'][label] = {'a': a, 'b': b, **compare(rows, a, b)}
    groups = defaultdict(lambda: [[], []])
    for row in rows:
        for temperature, stages in row['timing'].items():
            for stage, values in stages.items():
                for traffic in (str(row['traffic_count']), 'all'):
                    key = '/'.join((row['variant'], row['dynamics'], 'traffic='+traffic, temperature, stage))
                    groups[key][0].extend(values)
                    groups[key][1].extend([row['deadline_ms']]*len(values))
    result['timing'] = {key: summarize_timing(*values) for key, values in sorted(groups.items())}
    return result


def execute_matrix(entries, manifest_dir, model, output, protocol=None):
    output.mkdir(parents=True, exist_ok=True)
    journal = output/'rows.jsonl'
    rows = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    finished = {(r['scenario_hash'], r['variant']) for r in rows}
    if len(finished) != len(rows):
        raise ValueError('Duplicate journal rows')
    for row in rows:
        for key in ('archive', 'scenario_archive', 'original_metrics_archive'):
            load_verified(output, row[key])
    order = list(entries)
    random.Random(9127).shuffle(order)
    for i, entry in enumerate(order):
        scenario = load_verified(manifest_dir, entry)
        names = list(VARIANTS)
        names = names[i % len(names):]+names[:i % len(names)]
        for name in names:
            if (entry['scenario_hash'], name) in finished:
                continue
            row = run_unit(scenario, entry, name, model, output)
            with journal.open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
            rows.append(row)
            print(f'{len(rows)}/{len(entries)*len(VARIANTS)} {entry["id"]} {name} {row["status"]} free_MiB={shutil.disk_usage(output).free//1024**2}', flush=True)
    dump(output/'metrics.json', rows)
    result = summary(rows)
    result['protocol'] = protocol
    dump(output/'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['pilot', 'heldout', 'audit', 'summarize'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='../CrowdNav-20250813-DIP/crowd_nav/data/output_trained')
    parser.add_argument('--protocol', type=Path, default=Path('evidence/baseline/benchmark_protocol.json'))
    parser.add_argument('--protocol-commit')
    parser.add_argument('--protocol-sha256')
    args = parser.parse_args()
    if args.mode in ('audit', 'summarize'):
        rows = [json.loads(line) for line in (args.output/'rows.jsonl').read_text().splitlines()]
        if args.mode == 'audit':
            for row in rows:
                for key in ('archive', 'scenario_archive', 'original_metrics_archive'):
                    load_verified(args.output, row[key])
            print(f'Verified {len(rows)} rows and all raw/stored archive hashes')
        else:
            dump(args.output/'summary.json', summary(rows))
        return
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest_dir = ROOT/'scenarios/baseline'
    manifest = json.loads((manifest_dir/'splits.json').read_text())
    if args.mode == 'pilot':
        entries = [e for e in manifest['splits']['dev'] if e['id'] in ('dev_2000', 'dev_ubin_corridor_6000')]
        protocol = None
    else:
        if not args.protocol_commit or not args.protocol_sha256:
            parser.error('Heldout requires frozen protocol commit and SHA256')
        raw = args.protocol.read_bytes()
        if sha256(raw).hexdigest() != args.protocol_sha256:
            raise ValueError('Protocol hash mismatch')
        committed = subprocess.check_output(['git', 'show', f'{args.protocol_commit}:rebuild/{args.protocol.as_posix()}'], cwd=ROOT)
        if raw != committed:
            raise ValueError('Protocol is not the committed preregistration')
        data = json.loads(raw)
        if sha256((manifest_dir/'splits.json').read_bytes()).hexdigest() != data['manifest_sha256']:
            raise ValueError('Manifest changed after preregistration')
        if data['variants'] != VARIANTS:
            raise ValueError('Variants changed after preregistration')
        entries = manifest['splits']['test']
        protocol = {'sha256': args.protocol_sha256, 'commit': args.protocol_commit}
    execute_matrix(entries, manifest_dir, args.model, args.output, protocol)


if __name__ == '__main__':
    main()
