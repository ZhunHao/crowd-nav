"""Serial bounded-disk baseline execution, auditing and paired summaries.

Never use the heldout command before committing benchmark_protocol.json. The
protocol SHA and its exact Git commit are mandatory and verified before execution.
"""
from collections import Counter, defaultdict
from hashlib import sha256
from math import dist
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
from shipnav.scenarios import scenario_hash
from shipnav.scale import units
from shipnav.service import execute

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [('sarl_reference', 'sarl_theta', 'planner'), ('sarl_reference', 'sarl_no_goals', 'goals'),
         ('sarl_reference', 'sarl_filtered', 'SARL_filter'), ('orca_reference', 'orca_filtered', 'ORCA_filter'),
         ('marine_mpc', 'marine_mpc_unfiltered', 'MPC_filter'),
         ('sarl_filtered', 'sarl_cv_filter', 'prediction'), ('sarl_filtered', 'sarl_degraded', 'sensing'),
         ('marine_sarl', 'marine_mpc', 'matched_marine_controller')]


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def verify_limit(run, expected):
    if run.get('settings') and run['settings']['limit'] != expected:
        raise ValueError('Recorded episode limit differs from frozen shared budget')


def run_unit(scenario, entry, name, model, output):
    output = Path(output)
    if shutil.disk_usage(output).free < 400*1024**2:
        raise RuntimeError('Below 400 MiB headroom; pause without discarding evidence')
    began = time.monotonic()
    route = entry['reference_route'] or []
    reference_length = sum(dist(a, b) for a, b in zip(route, route[1:]))
    limit = max(100., 3*reference_length) if 'model_scale' in scenario['map']['metadata'] else 100.

    def matched_runner(*args, **kwargs):
        return execute(*args, **kwargs, limit=limit)

    [row] = benchmark([scenario], model, output, {name: VARIANTS[name]}, runner=matched_runner)
    run = json.loads((output/row['trace_file']).read_text())
    verify_limit(run, limit)
    row['shared_limit_model'] = limit
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


def verify_frozen_identities(data, model):
    """Reject a changed implementation, lock, manifest or checkpoint before reuse."""
    expected = {**data['source_sha256'], data['manifest']: data['manifest_sha256'],
                data['original_manifest']: data['original_manifest_sha256'],
                'uv.lock': data['identities']['lock_sha256']}
    for name, digest in expected.items():
        path = ROOT/name
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen source/lock/manifest identity mismatch: '+name)
    for name, digest in data['identities']['model'].items():
        path = Path(model)/name
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen checkpoint identity mismatch: '+name)
    if data['variants'] != VARIANTS:
        raise ValueError('Frozen variant identity mismatch')


def bind_output(output, protocol, entries, create=True):
    """A directory belongs to one protocol and one scenario/variant grid."""
    binding = {'protocol': protocol, 'scenarios': entries, 'variants': VARIANTS}
    path = output/'run-identity.json'
    if path.exists():
        if json.loads(path.read_text()) != binding:
            raise ValueError('Output protocol/grid identity mismatch')
    elif not create or any(output.iterdir()):
        raise ValueError('Existing output lacks a protocol binding; use a fresh directory')
    else:
        dump(path, binding)


def validate_rows(rows, entries, output, protocol, data=None, complete=False):
    """Validate partial resumes; final audit additionally requires the entire grid."""
    by_hash = {e['scenario_hash']: e for e in entries}
    if len(by_hash) != len(entries):
        raise ValueError('Duplicate frozen scenario identities')
    expected = {(h, name) for h in by_hash for name in VARIANTS}
    seen = set()
    for row in rows:
        pair = (row['scenario_hash'], row['variant'])
        if pair in seen:
            raise ValueError('Duplicate journal rows')
        if pair not in expected or row.get('protocol') != protocol:
            raise ValueError('Journal scenario/variant/protocol identity mismatch')
        seen.add(pair)
        entry = by_hash[pair[0]]
        for field, key in [('scenario_id', 'id'), ('family', 'family'), ('seed', 'seed'),
                           ('traffic_count', 'traffic_count')]:
            if row.get(field) != entry[key]:
                raise ValueError('Journal scenario metadata mismatch: '+field)
        run = load_verified(output, row['archive'])
        if (run['scenario_hash'], run['variant']) != pair or run['variant_config'] != VARIANTS[pair[1]]:
            raise ValueError('Trace scenario/variant identity mismatch')
        if (row['trace_hash'] != row['archive']['raw_sha256'] or
                row['trace_file'] != row['archive']['file'] or row['status'] != run['status']):
            raise ValueError('Journal trace identity mismatch')
        if data:
            limit = data['shared_limits_model'][pair[0]]
            verify_limit(run, limit)
            if row['shared_limit_model'] != limit:
                raise ValueError('Journal shared limit mismatch')
            if run.get('settings'):
                if (run.get('model_hashes') != data['identities']['model'] or
                        run.get('provenance', {}).get('uv_lock_sha256') != data['identities']['lock_sha256']):
                    raise ValueError('Trace model/lock identity mismatch')
        scenario = load_verified(output, row['scenario_archive'])
        if scenario_hash(scenario) != pair[0]:
            raise ValueError('Archived scenario identity mismatch')
        metrics = load_verified(output, row['original_metrics_archive'])
        if len(metrics) != 1 or any(metrics[0].get(k) != row[k]
                                   for k in ('scenario_hash', 'variant', 'trace_hash', 'status')):
            raise ValueError('Archived metrics identity mismatch')
    if complete and seen != expected:
        raise ValueError('Final audit requires the complete scenario/variant grid')


def execute_matrix(entries, manifest_dir, model, output, protocol=None, data=None):
    if protocol is not None:
        if data is None or entries != data['scenarios']['test']:
            raise ValueError('Frozen protocol scenario identity mismatch')
        verify_frozen_identities(data, model)
    output.mkdir(parents=True, exist_ok=True)
    bind_output(output, protocol, entries)
    journal = output/'rows.jsonl'
    rows = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    validate_rows(rows, entries, output, protocol, data)
    finished = {(r['scenario_hash'], r['variant']) for r in rows}
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
            row['protocol'] = protocol
            validate_rows([row], entries, output, protocol, data)
            with journal.open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
            rows.append(row)
            print(f'{len(rows)}/{len(entries)*len(VARIANTS)} {entry["id"]} {name} {row["status"]} free_MiB={shutil.disk_usage(output).free//1024**2}', flush=True)
    validate_rows(rows, entries, output, protocol, data, complete=True)
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
    manifest_dir = ROOT/'scenarios/baseline'
    manifest = json.loads((manifest_dir/'splits.json').read_text())
    data = None
    if args.mode == 'pilot':
        entries = [e for e in manifest['splits']['dev'] if e['id'] in ('dev_2000', 'dev_ubin_corridor_6000')]
        protocol = None
    else:
        if not args.protocol_commit or not args.protocol_sha256:
            parser.error('Heldout/audit/summarize require frozen protocol commit and SHA256')
        raw = args.protocol.read_bytes()
        if sha256(raw).hexdigest() != args.protocol_sha256:
            raise ValueError('Protocol hash mismatch')
        protocol_path = args.protocol.resolve().relative_to(ROOT.parent)
        committed = subprocess.check_output(['git', 'show', f'{args.protocol_commit}:{protocol_path.as_posix()}'], cwd=ROOT)
        if raw != committed:
            raise ValueError('Protocol is not the committed preregistration')
        data = json.loads(raw)
        verify_frozen_identities(data, args.model)
        entries = manifest['splits']['test']
        if entries != data['scenarios']['test']:
            raise ValueError('Frozen protocol scenario identity mismatch')
        protocol = {'sha256': args.protocol_sha256, 'commit': args.protocol_commit}
    if args.mode in ('audit', 'summarize'):
        bind_output(args.output, protocol, entries, create=False)
        rows = [json.loads(line) for line in (args.output/'rows.jsonl').read_text().splitlines()]
        validate_rows(rows, entries, args.output, protocol, data, complete=True)
        if args.mode == 'audit':
            print(f'Verified {len(rows)} unique complete rows and all raw/stored archive hashes')
        else:
            dump(args.output/'summary.json', {**summary(rows), 'protocol': protocol})
        return
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    execute_matrix(entries, manifest_dir, args.model, args.output, protocol, data)


if __name__ == '__main__':
    main()
