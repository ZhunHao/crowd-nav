"""Lossless archive, fixed-reference scoring and paired baseline analysis helpers."""
from copy import deepcopy
from hashlib import sha256
from math import dist, isfinite
from pathlib import Path
from statistics import mean
import gzip
import json

from shipnav.metrics import paired_interval, point_segment_distance, quantile
from shipnav.scale import units


def archive(path):
    """Replace only this new raw artifact after verifying its deterministic archive."""
    path = Path(path)
    raw = path.read_bytes()
    stored = gzip.compress(raw, compresslevel=6, mtime=0)
    target = path.with_suffix(path.suffix+'.gz')
    target.write_bytes(stored)
    entry = {'file': target.name, 'stored_sha256': sha256(stored).hexdigest(),
             'raw_sha256': sha256(raw).hexdigest(), 'raw_bytes': len(raw), 'stored_bytes': len(stored)}
    load_verified(target.parent, entry)
    path.unlink()
    return entry


def load_verified(root, entry):
    stored = (Path(root)/entry['file']).read_bytes()
    if sha256(stored).hexdigest() != entry['stored_sha256']:
        raise ValueError('Stored hash mismatch: '+entry['file'])
    raw = gzip.decompress(stored) if entry['file'].endswith('.gz') else stored
    if sha256(raw).hexdigest() != entry['raw_sha256']:
        raise ValueError('Raw hash mismatch: '+entry['file'])
    return json.loads(raw)


def reference_metrics(run, reference):
    result = dict.fromkeys(('reference_cross_track_mean_m', 'reference_cross_track_max_m', 'reference_detour_ratio'))
    if not reference or not run.get('frames'):
        return result
    length, _ = units(run['map'])
    errors = [min(point_segment_distance(f['position'], a, b)
                  for a, b in zip(reference, reference[1:]))*length for f in run['frames']]
    total = sum(dist(a, b) for a, b in zip(reference, reference[1:]))
    result.update(reference_cross_track_mean_m=mean(errors), reference_cross_track_max_m=max(errors),
                  reference_detour_ratio=run['distance']/total if total else None)
    return result


def compare(rows, a, b, samples=2000):
    arms = [{r['scenario_hash']: r for r in rows if r['variant'] == name} for name in (a, b)]
    if not arms[0] or set(arms[0]) != set(arms[1]):
        raise ValueError('Require complete matching scenario IDs')
    result = {}
    for metric, score in [('success', lambda r: int(r['status'] == 'success')),
                          ('collision', lambda r: int(bool(r['ship_collision'] or r['land_collision'])))]:
        result[metric] = paired_interval(*[{k: score(r) for k, r in arm.items()} for arm in arms],
                                          seed=9127, samples=samples)
    eligible = [k for k in arms[0] if all(arm[k]['status'] == 'success' for arm in arms)]
    result['excluded_joint_success'] = sorted(set(arms[0])-set(eligible))
    for output, metric in [('completion_time_s', 'elapsed_s'), ('fixed_reference_detour', 'reference_detour_ratio'),
                           ('fixed_reference_tracking_m', 'reference_cross_track_mean_m')]:
        keys = [k for k in eligible if all(isinstance(arm[k].get(metric), (int, float)) and
                                           isfinite(arm[k][metric]) for arm in arms)]
        result[output] = paired_interval(*[{k: arm[k][metric] for k in keys} for arm in arms],
                                         seed=9127, samples=samples) if keys else {'pairs': 0, 'difference': None, 'low': None, 'high': None}
        result[output]['excluded'] = sorted(set(arms[0])-set(keys))
    return result


def timing_samples(run):
    inference = run.get('inference_ms', [])
    decision = [d['decision_ms'] for d in run.get('diagnostics', [])]
    return {name: {'inference': inference[sl], 'decision': decision[sl]}
            for name, sl in [('cold_instance', slice(0, 1)), ('transition', slice(1, 5)), ('warmed', slice(5, None))]}


def summarize_timing(values, deadlines):
    if len(values) != len(deadlines):
        raise ValueError('Every timing needs its physical deadline')
    return {'count': len(values), 'mean_ms': mean(values) if values else None,
            'p95_ms': quantile(values, .95), 'p99_ms': quantile(values, .99),
            'deadline_misses': sum(v > d for v, d in zip(values, deadlines))}


def perturb(source, index):
    """Small fixed endpoint/speed variations, without controller feedback or seed relabeling."""
    result = deepcopy(source)
    result['seed'] = 4000+index
    result['start'][0] += (index-4.5)*.15
    for ship in result['traffic']:
        if 'speed' in ship:
            ship['speed'] *= .94+.015*index
        else:
            for waypoint in ship['waypoints'][1:]:
                waypoint[0] /= .94+.015*index
    return result
