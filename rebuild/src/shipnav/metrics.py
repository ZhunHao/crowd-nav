"""Score recorded truth without changing the trace's model coordinates.

Distances and durations in metric output use the recorded map's ``units()``.
``min_ship_clearance`` is metres, despite its historical name. Domain exposure is
an experimental <1 physical metre ship-clearance threshold, weighted by actual
frame intervals (including partial final ticks). Collision flags use the recorded
swept checks; land clearance is only sampled at frame positions.

Cross-track error is controller tracking error against the run's own route. Planner
comparisons must instead score against a single reference route frozen in their
protocol. Marine rate bounds (.35 rad/model second, .2 speed/model second) stay in
the model frame and start at rest on the first navigation leg's bearing. Decision
latencies are wall-clock milliseconds; the deadline budget is the configured
control period in physical seconds, not a shortened terminal interval.

Failed runs without a trace have unknown evaluated metrics (None). Bootstrap
functions require complete finite paired values: callers must encode failures in
an outcome metric or state a joint eligibility rule, never silently drop one arm.
"""
from math import atan2, dist, isfinite, pi
from random import Random
from statistics import mean

from shipnav.scale import units


def quantile(values, q):
    """Linearly interpolate a sample quantile; an empty sample has no quantile."""
    if not 0 <= q <= 1:
        raise ValueError('Quantile must lie between zero and one')
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered)-1)*q
    lower = int(position)
    upper = min(lower+1, len(ordered)-1)
    return ordered[lower]+(position-lower)*(ordered[upper]-ordered[lower])


def _units(run):
    return units(run.get('map', {'metadata': {}}))


def _exposure(run, length_scale, time_scale):
    frames = run.get('frames', [])
    if not frames:
        return None
    return sum((b['t']-a['t'])*time_scale
               for a, b, d in zip(frames, frames[1:], run.get('diagnostics', []))
               if d.get('clearance') is not None and d['clearance']*length_scale < 1.)


def metrics(run):
    """Episode safety, intervention, distance/time and decision timing metrics."""
    diagnostics = run.get('diagnostics', [])
    recorded = bool(run.get('frames') or diagnostics)
    settings = run.get('settings', {})
    length_scale, time_scale = _units(run)
    latencies = [d['decision_ms'] for d in diagnostics if 'decision_ms' in d]
    route = run.get('route', [])
    route_length = sum(dist(a, b) for a, b in zip(route, route[1:]))
    clearances = [d['clearance']*length_scale for d in diagnostics if d.get('clearance') is not None]
    feasibility = [d.get('no_feasible_action') for d in diagnostics]
    no_feasible = (sum(feasibility) if recorded and settings.get('filtered', False)
                   and all(v is not None for v in feasibility) else None)
    return {
        'status': run['status'],
        'ship_collision': any(d.get('ship_collision', False) for d in diagnostics) if recorded else None,
        'land_collision': any(d.get('land_collision', False) for d in diagnostics) if recorded else None,
        'elapsed_s': run['elapsed']*time_scale if 'elapsed' in run else None,
        'distance_m': run['distance']*length_scale if 'distance' in run else None,
        'min_ship_clearance': min(clearances) if clearances else None,
        'domain_time_s': _exposure(run, length_scale, time_scale),
        'detour_ratio': run['distance']/route_length if route_length and 'distance' in run else None,
        'overrides': sum(d.get('override', False) for d in diagnostics) if recorded else None,
        'override_rate': mean(d.get('override', False) for d in diagnostics) if diagnostics else None,
        'no_feasible_actions': no_feasible,
        'solver_failures': sum(d.get('solver_failed', False) for d in diagnostics) if recorded else None,
        'decision_mean_ms': mean(latencies) if latencies else None,
        'decision_p95_ms': quantile(latencies, .95),
        'decision_p99_ms': quantile(latencies, .99),
        'deadline_misses': sum(v > 1000*settings.get('dt', .25)*time_scale for v in latencies)
                           if recorded else None,
    }


def _validate_pairs(a, b):
    if not a or set(a) != set(b):
        raise ValueError('Require complete matching scenario IDs')
    for key in a:
        for value in (a[key], b[key]):
            if not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError('Require finite values for every paired scenario; failed runs cannot be dropped')


def _bootstrap_options(samples):
    if not isinstance(samples, int) or samples <= 0:
        raise ValueError('Bootstrap samples must be a positive integer')


def paired_interval(a, b, seed=0, samples=2000):
    """Scenario-paired percentile bootstrap for mean A minus B."""
    _validate_pairs(a, b)
    _bootstrap_options(samples)
    differences = [a[key]-b[key] for key in sorted(a)]
    rng = Random(seed)
    draws = [mean(rng.choices(differences, k=len(differences))) for _ in range(samples)]
    return {'difference': mean(differences), 'low': quantile(draws, .025),
            'high': quantile(draws, .975), 'pairs': len(differences)}


def point_segment_distance(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    squared = dx*dx+dy*dy
    fraction = max(0., min(1., ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/squared)) if squared else 0.
    return dist(p, (a[0]+fraction*dx, a[1]+fraction*dy))


def land_clearance(sea, p, radius):
    """Signed disk clearance in the supplied map frame; negative means on land."""
    from shapely.geometry import Point
    x, y = p
    x0, y0, x1, y1 = sea.bounds
    values = [x-x0, y-y0, x1-x, y1-y]
    point = Point(p)
    for polygon in sea.land:
        values.append(-polygon.boundary.distance(point) if polygon.contains(point) else polygon.distance(point))
    return min(values)-radius


def geometry_metrics(run):
    """Own-route tracking and sampled land geometry in metres; model dynamics checks."""
    from shipnav.maps import SeaMap
    frames, route = run.get('frames', []), run.get('route', [])
    if not frames:
        return {}
    sea = SeaMap.from_dict(run['map'])
    length_scale, time_scale = _units(run)
    errors = [min((point_segment_distance(f['position'], a, b) for a, b in zip(route, route[1:])),
                  default=dist(f['position'], route[0]) if route else 0.)*length_scale for f in frames]
    land = [land_clearance(sea, f['position'], run['settings']['radius'])*length_scale for f in frames]
    # global_goals=False initializes the simulated vessel on its direct navigation
    # leg, even when the stored planner route bends around land.
    navigation = run.get('goals', route)
    heading = atan2(navigation[1][1]-navigation[0][1], navigation[1][0]-navigation[0][0]) \
              if len(navigation) > 1 else 0.
    speed, violations = 0., 0
    for a, b, d in zip(frames, frames[1:], run.get('diagnostics', [])):
        dt = b['t']-a['t']
        if 'heading' in d:
            turn = (d['heading']-heading+pi) % (2*pi)-pi
            violations += int(abs(turn) > .35*dt+1e-6 or abs(d['speed']-speed) > .2*dt+1e-6)
            heading, speed = d['heading'], d['speed']
    return {'cross_track_mean_m': mean(errors), 'cross_track_max_m': max(errors),
            'sampled_land_clearance_m': min(land),
            'domain_time_s': _exposure(run, length_scale, time_scale), 'dynamics_violations': violations}


def hierarchical_interval(a, b, seed=0, samples=2000):
    """Pair both training seed and scenario, then resample both levels equally."""
    if not a or set(a) != set(b):
        raise ValueError('Training seeds differ')
    for key in a:
        _validate_pairs(a[key], b[key])
    _bootstrap_options(samples)
    rng, keys, draws = Random(seed), sorted(a), []
    for _ in range(samples):
        means = []
        for key in rng.choices(keys, k=len(keys)):
            ids = sorted(a[key])
            chosen = rng.choices(ids, k=len(ids))
            means.append(mean(a[key][i]-b[key][i] for i in chosen))
        draws.append(mean(means))
    return {'difference': mean(mean(a[key][i]-b[key][i] for i in a[key]) for key in keys),
            'low': quantile(draws, .025), 'high': quantile(draws, .975), 'training_seeds': len(keys)}
