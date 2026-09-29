"""Execute every arm of frozen scenarios, retaining failures and hashed raw traces.

The variant table isolates each default factor; other comparisons must explicitly
match dynamics, filtering and observations. Fixed reference routes for planner
comparison belong to the frozen evaluation protocol (own-route metrics here score
controller tracking). This function does not select, omit or bootstrap failed runs.
"""
from hashlib import sha256
from pathlib import Path
import json

from shipnav.service import execute
from shipnav.scenarios import scenario_hash
from shipnav.planning import InvalidRoute, NoPath, PlanningLimit
from shipnav.metrics import metrics, geometry_metrics
from shipnav.scale import map_options

VARIANTS = {
    'sarl_reference': dict(policy_name='sarl'),
    'sarl_theta': dict(policy_name='sarl', planner='theta'),
    'sarl_no_goals': dict(policy_name='sarl', global_goals=False),
    'sarl_filtered': dict(policy_name='sarl', filtered=True),
    'sarl_cv_filter': dict(policy_name='sarl', filtered=True, uncertainty=False),
    'sarl_degraded': dict(policy_name='sarl', filtered=True, observation={'noise': .1, 'delay': .5, 'dropout': .1}),
    'orca_reference': dict(policy_name='orca'),
    'orca_filtered': dict(policy_name='orca', filtered=True),
    'direct_reference': dict(policy_name='direct'),
    'marine_sarl': dict(policy_name='sarl', dynamics='marine', filtered=True),
    'marine_mpc': dict(policy_name='mpc', dynamics='marine', filtered=True),
    'marine_mpc_unfiltered': dict(policy_name='mpc', dynamics='marine', filtered=False),
}


def benchmark(scenarios, model_dir, output, variants=VARIANTS, runner=execute):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in scenarios:
        identity = scenario_hash(scenario)
        (output/f'{identity}-scenario.json').write_text(json.dumps(scenario, allow_nan=False))
        for name, config in variants.items():
            try:
                run = runner(scenario['map'], scenario['start'], scenario['goal'], model_dir,
                             scenario=scenario, **{**map_options(scenario['map']), **config})
            except (NoPath, PlanningLimit, InvalidRoute) as error:
                run = {'status': 'planning_failure', 'error': f'{type(error).__name__}: {error}'}
            except Exception as error:
                run = {'status': 'error', 'error': f'{type(error).__name__}: {error}'}
            run = {**run, 'scenario_hash': identity, 'variant': name, 'variant_config': config}
            trace_file = f'{identity}-{name}.json'
            payload = json.dumps(run, allow_nan=False).encode('utf-8')
            (output/trace_file).write_bytes(payload)
            row = {'scenario_hash': identity, 'variant': name, 'trace_file': trace_file,
                   'trace_hash': sha256(payload).hexdigest(), **metrics(run), **geometry_metrics(run)}
            if 'error' in run:
                row['error'] = run['error']
            rows.append(row)
    (output/'metrics.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    return rows
