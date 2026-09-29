from pathlib import Path
import csv
import json
from statistics import mean
from shipnav.maps import SeaMap
from shipnav.service import execute

VARIANTS = [('global_sarl', 'sarl', True), ('final_goal_sarl', 'sarl', False),
            ('global_direct', 'direct', True), ('global_orca', 'orca', True)]


def summarize(runs: list[dict]) -> dict:
    successful = [r for r in runs if r['status'] == 'success']
    all_latency = [t for r in runs for t in r.get('inference_ms', [])]
    return {'runs': len(runs),
            'successes': len(successful),
            'collisions': sum(r['status'] == 'collision' for r in runs),
            'timeouts': sum(r['status'] == 'timeout' for r in runs),
            'errors': sum(r['status'] == 'error' for r in runs),
            'mean_success_time_s': mean(r['elapsed'] for r in successful) if successful else None,
            'mean_success_distance_m': mean(r['distance'] for r in successful) if successful else None,
            'mean_step_inference_ms': mean(all_latency) if all_latency else None}


def evaluate(map_data: dict, model_dir: str, output: Path, seeds=range(10),
             counts=(5, 8, 12, 15), runner=execute) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for count in counts:
        grouped = {name: [] for name, _, _ in VARIANTS}
        for seed in seeds:
            scenario = None
            for name, policy, global_goals in VARIANTS:
                try:
                    result = runner(map_data, (2, 2), (22, 22), model_dir,
                                    policy, global_goals, seed, count)
                    if scenario is None:
                        scenario = result['traffic_definitions']
                    if result['traffic_definitions'] != scenario:
                        raise AssertionError('Paired scenarios differ')
                except (ValueError, FileNotFoundError, RuntimeError) as error:
                    result = {'status': 'error', 'error': str(error), 'seed': seed, 'count': count}
                (output/f'{name}-n{count}-seed{seed}.json').write_text(
                    json.dumps(result, indent=2, allow_nan=False))
                grouped[name].append(result)
        for name, runs in grouped.items():
            summaries.append({'variant': name, 'traffic_count': count, **summarize(runs)})
    with (output/'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=Path, default=Path('maps/harbour.json'))
    parser.add_argument('--model', default='../CrowdNav-20250813-DIP/crowd_nav/data/output_trained')
    parser.add_argument('--output', type=Path, default=Path('results/evaluation'))
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--counts', type=int, nargs='+', default=[5, 8, 12, 15])
    args = parser.parse_args()
    if args.seeds <= 0 or any(n < 0 for n in args.counts):
        parser.error('Use positive seed count and non-negative traffic counts')
    print(evaluate(SeaMap.load(args.map).to_dict(), args.model, args.output,
                   range(args.seeds), tuple(args.counts)))
