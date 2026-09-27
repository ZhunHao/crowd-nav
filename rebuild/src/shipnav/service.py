from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json
from shipnav.maps import SeaMap
from shipnav.planning import astar, smooth
from shipnav.simulation import traffic_for_route, run_episode
from shipnav.policies import Direct, Learned, Reciprocal


def execute(map_data: dict, start: tuple, goal: tuple, model_dir: str = '',
            policy_name: str = 'sarl', global_goals: bool = True, seed: int = 0,
            count: int = 5, cancel=lambda: False) -> dict:
    sea = SeaMap.from_dict(map_data)
    route = smooth(sea, astar(sea, tuple(start), tuple(goal)))
    from shipnav.scenarios import make_scenario, load_traffic
    scenario = make_scenario(sea, start, goal, count, seed)
    traffic = load_traffic(scenario)
    goals = route if global_goals or len(route) == 1 else [tuple(start), tuple(goal)]
    hashes = {}
    if policy_name == 'sarl':
        policy = Learned(Path(model_dir))
        for name in ('rl_model.pth', 'policy.config'):
            hashes[name] = sha256((Path(model_dir)/name).read_bytes()).hexdigest()
    elif policy_name == 'orca':
        policy = Reciprocal()
    elif policy_name == 'direct':
        policy = Direct()
    else:
        raise ValueError('Policy must be sarl, orca or direct')
    result = run_episode(sea, goals, traffic, policy, cancel=cancel)
    result.update({'schema': 1, 'map': map_data, 'route': route, 'goals': goals,
                   'traffic_definitions': [asdict(s) for s in traffic],
                   'settings': {'seed': seed, 'requested_traffic': count,
                                'actual_traffic': len(traffic), 'policy': policy_name,
                                'global_goals': global_goals, 'dt': .25, 'limit': 100,
                                'radius': .5, 'speed': 1.0, 'query_env': False,
                                'traffic_model': 'constant_velocity_then_stop'},
                   'model_hashes': hashes})
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=Path, default=Path('maps/harbour.json'))
    parser.add_argument('--model', default='../CrowdNav-20250813-DIP/crowd_nav/data/output_trained')
    parser.add_argument('--policy', choices=['sarl', 'orca', 'direct'], default='sarl')
    parser.add_argument('--start', type=float, nargs=2, default=(2, 2))
    parser.add_argument('--goal', type=float, nargs=2, default=(22, 22))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--no-global-goals', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('results/run.json'))
    args = parser.parse_args()
    try:
        result = execute(SeaMap.load(args.map).to_dict(), tuple(args.start), tuple(args.goal),
                         args.model, args.policy, not args.no_global_goals, args.seed, args.count)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.exit(2, str(error)+'\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(result['status'], result['elapsed'])
