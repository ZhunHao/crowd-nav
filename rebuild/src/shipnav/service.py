from hashlib import sha256
from math import dist
from pathlib import Path
import json
from shipnav.maps import SeaMap, canonical_json
from shipnav.planning import astar, smooth
from shipnav.simulation import CourseChangeTraffic, run_episode
from shipnav.scenarios import make_scenario, load_traffic, scenario_hash, validate_observation, validate_scenario
from shipnav.observations import Observer
from shipnav.theta import theta_star
from shipnav.policies import Direct, Learned, Reciprocal
from shipnav.provenance import provenance

_ROOT = Path(__file__).resolve().parents[3]


def execute(map_data: dict, start: tuple, goal: tuple, model_dir: str = '',
            policy_name: str = 'sarl', global_goals: bool = True, seed: int = 0,
            count: int = 5, cancel=lambda: False, *, scenario=None, planner='astar_smooth',
            filtered=False, uncertainty=True, observation=None, dynamics='holonomic',
            resolution=1., clearance=.7, limit=None, placement='uniform') -> dict:
    if scenario is not None:
        validate_scenario(scenario)
    validate_observation(observation or {})
    sea = SeaMap.from_dict(map_data)
    if placement not in ('uniform', 'corridor'):
        raise ValueError('Placement must be uniform or corridor')
    if planner == 'astar_smooth':
        route = smooth(sea,astar(sea,tuple(start),tuple(goal),clearance,resolution),clearance)
    elif planner == 'theta':
        route = theta_star(sea,tuple(start),tuple(goal),clearance,resolution)
    else:
        raise ValueError('Unknown planner')
    if scenario is None:
        # Corridor traffic is timed on the A*-smoothed reference route whichever planner
        # runs, so planner comparisons share identical traffic.
        reference = None if placement == 'uniform' else route if planner == 'astar_smooth' else \
            smooth(sea,astar(sea,tuple(start),tuple(goal),clearance,resolution),clearance)
        scenario = make_scenario(sea,start,goal,count,seed,corridor=reference)
    if (canonical_json(SeaMap.from_dict(scenario['map']).to_dict()) != canonical_json(sea.to_dict())
            or tuple(scenario['start'])!=tuple(start) or tuple(scenario['goal'])!=tuple(goal)):
        raise ValueError('Scenario map/endpoints differ from run request')
    traffic = load_traffic(scenario)
    if scenario.get('traffic_mode')=='reactive':
        from shipnav.reactive import ReactiveTraffic
        if any(isinstance(s, CourseChangeTraffic) for s in traffic):
            raise ValueError('Reactive traffic mode does not support course_change entries')
        traffic = [ReactiveTraffic(s,sea) for s in traffic]
    elif scenario.get('traffic_mode')!='scripted':
        raise ValueError('Unknown traffic model')
    goals = route if global_goals or len(route) == 1 else [tuple(start), tuple(goal)]
    hashes = {}
    if policy_name == 'sarl':
        policy = Learned(Path(model_dir))
        for name in ('rl_model.pth', 'policy.config'):
            hashes[name] = sha256((Path(model_dir)/name).read_bytes()).hexdigest()
    elif policy_name == 'orca':
        policy = Reciprocal()
    elif policy_name == 'mpc':
        from shipnav.controllers.mpc import MPC
        policy = MPC()
    elif policy_name == 'direct':
        policy = Direct()
    else:
        raise ValueError('Policy must be sarl, orca, mpc or direct')
    observer = Observer(seed=scenario['seed'],**(observation or {}))
    # Scaled real maps get three times the nominal route time; synthetic maps retain
    # their legacy 100 s default, including previously timed-out long routes.
    length = sum(dist(a, b) for a, b in zip(route, route[1:]))
    if limit is None:
        limit = max(100., 3*length) if 'model_scale' in map_data['metadata'] else 100.
    result = run_episode(sea,goals,traffic,policy,limit=limit,cancel=cancel,observer=observer,filtered=filtered,uncertainty=uncertainty,dynamics=dynamics)
    settings = {'seed': scenario['seed'], 'planner': planner, 'filtered': filtered, 'uncertainty': uncertainty, 'observation': observation or {}, 'dynamics': dynamics, 'requested_traffic': count,
                'actual_traffic': len(traffic), 'policy': policy_name,
                'global_goals': global_goals, 'dt': .25, 'limit': limit,
                'resolution': resolution, 'clearance': clearance, 'placement': placement,
                'radius': .5, 'speed': 1.0, 'query_env': False,
                'traffic_model': scenario['traffic_mode']}
    result.update({'schema': 2, 'scenario': scenario, 'scenario_hash': scenario_hash(scenario), 'map': map_data, 'route': route, 'goals': goals,
                   'traffic_definitions': scenario['traffic'],
                   'settings': settings,
                   'model_hashes': hashes,
                   'provenance': provenance(settings, _ROOT)})
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=Path, default=Path('maps/harbour.json'))
    parser.add_argument('--model', default='../CrowdNav-20250813-DIP/crowd_nav/data/output_trained')
    parser.add_argument('--policy', choices=['sarl', 'orca', 'mpc', 'direct'], default='sarl')
    parser.add_argument('--start', type=float, nargs=2, default=(2, 2))
    parser.add_argument('--goal', type=float, nargs=2, default=(22, 22))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--no-global-goals', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('results/run.json'))
    parser.add_argument('--scenario', type=Path)
    parser.add_argument('--planner', choices=['astar_smooth','theta'], default='astar_smooth')
    parser.add_argument('--filtered', action='store_true')
    parser.add_argument('--dynamics', choices=['holonomic','marine'], default='holonomic')
    parser.add_argument('--profile', help='Vessel profile for a metric map (default: model for maps up to '
                        '250,000 m², harbour_craft above). --start/--goal stay in metres')
    parser.add_argument('--placement', choices=['uniform','corridor'],
                        help='Traffic placement (default: uniform for model profile, corridor otherwise)')
    args = parser.parse_args()
    try:
        # json.JSONDecodeError is a ValueError; schema problems surface as
        # ValueError from validate_scenario rather than KeyError/TypeError.
        from shipnav.scale import PROFILES, check_grid, default_profile, map_options, scaled
        scenario = json.loads(args.scenario.read_text()) if args.scenario else None
        if scenario is not None:
            # A frozen scenario is already in model units and records its own planner settings.
            validate_scenario(scenario)
            if args.profile:
                raise ValueError('--profile applies to --map, not to a frozen --scenario')
            map_data, start, goal = scenario['map'], tuple(scenario['start']), tuple(scenario['goal'])
            options = map_options(map_data)
        else:
            real = SeaMap.load(args.map)
            if args.profile is not None and args.profile not in PROFILES:
                raise ValueError(f'Unknown profile; choose from {", ".join(PROFILES)}')
            profile = PROFILES[args.profile] if args.profile else default_profile(real)
            check_grid(real, profile)
            map_data = scaled(real, profile).to_dict()
            start, goal = (tuple(v/profile.length for v in p) for p in (args.start, args.goal))
            options = profile.service_options()
        placement = args.placement or ('uniform' if not map_data['metadata'].get('model_scale') else 'corridor')
        result = execute(map_data, start, goal,
                         args.model, args.policy, not args.no_global_goals, args.seed, args.count,
                         scenario=scenario,planner=args.planner,filtered=args.filtered,dynamics=args.dynamics,
                         placement=placement, **options)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.exit(2, str(error)+'\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(result['status'], result['elapsed'])
