"""Episode setup shared by the service and the training environment.

Both must build identical routes, traffic and time budgets from one frozen
scenario; otherwise a policy trained in one setting is evaluated in another.
"""
from math import dist

from shipnav.planning import astar, smooth
from shipnav.scenarios import load_traffic
from shipnav.simulation import CourseChangeTraffic
from shipnav.theta import theta_star

SYNTHETIC_LIMIT = 100.


def plan_route(sea, start, goal, planner='astar_smooth', resolution=1., clearance=.7):
    if planner == 'astar_smooth':
        return smooth(sea, astar(sea, tuple(start), tuple(goal), clearance, resolution), clearance)
    if planner == 'theta':
        return theta_star(sea, tuple(start), tuple(goal), clearance, resolution)
    raise ValueError('Unknown planner')


def build_traffic(scenario, sea):
    traffic = load_traffic(scenario)
    mode = scenario.get('traffic_mode')
    if mode == 'reactive':
        from shipnav.reactive import ReactiveTraffic
        if any(isinstance(ship, CourseChangeTraffic) for ship in traffic):
            raise ValueError('Reactive traffic mode does not support course_change entries')
        return [ReactiveTraffic(ship, sea) for ship in traffic]
    if mode != 'scripted':
        raise ValueError('Unknown traffic model')
    return traffic


def episode_limit(map_data, route):
    """Synthetic maps keep 100 model seconds; scaled real maps get 3x the route time."""
    if 'model_scale' not in map_data['metadata']:
        return SYNTHETIC_LIMIT
    return max(SYNTHETIC_LIMIT, 3*sum(dist(a, b) for a, b in zip(route, route[1:])))
