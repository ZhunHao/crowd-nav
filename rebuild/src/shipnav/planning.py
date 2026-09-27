"""Deterministic grid A* and Basic Theta*, with vector-validated edges."""
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import ceil, dist, floor, isfinite
from time import perf_counter

from shipnav.maps import valid_point


class NoPath(ValueError):
    """No route in the selected graph, or a blocked endpoint."""


class PlanningLimit(RuntimeError):
    """Compute budget exceeded; feasibility is unknown."""


class InvalidRoute(RuntimeError):
    """A planner output failed final geometric validation."""


@dataclass
class RouteResult:
    points: list
    stats: dict


def smooth(sea, route, clearance=.7, *, check_budget=lambda: None):
    if not route:
        raise ValueError('Route is empty')
    result, i = [tuple(route[0])], 0
    if not sea.clear(route[0], route[0], clearance):
        raise NoPath('Blocked route start')
    while i < len(route)-1:
        j = len(route)-1
        while j > i:
            check_budget()
            if sea.clear(route[i], route[j], clearance):
                break
            j -= 1
        if j == i:
            raise NoPath('Unsafe route segment')
        result.append(tuple(route[j]))
        i = j
    return result


def plan(sea, start, goal, *, planner='astar', resolution=1., clearance=.7,
         max_nodes=250_000, max_cells=250_000, timeout=30.):
    """Plan a disk-centre route, not a dynamically feasible vessel trajectory.

    Theta* performs parent line-of-sight relaxation during search. A* output is
    smoothed; both use identical grid connectors and continuous collision checks.
    """
    if planner not in ('astar', 'theta'):
        raise ValueError('planner must be astar or theta')
    if any(not isfinite(v) or v <= 0 for v in (resolution, timeout)):
        raise ValueError('Resolution and timeout must be positive finite values')
    if not isinstance(max_nodes, int) or max_nodes <= 0 or not isinstance(max_cells, int) or max_cells <= 0:
        raise ValueError('Budgets must be positive integers')
    start, goal = tuple(start), tuple(goal)
    if not valid_point(start) or not valid_point(goal):
        raise ValueError('Endpoints must be finite 2D coordinates')
    begin = perf_counter()
    expanded = 0

    def failure(kind, message):
        error = kind(message)
        error.stats = {'expanded_nodes': expanded, 'elapsed_s': perf_counter()-begin,
                       'resolution_m': resolution, 'clearance_m': clearance}
        return error

    def check_budget():
        if perf_counter()-begin > timeout:
            raise failure(PlanningLimit, 'Planning deadline exceeded')

    def finish(points, raw_count):
        check_budget()
        segments = list(zip(points, points[1:])) or [(points[0], points[0])]
        minimum = float('inf')
        for a, b in segments:
            check_budget()
            if not sea.clear(a, b, clearance):
                raise failure(InvalidRoute, 'Planner produced an invalid route')
            minimum = min(minimum, sea.minimum_clearance(a, b))
        check_budget()
        return RouteResult(points, {'planner': 'astar_smooth' if planner == 'astar' else 'theta',
            'resolution_m': resolution, 'clearance_m': clearance, 'expanded_nodes': expanded,
            'raw_waypoints': raw_count, 'length_m': sum(dist(a, b) for a, b in segments),
            'minimum_centre_clearance_m': minimum,
            'minimum_clearance_slack_m': minimum-clearance,
            'elapsed_s': perf_counter()-begin, 'geometry_validation': 'continuous-disk',
            'max_nodes': max_nodes, 'max_cells': max_cells, 'timeout_s': timeout})

    if not sea.clear(start, start, clearance) or not sea.clear(goal, goal, clearance):
        raise failure(NoPath, 'Start or goal is blocked')
    if start == goal:
        return finish([start], 1)
    if sea.clear(start, goal, clearance):
        return finish([start, goal], 2)
    x0, y0, x1, y1 = sea.bounds
    nx, ny = ceil((x1-x0)/resolution), ceil((y1-y0)/resolution)
    if nx*ny > max_cells:
        raise failure(PlanningLimit, f'Grid has {nx*ny} cells; limit is {max_cells}')
    first, last = (-1, -1), (-2, -2)

    def point(node):
        if node == first:
            return start
        if node == last:
            return goal
        return x0+(node[0]+.5)*resolution, y0+(node[1]+.5)*resolution

    def neighbours(node):
        ix, iy = (floor((start[0]-x0)/resolution), floor((start[1]-y0)/resolution)) if node == first else node
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if node != first and dx == dy == 0:
                    continue
                other = ix+dx, iy+dy
                if 0 <= other[0] < nx and 0 <= other[1] < ny and sea.clear(point(node), point(other), clearance):
                    yield other
        if dist(point(node), goal) <= 2*resolution and sea.clear(point(node), goal, clearance):
            yield last

    serial = count()
    queue = [(dist(start, goal), next(serial), first, 0.)]
    costs, parent = {first: 0.}, {}
    while queue:
        check_budget()
        _, _, node, popped_cost = heappop(queue)
        if popped_cost != costs[node]:
            continue
        if node == last:
            route = [goal]
            while node != first:
                node = parent[node]
                route.append(point(node))
            route.reverse()
            raw_count = len(route)
            if planner == 'astar':
                route = smooth(sea, route, clearance, check_budget=check_budget)
            return finish(route, raw_count)
        if expanded >= max_nodes:
            raise failure(PlanningLimit, 'Expanded-node budget exceeded')
        expanded += 1
        for other in neighbours(node):
            via = node
            if planner == 'theta':
                ancestor = parent.get(node, node)
                if sea.clear(point(ancestor), point(other), clearance):
                    via = ancestor
            cost = costs[via] + dist(point(via), point(other))
            if cost < costs.get(other, float('inf')):
                costs[other], parent[other] = cost, via
                heappush(queue, (cost+dist(point(other), goal), next(serial), other, cost))
    check_budget()
    raise failure(NoPath, 'No route at this grid resolution')


def astar(sea, start, goal, clearance=.7, resolution=1.):
    return plan(sea, start, goal, clearance=clearance, resolution=resolution).points


def theta_star(sea, start, goal, clearance=.7, resolution=1.):
    return plan(sea, start, goal, planner='theta', clearance=clearance, resolution=resolution).points
