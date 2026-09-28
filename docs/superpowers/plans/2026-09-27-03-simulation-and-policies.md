# 03 — Continuous simulation and policy integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run complete start-to-destination episodes with moving traffic, global waypoints and learned or rule-based local policies.

**Architecture:** Use a headless deterministic simulator with swept collision checks. Load the existing SARL network through an adapter; keep the new simulator independent of Gym and preserve one clock across all goals.

**Tech Stack:** Python standard library, PyTorch and original SARL/ORCA through adapters, pytest

**Spec:** [../specs/2026-09-27-solo-rebuild-design.md](../specs/2026-09-27-solo-rebuild-design.md)

## Global Constraints

- Never use `codex` in branch names or worktree directory names.
- Preserve `CrowdNav-20250813-DIP/` and the supplied briefing unchanged.
- Implement only inside `rebuild/`; keep planning documents inside `docs/superpowers/` and Git-ignored.
- Use `.venv-legacy` only for baseline verification and migration comparison; remove the legacy environment, build dependencies and runtime support after modern acceptance passes.
- Retain baseline results, hashes and frozen comparison fixtures as verification evidence; do not retain a legacy fallback or maintain a second runtime.
- After that gate, attempt the newest stable Python and dependency releases available at execution time; prereleases require an explicit separate experiment.
- All subsequent phases use `.venv-modern` and its committed lockfile; Linux EC2 training uses a separately verified CUDA environment with the same application version and Python minor.
- Do not silently downgrade the modern stack. Record compatibility blockers, isolate optional native dependencies, and re-run parity checks after any port.
- Run migration parity on CPU first; validate MPS and CUDA separately before using their results.
- Use metres, seconds, radians, and Cartesian East–North `(x, y)` coordinates in the rebuilt simulation.
- Keep one episode clock and continuous vessel state across intermediate goals.
- Freeze scenarios independently of planners; pair initial conditions and exogenous randomness across comparisons.
- Keep perceived observations separate from scoring truth; report collisions, failures, safety interventions and missed deadlines honestly.
- Use AWS CLI for EC2 retraining operations; choose account, region, resource sizes and an explicit spending limit when executing that phase.
- Do not implement literature work, academic report writing, or competition presentation preparation in these plans.

---

All file paths below are relative to the workspace root. Run commands from `rebuild/` unless explicitly stated otherwise. Code blocks are proposed implementation content, not evidence of passing tests. A code step can be worked through function by function in 2–5 minute increments; commit only after the task's tests pass. Git is initialized on `main`; commit only explicitly named implementation files. Documentation remains ignored.

## File structure and boundaries

Create `rebuild/src/shipnav/simulation.py`, `policies.py`, `service.py` and corresponding `rebuild/tests/test_simulation.py`, `test_policies.py`, `test_service.py`. Task 4 can be tested with tiny callable policies before PyTorch works; Task 5 requires the completed plan 01b migration; Task 6 joins plans 02 and 03.

The checkpoint was trained in the original environment. Constant-velocity traffic and disabling `query_env` are deliberate distribution changes in the rebuild. Results must identify them. A rebuilt simulator demonstration is not evidence that the original model is robust to land or changed ship behaviour.

### Task 4: Simulate continuous episodes and moving traffic

**Files:** `rebuild/src/shipnav/simulation.py`, `rebuild/tests/test_simulation.py`.

**Interfaces:** Consumes `SeaMap.clear`. Produces `Traffic(start: Point, goal: Point, speed=.3, radius=.6)`, `.at(time) -> (position, velocity, radius)`, `traffic_for_route(sea, route, count, seed) -> list[Traffic]`, and `run_episode(sea, route, traffic, policy, dt=.25, limit=100, radius=.5, speed=1, cancel=callable) -> dict`. A policy is callable `(position, velocity, goal, neighbours, radius, speed, dt) -> Point`.

- [ ] **Step 1: Write the behaviour tests.**

Create `rebuild/tests/test_simulation.py` with:

```python
import pytest
from math import dist
from shipnav.maps import SeaMap
from shipnav.simulation import Traffic, run_episode, traffic_for_route, swept_clearance


def toward(p, v, goal, neighbours, radius, speed, dt):
    d = dist(p, goal)
    return tuple((b-a)/d*min(speed, d/dt) if d else 0 for a, b in zip(p, goal))


def test_waypoints_keep_one_clock_and_continuous_motion():
    sea = SeaMap((0, 0, 20, 20))
    route = [(2, 2), (8, 2), (8, 8)]
    result = run_episode(sea, route, [], toward)
    assert result['status'] == 'success'
    frames = result['frames']
    assert [f['t'] for f in frames] == sorted(set(f['t'] for f in frames))
    assert result['elapsed'] > 10
    assert all(dist(a['position'], b['position']) <= .250001 for a, b in zip(frames, frames[1:]))


def test_land_collision_timeout_cancellation_and_already_arrived():
    sea = SeaMap((0, 0, 10, 10), ((4, 4, 6, 6),))
    assert run_episode(sea, [(2, 5), (8, 5)], [], toward)['status'] == 'collision'
    assert run_episode(sea, [(2, 2), (8, 2)], [], toward, limit=.25)['status'] == 'timeout'
    assert run_episode(sea, [(2, 2), (8, 2)], [], toward, cancel=lambda: True)['status'] == 'cancelled'
    assert run_episode(sea, [(2, 2)], [], toward)['status'] == 'success'


def test_swept_crossing_and_stopped_ship_are_checked():
    ship = Traffic((5, 2), (5, 8), speed=12, radius=.1)
    assert swept_clearance((2, 5), (8, 5), ship, 0, .5, .1) < 0
    stopped = Traffic((5, 4), (5, 5), speed=10, radius=.1)
    assert swept_clearance((2, 5), (8, 5), stopped, 0, 1, .1) < 0


def test_scenarios_are_seeded_and_not_recreated_at_waypoints():
    sea = SeaMap((0, 0, 24, 24))
    route = [(2, 2), (12, 12), (22, 22)]
    a = traffic_for_route(sea, route, 5, 7)
    b = traffic_for_route(sea, route, 5, 7)
    assert a == b and len(a) == 5
    result = run_episode(sea, route, a, toward)
    for frame in result['frames']:
        assert frame['traffic'] == [list(ship.at(frame['t'])[0]) for ship in a]
```

- [ ] **Step 2: Run the tests before implementation.**

```bash
python -m pytest tests/test_simulation.py -v
```

Expected: failure because the new module or behaviour does not exist. Missing environment dependencies must be resolved before treating a test failure as evidence about application behaviour.

- [ ] **Step 3: Implement the following files.**

Create `rebuild/src/shipnav/simulation.py` with:

```python
from dataclasses import dataclass
from math import dist, hypot, isfinite
from random import Random
from time import perf_counter
from typing import Callable
from shipnav.maps import Point, SeaMap

Neighbour = tuple[Point, Point, float]
Policy = Callable[[Point, Point, Point, list[Neighbour], float, float, float], Point]


@dataclass(frozen=True)
class Traffic:
    start: Point
    goal: Point
    speed: float = .3
    radius: float = .6

    def __post_init__(self):
        if (not all(isfinite(x) for x in (*self.start, *self.goal, self.speed, self.radius))
                or self.speed <= 0 or self.radius <= 0 or self.start == self.goal):
            raise ValueError('Traffic requires finite distinct endpoints, positive speed and radius')

    @property
    def arrival(self) -> float:
        return dist(self.start, self.goal) / self.speed

    def at(self, time: float) -> Neighbour:
        fraction = min(1.0, max(0.0, time / self.arrival))
        p = tuple(a + fraction*(b-a) for a, b in zip(self.start, self.goal))
        v = tuple((b-a)/self.arrival if fraction < 1 else 0.0
                  for a, b in zip(self.start, self.goal))
        return p, v, self.radius


def traffic_for_route(sea: SeaMap, route: list[Point], count: int, seed: int) -> list[Traffic]:
    if count < 0:
        raise ValueError('Traffic count cannot be negative')
    if count == 0:
        return []
    if len(route) < 2:
        raise ValueError('Moving traffic requires a nonzero route')
    rng, ships = Random(seed), []
    for index in range(count):
        a, b = route[index % (len(route)-1):][:2]
        length = dist(a, b)
        if length == 0:
            raise ValueError('Repeated route points')
        normal = (-(b[1]-a[1])/length, (b[0]-a[0])/length)
        for _ in range(500):
            f, width = rng.uniform(.15, .85), rng.uniform(1.5, 4.0)
            center = tuple(x + f*(y-x) for x, y in zip(a, b))
            start = tuple(x + width*n for x, n in zip(center, normal))
            goal = tuple(x - width*n for x, n in zip(center, normal))
            if rng.random() < .5:
                start, goal = goal, start
            if not sea.clear(start, goal, .6):
                continue
            if dist(start, route[0]) <= 1.3 or dist(start, route[-1]) <= 1.3:
                continue
            if any(dist(start, other.start) <= 1.4 for other in ships):
                continue
            ships.append(Traffic(start, goal))
            break
        else:
            raise ValueError('Cannot place traffic on this route; lower count or change map')
    return ships


def segment_distance(a: Point, b: Point) -> float:
    delta = (b[0]-a[0], b[1]-a[1])
    squared = delta[0]**2 + delta[1]**2
    u = min(1.0, max(0.0, -(a[0]*delta[0]+a[1]*delta[1])/squared)) if squared else 0.0
    return hypot(a[0]+u*delta[0], a[1]+u*delta[1])


def swept_clearance(a: Point, b: Point, ship: Traffic, time: float, dt: float, radius: float) -> float:
    cuts = sorted({time, time+dt, min(time+dt, max(time, ship.arrival))})
    values = []
    for left, right in zip(cuts, cuts[1:]):
        relative = []
        for t in (left, right):
            ego = tuple(x+(t-time)/dt*(y-x) for x, y in zip(a, b))
            other = ship.at(t)[0]
            relative.append(tuple(x-y for x, y in zip(ego, other)))
        values.append(segment_distance(*relative)-radius-ship.radius)
    return min(values)


def run_episode(sea: SeaMap, route: list[Point], traffic: list[Traffic], policy: Policy,
                dt: float = .25, limit: float = 100, radius: float = .5,
                speed: float = 1.0, cancel: Callable[[], bool] = lambda: False) -> dict:
    if not all(isfinite(v) and v > 0 for v in (dt, limit, radius, speed)):
        raise ValueError('Episode parameters must be positive and finite')
    if not route or any(not sea.clear(p, p, radius) for p in route):
        raise ValueError('Missing route or blocked waypoint')
    for i, ship in enumerate(traffic):
        if not sea.clear(ship.start, ship.goal, ship.radius):
            raise ValueError('Traffic crosses land')
        if dist(ship.start, route[0]) <= radius+ship.radius:
            raise ValueError('Traffic overlaps start')
        if any(dist(ship.start, other.start) <= ship.radius+other.radius for other in traffic[:i]):
            raise ValueError('Traffic overlaps traffic at start')
    position, velocity, time, index = route[0], (0.0, 0.0), 0.0, 1
    minimum, latencies, frames, travelled = float('inf'), [], [], 0.0

    def record():
        frames.append({'t': time, 'position': list(position), 'velocity': list(velocity),
                       'goal_index': min(index, len(route)-1),
                       'traffic': [list(ship.at(time)[0]) for ship in traffic]})

    record()
    status = 'success' if len(route) == 1 else 'running'
    while status == 'running':
        if cancel():
            status = 'cancelled'
            break
        if time >= limit - 1e-9:
            status = 'timeout'
            break
        step = min(dt, limit-time)
        neighbours = [ship.at(time) for ship in traffic]
        began = perf_counter()
        velocity = tuple(policy(position, velocity, route[index], neighbours, radius, speed, step))
        latencies.append((perf_counter()-began)*1000)
        if len(velocity) != 2 or not all(isfinite(x) for x in velocity) or hypot(*velocity) > speed+1e-6:
            raise ValueError('Policy returned invalid velocity')
        next_position = tuple(x + step*v for x, v in zip(position, velocity))
        clearance = min((swept_clearance(position, next_position, ship, time, step, radius)
                         for ship in traffic), default=float('inf'))
        minimum = min(minimum, clearance)
        collision = clearance <= 0 or not sea.clear(position, next_position, radius)
        travelled += dist(position, next_position)
        position, time = next_position, time+step
        if collision:
            status = 'collision'
        elif dist(position, route[index]) < radius:
            index += 1
            if index == len(route):
                status = 'success'
        record()
    return {'status': status, 'elapsed': time, 'distance': travelled,
            'min_dynamic_clearance': minimum if isfinite(minimum) else None,
            'inference_ms': latencies, 'frames': frames}
```

- [ ] **Step 4: Run the same test command and inspect the result.**

```bash
python -m pytest tests/test_simulation.py -v
```

Expected: PASS. Do not weaken assertions to conceal an algorithm or interface failure.

- [ ] **Step 5: Review the task's diff and save a checkpoint.**

From the workspace root, if it is a Git repository:

```bash
git add rebuild/src/shipnav/simulation.py rebuild/tests/test_simulation.py
git commit -m "feat: simulate continuous episodes and moving traffic"
```

### Task 5: Adapt learned and reciprocal policies

**Files:** `rebuild/src/shipnav/policies.py`, `rebuild/tests/test_policies.py`.

**Interfaces:** Consumes `load_policy(Path) -> SARL` from plan 01b and the policy callable interface in Task 4. Produces `Direct()`, `Learned(model_dir: Path)`, `Reciprocal()`. Empty crowds use direct motion explicitly; only nonempty-crowd SARL steps invoke the neural network.

- [ ] **Step 1: Write the behaviour tests.**

Create `rebuild/tests/test_policies.py` with:

```python
from pathlib import Path
from math import hypot, isfinite
import pytest
from shipnav.policies import Direct, Learned, Reciprocal

MODEL = Path(__file__).resolve().parents[2] / 'CrowdNav-20250813-DIP/crowd_nav/data/output_trained'


def test_direct_stops_exactly_and_respects_speed():
    direct = Direct()
    assert direct((2, 2), (0, 0), (2, 2), [], .5, 1, .25) == (0, 0)
    assert hypot(*direct((2, 2), (0, 0), (8, 8), [], .5, 1, .25)) == pytest.approx(1)


@pytest.mark.integration
@pytest.mark.parametrize('kind', ['sarl', 'orca'])
def test_adapters_handle_empty_and_nonempty_neighbours(kind):
    policy = Learned(MODEL) if kind == 'sarl' else Reciprocal()
    assert policy((2, 2), (0, 0), (4, 2), [], .5, 1, .25) == (1, 0)
    velocity = policy((2, 2), (0, 0), (8, 2), [((5, 3), (0, -.3), .6)], .5, 1, .25)
    assert all(isfinite(v) for v in velocity)
    assert hypot(*velocity) <= 1.000001
```

- [ ] **Step 2: Run the tests before implementation.**

```bash
python -m pytest tests/test_policies.py -v
```

Expected: failure because the new module or behaviour does not exist. Missing environment dependencies must be resolved before treating a test failure as evidence about application behaviour.

- [ ] **Step 3: Implement the following files.**

Create `rebuild/src/shipnav/policies.py` with:

```python
from math import dist
from pathlib import Path


class Direct:
    def __call__(self, position, velocity, goal, neighbours, radius, speed, dt):
        distance = dist(position, goal)
        magnitude = min(speed, distance/dt)
        return tuple((b-a)/distance*magnitude if distance else 0.0
                     for a, b in zip(position, goal))


class Learned:
    def __init__(self, model_dir: Path):
        from shipnav.model import load_policy
        self.policy = load_policy(model_dir)
        self.policy.query_env = False

    def __call__(self, position, velocity, goal, neighbours, radius, speed, dt):
        import torch
        from shipnav.compat.crowd_sim.envs.utils.state import FullState, ObservableState, JointState
        if not neighbours:
            return Direct()(position, velocity, goal, neighbours, radius, speed, dt)
        self.policy.time_step = dt
        state = FullState(*position, *velocity, radius, *goal, speed, 0.0)
        others = [ObservableState(*p, *v, r) for p, v, r in neighbours]
        with torch.inference_mode():
            action = self.policy.predict(JointState(state, others))
        return float(action.vx), float(action.vy)


class Reciprocal:
    def __init__(self):
        from shipnav.compat.crowd_sim.envs.policy.orca import ORCA
        self.policy = ORCA()

    def __call__(self, position, velocity, goal, neighbours, radius, speed, dt):
        from shipnav.compat.crowd_sim.envs.utils.state import FullState, ObservableState, JointState
        if not neighbours:
            return Direct()(position, velocity, goal, neighbours, radius, speed, dt)
        self.policy.time_step = dt
        state = FullState(*position, *velocity, radius, *goal, speed, 0.0)
        others = [ObservableState(*p, *v, r) for p, v, r in neighbours]
        action = self.policy.predict(JointState(state, others))
        return float(action.vx), float(action.vy)
```

- [ ] **Step 4: Run the same test command and inspect the result.**

```bash
python -m pytest tests/test_policies.py -v
```

Expected: PASS. Do not weaken assertions to conceal an algorithm or interface failure.

- [ ] **Step 5: Review the task's diff and save a checkpoint.**

From the workspace root, if it is a Git repository:

```bash
git add rebuild/src/shipnav/policies.py rebuild/tests/test_policies.py
git commit -m "feat: adapt learned and reciprocal policies"
```

### Task 6: Expose a shared navigation service and CLI

**Files:** `rebuild/src/shipnav/service.py`, `rebuild/tests/test_service.py`.

**Interfaces:** Consumes all prior headless modules. Produces `execute(map_data, start, goal, model_dir="", policy_name="sarl", global_goals=True, seed=0, count=5, cancel=callable) -> dict`. JSON-safe traces include original generated route and actual active goals; paired variants consume traffic generated independently of routing; complete Task 6b before executing the service. Invalid maps/planning fail explicitly through exceptions/CLI exit 2.

- [ ] **Step 1: Write the behaviour tests.**

Create `rebuild/tests/test_service.py` with:

```python
from shipnav.service import execute
from shipnav.maps import SeaMap
import pytest


def test_service_is_deterministic_and_records_its_settings():
    sea = SeaMap((0, 0, 24, 24)).to_dict()
    a = execute(sea, (2, 2), (22, 22), policy_name='direct', count=0)
    b = execute(sea, (2, 2), (22, 22), policy_name='direct', count=0)
    assert a['frames'] == b['frames']
    assert a['status'] == 'success'
    assert a['settings']['query_env'] is False
    assert a['model_hashes'] == {}


def test_paired_variants_keep_identical_scenarios():
    sea = SeaMap((0, 0, 24, 24), ((9, 5, 14, 18),)).to_dict()
    a = execute(sea, (2, 2), (22, 22), policy_name='direct', seed=3, count=3)
    b = execute(sea, (2, 2), (22, 22), policy_name='direct', global_goals=False, seed=3, count=3)
    assert a['traffic_definitions'] == b['traffic_definitions']
    assert a['goals'] != b['goals']
    assert b['status'] == 'collision'


def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError, match='Policy must'):
        execute(SeaMap((0, 0, 24, 24)).to_dict(), (2, 2), (22, 22), policy_name='missing', count=0)
```

- [ ] **Step 2: Run the tests before implementation.**

```bash
python -m pytest tests/test_service.py -v
```

Expected: failure because the new module or behaviour does not exist. Missing environment dependencies must be resolved before treating a test failure as evidence about application behaviour.

- [ ] **Step 3: Implement the following files.**

Create `rebuild/src/shipnav/service.py` with:

```python
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
```

- [ ] **Step 4: Run the same test command and inspect the result.**

```bash
python -m pytest tests/test_service.py -v
```

Expected: PASS. Do not weaken assertions to conceal an algorithm or interface failure.

- [ ] **Step 5: Review the task's diff and save a checkpoint.**

From the workspace root, if it is a Git repository:

```bash
git add rebuild/src/shipnav/service.py rebuild/tests/test_service.py
git commit -m "feat: expose a shared navigation service and cli"
```

### Completion gate

- [ ] Run `python -m pytest tests/test_simulation.py tests/test_policies.py tests/test_service.py -v`.
- [ ] Run `python -m shipnav.service --policy direct --count 0 --output results/direct.json`.
- [ ] Run `python -m shipnav.service --policy sarl --count 5 --output results/sarl.json`.
- [ ] Inspect actual success/collision/timeout status and exported metadata. Do not replace a collision with a successful status to make the demo look better. The supplied SARL model has not learned land geometry. Complete mandatory Task 6b safety/perception integration and plan 03b before running the final comparisons; retain this unfiltered result as an ablation.

The preceding Tasks 4–6 provide the minimal holonomic reference. The route-dependent `traffic_for_route` helper and its test are historical scaffolding only: remove both at Task 6b; retain continuous-motion and swept-collision tests. The accepted phase includes the following mandatory revisions, not just that minimal reference.

### Task 6b — Freeze scenarios and separate perception/prediction/safety

**Files:** `src/shipnav/scenarios.py`, `observations.py`, `prediction.py`, `safety.py`, `tests/test_safety_pipeline.py`

**Interfaces:** `make_scenario(sea,start,goal,count,seed)->dict`, `scenario_hash(dict)->str`, `load_traffic(dict)->list[Traffic]`; `Observer.observe(traffic,t,tick)->list[dict]`; `predict(observed,dt,steps=12,uncertainty=True,growth=.15)->list[dict]`; `choose(...)->decision dict`. None uses simulator future truth to choose ego action.

- [ ] Write the behavior test:

File: `rebuild/tests/test_safety_pipeline.py`

```python
from shipnav.maps import SeaMap
from shipnav.scenarios import make_scenario, scenario_hash
from shipnav.observations import Observer
from shipnav.simulation import Traffic
from shipnav.safety import choose

def test_scenario_ignores_planner_and_hash_detects_changes():
    sea=SeaMap((0,0,24,24))
    a=make_scenario(sea,(2,2),(22,22),4,9)
    b=make_scenario(sea,(2,2),(22,22),4,9)
    assert scenario_hash(a)==scenario_hash(b)
    b['seed']=10
    assert scenario_hash(a)!=scenario_hash(b)

def test_noise_is_keyed_and_does_not_mutate_truth():
    traffic=[Traffic((4,4),(9,4))]
    truth=traffic[0].at(1.)
    a=Observer(7,noise=.2).observe(traffic,1.,4)
    b=Observer(7,noise=.2).observe(traffic,1.,4)
    assert a==b and tuple(a[0]['position'])!=truth[0]
    assert traffic[0].at(1.)==truth

def test_filter_prevents_land_command_and_reports_no_solution():
    sea=SeaMap((0,0,10,10),((4,0,6,10),))
    r=choose(sea,(3,5),(1,0),[],.5,1,.25)
    assert r['override'] and not r['no_feasible_action']
    assert all(sea.clear(a,b,.5) for a,b in zip(r['path'],r['path'][1:]))
    impossible=[{'radius':10.,'points':[(3,5)]*13,'margins':[0.]*13}]
    r=choose(sea,(3,5),(1,0),impossible,.5,1,.25)
    assert r['no_feasible_action']
```

- [ ] Run `python -m pytest tests/test_safety_pipeline.py -v`. Expect a missing module or unmet behavior, not a missing third-party dependency.
- [ ] Implement the following content:

File: `rebuild/src/shipnav/scenarios.py`

```python
from dataclasses import asdict
from hashlib import sha256
from random import Random
from math import dist
import json
from shipnav.simulation import Traffic

def scenario_hash(data):
    return sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def make_scenario(sea, start, goal, count, seed):
    if count < 0:
        raise ValueError('Traffic count cannot be negative')
    rng, ships = Random(seed), []
    x0, y0, x1, y1 = sea.bounds
    for _ in range(count):
        for attempt in range(500):
            a = (rng.uniform(x0+.7, x1-.7), rng.uniform(y0+.7, y1-.7))
            b = (rng.uniform(x0+.7, x1-.7), rng.uniform(y0+.7, y1-.7))
            if dist(a,b) < 1 or not sea.clear(a,b,.6):
                continue
            if min(dist(a,start), dist(a,goal)) <= 1.3:
                continue
            if any(dist(a,s.start) <= 1.4 for s in ships):
                continue
            ships.append(Traffic(a,b)); break
        else:
            raise ValueError('Scenario placement failed; preserve this failed seed')
    return {'schema': 1, 'seed': seed, 'map': sea.to_dict(), 'start': list(start),
            'goal': list(goal), 'traffic': [asdict(s) for s in ships],
            'traffic_mode': 'scripted', 'family': 'random', 'split': 'smoke'}

def load_traffic(data):
    return [Traffic(tuple(s['start']), tuple(s['goal']), s['speed'], s['radius']) for s in data['traffic']]
```

File: `rebuild/src/shipnav/observations.py`

```python
from hashlib import sha256
from random import Random
from math import isfinite

class Observer:
    def __init__(self, seed=0, noise=0., delay=0., dropout=0., stale_speed_bound=2.):
        if not all(isfinite(x) for x in (noise,delay,dropout,stale_speed_bound)) or min(noise,delay,stale_speed_bound)<0 or not 0<=dropout<=1:
            raise ValueError('Invalid observation configuration')
        self.seed, self.noise, self.delay = seed, noise, delay
        self.dropout, self.bound, self.last = dropout, stale_speed_bound, {}

    def observe(self, traffic, t, tick):
        result = []
        for target, ship in enumerate(traffic):
            key = sha256(f'{self.seed}:{tick}:{target}'.encode()).digest()
            rng = Random(int.from_bytes(key, 'big'))
            stamp = max(0., t-self.delay)
            if rng.random() >= self.dropout:
                p,v,r = ship.at(stamp)
                measured = tuple(x+rng.gauss(0,self.noise) for x in p)
                self.last[target] = (measured,v,r,stamp)
            if target in self.last:
                p,v,r,stamp = self.last[target]
                age = t-stamp
                estimate = tuple(x+age*y for x,y in zip(p,v))
                # Bound missed motion conservatively; Gaussian margin is not a hard bound.
                result.append({'id': target, 'position': estimate, 'velocity': v,
                               'radius': r, 'age': age,
                               'margin': 3*self.noise+2*self.bound*age})
        return result
```

File: `rebuild/src/shipnav/prediction.py`

```python
def predict(observed, dt, steps=12, uncertainty=True, growth=.15):
    if dt<=0 or steps<1 or growth<0:
        raise ValueError('Invalid prediction horizon')
    return [{'id': s['id'], 'radius': s['radius'],
             'points': [[x+k*dt*v for x,v in zip(s['position'],s['velocity'])] for k in range(steps+1)],
             'margins': [(s['margin']+growth*k*dt) if uncertainty else 0. for k in range(steps+1)]}
            for s in observed]
```

File: `rebuild/src/shipnav/safety.py`

```python
from math import cos, sin, pi, hypot, dist

def relative_clearance(a, b, c, d, radius):
    x,y = a[0]-c[0], a[1]-c[1]
    u,v = b[0]-d[0]-x, b[1]-d[1]-y
    f = max(0., min(1., -(x*u+y*v)/(u*u+v*v))) if u*u+v*v else 0.
    return hypot(x+f*u,y+f*v)-radius

def rollout(position, desired, dt, steps, motion=None):
    if motion is not None:
        return motion(desired, dt, steps)
    return [tuple(x+k*dt*v for x,v in zip(position,desired)) for k in range(steps+1)]

def assess(sea, path, predictions, radius):
    margin = float('inf')
    for k,(a,b) in enumerate(zip(path,path[1:])):
        if not sea.clear(a,b,radius):
            margin = min(margin,-1e6)
        for target in predictions:
            extra = max(target['margins'][k:k+2])
            margin = min(margin, relative_clearance(a,b,*target['points'][k:k+2],radius+target['radius']+extra))
    return margin

def choose(sea, position, nominal, predictions, radius, speed, dt, steps=12, motion=None):
    candidates = [tuple(nominal),(0.,0.)]
    candidates += [(s*cos(k*pi/8),s*sin(k*pi/8)) for s in (speed*.5,speed) for k in range(16)]
    scored = []
    for command in candidates:
        path = rollout(position,command,dt,steps,motion)
        clearance = assess(sea,path,predictions,radius)
        scored.append((command,clearance,path))
    feasible = [x for x in scored if x[1] > 0]
    chosen = min(feasible,key=lambda x:dist(x[0],nominal)) if feasible else max(scored,key=lambda x:(x[1],-dist(x[0],nominal)))
    return {'executed': chosen[0], 'path': chosen[2],
            'override': dist(chosen[0],nominal)>1e-9,
            'no_feasible_action': not feasible,
            'predicted_clearance': None if chosen[1]==float('inf') else chosen[1]}
```

- [ ] Run `python -m pytest tests/test_safety_pipeline.py -v` again. Inspect failures before proceeding.
- [ ] Review and commit only the implementation/test files named above after the check passes; keep `docs/` ignored.

### Task 6c — Integrate perception and safety before truth advances

**Files:** replace `src/shipnav/simulation.py`; extend `tests/test_simulation.py` and `tests/test_safety_pipeline.py`.

- [ ] Keep the original continuous-clock, swept-collision, cancellation and timeout tests. Remove only the route-based generation test and import. Add this behavioral test, run it failing, then replace the simulator with the code below and rerun both test modules.

File: `rebuild/tests/test_safety_pipeline.py (append)`

```python
from shipnav.simulation import run_episode
from shipnav.policies import Direct

def test_filter_is_in_the_episode_loop():
    sea=SeaMap((0,0,10,10),((4,0,6,10),))
    bare=run_episode(sea,[(2,5),(8,5)],[],Direct(),limit=4)
    safe=run_episode(sea,[(2,5),(8,5)],[],Direct(),limit=4,filtered=True)
    assert bare['status']=='collision'
    assert safe['status']=='timeout'
    assert any(d['override'] for d in safe['diagnostics'])
    assert not any(d['land_collision'] for d in safe['diagnostics'])
```

File: `rebuild/src/shipnav/simulation.py (replacement)`

```python
from dataclasses import dataclass
from math import dist, hypot, isfinite
from random import Random
from time import perf_counter
from typing import Callable
from shipnav.maps import Point, SeaMap

Neighbour = tuple[Point, Point, float]
Policy = Callable[[Point, Point, Point, list[Neighbour], float, float, float], Point]


@dataclass(frozen=True)
class Traffic:
    start: Point
    goal: Point
    speed: float = .3
    radius: float = .6

    def __post_init__(self):
        if (not all(isfinite(x) for x in (*self.start, *self.goal, self.speed, self.radius))
                or self.speed <= 0 or self.radius <= 0 or self.start == self.goal):
            raise ValueError('Traffic requires finite distinct endpoints, positive speed and radius')

    @property
    def arrival(self) -> float:
        return dist(self.start, self.goal) / self.speed

    def at(self, time: float) -> Neighbour:
        fraction = min(1.0, max(0.0, time / self.arrival))
        p = tuple(a + fraction*(b-a) for a, b in zip(self.start, self.goal))
        v = tuple((b-a)/self.arrival if fraction < 1 else 0.0
                  for a, b in zip(self.start, self.goal))
        return p, v, self.radius


def segment_distance(a: Point, b: Point) -> float:
    delta = (b[0]-a[0], b[1]-a[1])
    squared = delta[0]**2 + delta[1]**2
    u = min(1.0, max(0.0, -(a[0]*delta[0]+a[1]*delta[1])/squared)) if squared else 0.0
    return hypot(a[0]+u*delta[0], a[1]+u*delta[1])


def swept_clearance(a: Point, b: Point, ship: Traffic, time: float, dt: float, radius: float) -> float:
    cuts = sorted({time, time+dt, min(time+dt, max(time, ship.arrival))})
    values = []
    for left, right in zip(cuts, cuts[1:]):
        relative = []
        for t in (left, right):
            ego = tuple(x+(t-time)/dt*(y-x) for x, y in zip(a, b))
            other = ship.at(t)[0]
            relative.append(tuple(x-y for x, y in zip(ego, other)))
        values.append(segment_distance(*relative)-radius-ship.radius)
    return min(values)


def run_episode(sea: SeaMap, route: list[Point], traffic: list[Traffic], policy: Policy,
                dt: float = .25, limit: float = 100, radius: float = .5,
                speed: float = 1.0, cancel: Callable[[], bool] = lambda: False, *, observer=None,
                filtered=False, uncertainty=True, dynamics='holonomic') -> dict:
    if not all(isfinite(v) and v > 0 for v in (dt, limit, radius, speed)):
        raise ValueError('Episode parameters must be positive and finite')
    if not route or any(not sea.clear(p, p, radius) for p in route):
        raise ValueError('Missing route or blocked waypoint')
    for i, ship in enumerate(traffic):
        if not sea.clear(ship.start, ship.goal, ship.radius):
            raise ValueError('Traffic crosses land')
        if dist(ship.start, route[0]) <= radius+ship.radius:
            raise ValueError('Traffic overlaps start')
        if any(dist(ship.start, other.start) <= ship.radius+other.radius for other in traffic[:i]):
            raise ValueError('Traffic overlaps traffic at start')
    from shipnav.observations import Observer
    from shipnav.prediction import predict
    from shipnav.safety import choose
    observer = observer or Observer()
    diagnostics = []
    vessel = None
    if dynamics == 'marine':
        from shipnav.dynamics import Vessel, advance, motion_from
        vessel = Vessel(*route[0], 0., 0.)
    elif dynamics != 'holonomic':
        raise ValueError('Unknown dynamics')
    position, velocity, time, index = route[0], (0.0, 0.0), 0.0, 1
    minimum, latencies, frames, travelled = float('inf'), [], [], 0.0

    def record():
        frames.append({'t': time, 'position': list(position), 'velocity': list(velocity),
                       'goal_index': min(index, len(route)-1),
                       'traffic': [list(ship.at(time)[0]) for ship in traffic]})

    record()
    status = 'success' if len(route) == 1 else 'running'
    while status == 'running':
        if cancel():
            status = 'cancelled'
            break
        if time >= limit - 1e-9:
            status = 'timeout'
            break
        step = min(dt, limit-time)
        began = perf_counter()
        perceived = observer.observe(traffic,time,len(frames)-1)
        predictions = predict(perceived,step,uncertainty=uncertainty)
        neighbours = [(s['position'],s['velocity'],s['radius']) for s in perceived]
        inference_start = perf_counter()
        nominal = tuple(policy(position,velocity,route[index],neighbours,radius,speed,step))
        latencies.append((perf_counter()-inference_start)*1000)
        if len(nominal)!=2 or not all(isfinite(x) for x in nominal) or hypot(*nominal)>speed+1e-6:
            raise ValueError('Policy returned invalid velocity')
        motion = motion_from(vessel,speed) if vessel is not None else None
        decision = choose(sea,position,nominal,predictions,radius,speed,step,motion=motion) if filtered else {
            'executed': nominal, 'override': False, 'no_feasible_action': False, 'path': [], 'predicted_clearance': None}
        velocity = tuple(decision['executed'])
        diagnostics.append({'t': time,'observed': perceived,'predictions': predictions,
                            'nominal': nominal,**decision,
                            'decision_ms': (perf_counter()-began)*1000})
        if len(velocity) != 2 or not all(isfinite(x) for x in velocity) or hypot(*velocity) > speed+1e-6:
            raise ValueError('Policy returned invalid velocity')
        next_position = tuple(x + step*v for x, v in zip(position, velocity))
        if vessel is not None:
            vessel = advance(vessel,velocity,step,max_speed=speed)
            next_position = (vessel.x,vessel.y)
            velocity = tuple((b-a)/step for a,b in zip(position,next_position))
            diagnostics[-1]['heading'] = vessel.heading
            diagnostics[-1]['speed'] = vessel.speed
        for ship in traffic:
            if hasattr(ship,'advance'):
                ship.advance(time,step,position)
        clearance = min((swept_clearance(position, next_position, ship, time, step, radius)
                         for ship in traffic), default=float('inf'))
        minimum = min(minimum, clearance)
        land_hit = not sea.clear(position,next_position,radius)
        ship_hit = clearance <= 0
        collision = ship_hit or land_hit
        diagnostics[-1].update(land_collision=land_hit,ship_collision=ship_hit,actual_velocity=velocity,clearance=clearance if isfinite(clearance) else None)
        travelled += dist(position, next_position)
        position, time = next_position, time+step
        if collision:
            status = 'collision'
        elif dist(position, route[index]) < radius:
            index += 1
            if index == len(route):
                status = 'success'
        record()
    return {'status': status, 'elapsed': time, 'distance': travelled,
            'min_dynamic_clearance': minimum if isfinite(minimum) else None,
            'inference_ms': latencies, 'frames': frames, 'diagnostics': diagnostics}
```

- [ ] Add crossing-between-samples and a moving target hitting a stationary ego test. Safety uses swept relative segments over each prediction interval; truth uses actual trajectories (including arrival stops). Log lack of any feasible candidate; do not fabricate success or assume zero velocity is safe. Horizon, candidate discretization and uncertain/unseen targets limit safety claims.
- [ ] Define canonical JSON scenarios before benchmarks: head-on `ego (3,12)→(21,12), target (21,12)→(3,12)`; crossing `ego (3,12)→(21,12), target (12,3)→(12,21)` and its mirrored version; overtake `ego (3,12)→(21,12), target (7,12)→(21,12), speed .3`; narrow passage `land (8,0,12,10),(8,14,12,24)`; detour the harbour map; unreachable `land (10,0,14,24)`; shore goal `(22.9,12)`; multi-conflict combines nonoverlapping crossings. Save the complete map, start/goal and `traffic` list, not a planner-specific seed recipe. Plan 03b adds course-changing/reactive targets.
- [ ] Freeze `scenarios/splits.json` with train/dev/calibration/test filenames and SHA256 hashes. Use separate maps and random seeds per split; designate withheld encounter families in a separate generalization report. Scenario validation failures are reported, not resampled differently per algorithm. Noise uses fixed `(scenario seed,tick,target ID)` keys; future targets need stable IDs rather than indices if insertion/removal is later allowed.

### Task 6d — Final shared service contract

After Tasks 6b–6c and plan 03b, replace the minimal Task 6 service with this final version. Existing GUI positional calls remain valid and intentionally default to the unfiltered holonomic baseline; the revised GUI exposes the additional choices. Benchmark callers supply `scenario=` explicitly and persist it before planning, including planner failures.

File: `rebuild/src/shipnav/service.py (replacement)`

```python
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json
from shipnav.maps import SeaMap
from shipnav.planning import astar, smooth
from shipnav.simulation import run_episode
from shipnav.scenarios import make_scenario, load_traffic, scenario_hash
from shipnav.observations import Observer
from shipnav.theta import theta_star
from shipnav.policies import Direct, Learned, Reciprocal


def execute(map_data: dict, start: tuple, goal: tuple, model_dir: str = '',
            policy_name: str = 'sarl', global_goals: bool = True, seed: int = 0,
            count: int = 5, cancel=lambda: False, *, scenario=None, planner='astar_smooth',
            filtered=False, uncertainty=True, observation=None, dynamics='holonomic') -> dict:
    sea = SeaMap.from_dict(map_data)
    scenario = make_scenario(sea,start,goal,count,seed) if scenario is None else scenario
    if scenario['map']!=map_data or tuple(scenario['start'])!=tuple(start) or tuple(scenario['goal'])!=tuple(goal):
        raise ValueError('Scenario map/endpoints differ from run request')
    if planner == 'astar_smooth':
        route = smooth(sea,astar(sea,tuple(start),tuple(goal)))
    elif planner == 'theta':
        route = theta_star(sea,tuple(start),tuple(goal))
    else:
        raise ValueError('Unknown planner')
    traffic = load_traffic(scenario)
    if scenario.get('traffic_mode')=='reactive':
        from shipnav.reactive import ReactiveTraffic
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
    result = run_episode(sea,goals,traffic,policy,cancel=cancel,observer=observer,filtered=filtered,uncertainty=uncertainty,dynamics=dynamics)
    result.update({'schema': 2, 'scenario': scenario, 'scenario_hash': scenario_hash(scenario), 'map': map_data, 'route': route, 'goals': goals,
                   'traffic_definitions': scenario['traffic'],
                   'settings': {'seed': scenario['seed'], 'planner': planner, 'filtered': filtered, 'uncertainty': uncertainty, 'observation': observation or {}, 'dynamics': dynamics, 'requested_traffic': count,
                                'actual_traffic': len(traffic), 'policy': policy_name,
                                'global_goals': global_goals, 'dt': .25, 'limit': 100,
                                'radius': .5, 'speed': 1.0, 'query_env': False,
                                'traffic_model': scenario['traffic_mode']},
                   'model_hashes': hashes})
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
    args = parser.parse_args()
    scenario = json.loads(args.scenario.read_text()) if args.scenario else None
    if scenario:
        args.start,args.goal=scenario['start'],scenario['goal']
    try:
        result = execute(scenario['map'] if scenario else SeaMap.load(args.map).to_dict(), tuple(args.start), tuple(args.goal),
                         args.model, args.policy, not args.no_global_goals, args.seed, args.count,
                         scenario=scenario,planner=args.planner,filtered=args.filtered,dynamics=args.dynamics)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.exit(2, str(error)+'\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(result['status'], result['elapsed'])
```

- [ ] Update the paired service test to compare `scenario_hash` across A*, Theta*, guidance on/off and filter on/off. Supply the same `scenario` object to each call. Assert the object is unchanged after execution and metadata includes every selected control.
- [ ] Retain invalid-policy, deterministic-frame and invalid-map tests. A failed route raises `NoPath`; the benchmark records `planning_failure` with the already saved scenario/hash. The GUI shows the same exception visibly.
- [ ] Record application Git revision, dirty-diff hash, config hash, Python/platform and lockfile hash alongside `model_hashes` before accepting benchmark artifacts. Use SHA256 over the serialized settings and relevant lock files; generated timing fields are not part of scenario identity.
