# 05 — Prediction-aware retraining and AWS CLI EC2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate modern policies on the selected marine environment with reproducible, budgeted EC2 execution.

**Architecture:** Share episode setup between the service and a generator-based step engine, then expose that engine as a bounded Gymnasium environment over a large, procedurally generated training corpus that is disjoint from every frozen evaluation case. Train PPO and a Lagrangian domain-cost ablation in resumable chunks, select checkpoints on a generated selection split, then evaluate under a newly preregistered protocol that re-runs the matched comparators on the same source. EC2 runs CPU-first, chosen by a measured pilot.

**Tech Stack:** Python 3.14, the committed `rebuild/uv.lock` (NumPy, PyTorch, Gymnasium 1.3.0, Stable-Baselines3 2.9.0, Shapely, pyproj), AWS CLI v2 (EC2, SSM, S3, Pricing, EventBridge Scheduler)

**Spec:** [Revised design](../specs/2026-09-27-solo-rebuild-design.md)

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

All implementation paths below are relative to `rebuild/`. Run commands there with `.venv-modern/bin/python` (written `python` below). These are instructions and reference code, not installed software or benchmark results.

## Prerequisites and decisions recorded in this revision

- **Completed before this phase:** plans 01–04 and the preregistered 110-scenario CPU baseline (`evidence/baseline/`, protocol commit `8611fa0`). The current runner intentionally rejects that protocol once source hashes change, so this phase **does not compare against the archived baseline numbers**. Task 12 freezes a new protocol that re-runs the matched comparators on the retraining source.
- **Supplied-source update:** `CrowdNav-20251014-DIP/` (untracked) differs from `CrowdNav-20250813-DIP/` only in `env.config`, `crowd_nav/test.py` and `crowd_sim/envs/crowd_sim.py` (`migration/supplied-20250813-to-20251014.diff`). The SARL checkpoint bytes are unchanged, and the rebuild never executes those legacy scripts. This phase keeps `CrowdNav-20250813-DIP/` as the checkpoint reference. Re-check `rl_model.pth`/`policy.config` SHA256 against `identities.model` in Task 12 before freezing.
- **Planning documents are tracked** (commit `a7b8142`), although the copied global constraint still says ignored. Commit edits to this plan with docs-only commits.
- **New policy, not reshaped weights:** the SARL checkpoint is reused only as the `marine_sarl` comparator through its original adapter. PPO is a reproducible modern baseline, not a claim that it is the newest or best maritime algorithm.
- **Training data is generated, not the historical 10-file `train` split.** Training on `train_1000…1009` would memorise ten synthetic scenes. Task 4 generates 5,000 training and 200 selection scenarios on procedural 24–48-unit island maps, from seed ranges disjoint from every frozen manifest. `maps/harbour.json`, Ubin (development/calibration only) and Southern Islands (held out) are **never** used for training. Real-map held-out performance is therefore a zero-shot geographic-transfer result and must be reported as such.
- **Sensing:** the primary training runs use exact sensing (`observation={}`) to match the `marine_mpc` comparator. Degraded-sensing training is an optional, separately labelled extra run, never mixed into the primary comparison.
- **Hardware:** SB3 documents that PPO "is meant to be run primarily on the CPU" without a CNN and recommends `SubprocVecEnv` ([SB3 PPO docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)). CPU instances with parallel environments are the default. A GPU is chosen only if the Task 10 pilot shows lower cost per million environment steps.
- **One lock:** `uv.lock` already resolves `linux/x86_64` (including CUDA wheels). Do not create `environments/ec2/`. The training host runs `uv sync --locked --extra train --extra maps`, because `shipnav.maps` imports Shapely and pyproj at import time.

## File map

| Path | Responsibility | Task |
|---|---|---|
| `tools/engine_parity.py`, `tests/fixtures/engine_parity.json`, `tests/test_engine_parity.py` | Pre-refactor physics digests and their regression test | 1 |
| `src/shipnav/episode.py`, `tests/test_episode.py`; modify `src/shipnav/service.py` | Route, traffic and time-limit setup shared by service and training | 2 |
| modify `src/shipnav/simulation.py`; `tests/test_episode_steps.py` | Generator step engine; `run_episode` becomes its driver | 3 |
| `src/shipnav/training/__init__.py`, `training/scenarios.py`, `tools/generate_training_scenarios.py`, `scenarios/training/`, `tests/test_training_scenarios.py` | Disjoint procedural corpus | 4 |
| `src/shipnav/training/observation.py`, `tests/test_observation_encoding.py` | Bounded observation schema 1 and action transform | 5 |
| `src/shipnav/training/env.py`, `tests/test_training_env.py` | Gymnasium contract, reward, truth-based domain cost | 6 |
| `src/shipnav/training/checkpoints.py`, `tests/test_checkpoint.py` | Atomic, hash-verified full-state checkpoints | 7 |
| `src/shipnav/training/train.py`, `tests/test_train.py` | Chunked PPO/Lagrangian trainer with vector envs, SIGTERM and deadline | 8 |
| `src/shipnav/training/adapter.py`, `tests/test_modern_learned.py`; modify `simulation.py`, `service.py`, `gui.py`, `tests/test_gui.py` | `ppo` policy in service, CLI and GUI | 9 |
| `tools/training_pilot.py`, `tools/freeze_training_config.py`, `evidence/retraining/training_config.json` | Throughput pilot and preregistered training settings | 10 |
| `tools/select_checkpoints.py`, `tests/test_select_checkpoints.py` | Selection-split checkpoint choice | 11 |
| modify `tools/baseline_run.py`, `tests/test_baseline_guards.py`; `tools/freeze_retraining_protocol.py`, `tools/retrain_eval.py`, `tests/test_retrain_eval.py`, `evidence/retraining/protocol.json` | Generalised runner and retraining preregistration | 12 |
| `cloud/prepare_launch.py`, `cloud/ssm_commands.py`, `cloud/budget.py`, `cloud/run-training.sh`, `tests/test_cloud.py` | Launch JSON, SSM documents, cost estimate, host runner | 13 |
| `cloud/local/` (ignored) | Execution-time account/resource records | 14 |
| `README.md`, `evidence/retraining/README.md` | User instructions and results | 15 |

`cloud/local/` and `results/` are already in `rebuild/.gitignore`.

---

### Task 1: Freeze pre-refactor physics digests

**Files:**
- Create: `tools/engine_parity.py`, `tests/fixtures/engine_parity.json`, `tests/test_engine_parity.py`

**Interfaces:**
- Consumes: `shipnav.service.execute`, `shipnav.scale.map_options`, frozen `scenarios/*.json`
- Produces: `CASES: dict[str, tuple]`, `digest(case: str) -> str`. Tasks 2, 3 and 9 rely on this test staying green.

This must run on the **unchanged** source, before Tasks 2 and 3 touch `service.py`/`simulation.py`.

- [ ] **Step 1: Write the digest tool**

File: `tools/engine_parity.py`

```python
"""Timing-free digests of complete episodes, captured before the step-engine refactor.

Run `python tools/engine_parity.py` once on the pre-refactor source to write the
fixture; afterwards tests/test_engine_parity.py asserts the digests never change.
"""
from hashlib import sha256
from pathlib import Path
import json

from shipnav.scale import map_options
from shipnav.service import execute

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/'tests/fixtures/engine_parity.json'
SARL = ROOT.parent/'CrowdNav-20250813-DIP/crowd_nav/data/output_trained'
DEGRADED = {'noise': .1, 'delay': .5, 'dropout': .1}
TIMING = {'decision_ms', 'deadline_miss'}
# name: (scenario file, policy, dynamics, filtered, observation)
CASES = {
    'crossing-direct-holonomic': ('crossing.json', 'direct', 'holonomic', False, {}),
    'crossing-direct-filtered': ('crossing.json', 'direct', 'holonomic', True, {}),
    'narrow-direct-filtered-degraded': ('narrow_passage.json', 'direct', 'holonomic', True, DEGRADED),
    'reactive-direct-filtered': ('noncooperative_reactive.json', 'direct', 'holonomic', True, {}),
    'course-change-mpc-marine-filtered': ('course_change.json', 'mpc', 'marine', True, {}),
    'head-on-mpc-marine-unfiltered': ('head_on.json', 'mpc', 'marine', False, {}),
    'crossing-sarl-filtered': ('crossing.json', 'sarl', 'holonomic', True, {}),
}
NEEDS_CHECKPOINT = {name for name, case in CASES.items() if case[1] == 'sarl'}


def digest(name):
    file, policy, dynamics, filtered, observation = CASES[name]
    scenario = json.loads((ROOT/'scenarios'/file).read_text())
    run = execute(scenario['map'], scenario['start'], scenario['goal'], str(SARL), policy,
                  scenario=scenario, dynamics=dynamics, filtered=filtered, observation=observation,
                  **map_options(scenario['map']))
    kept = {key: run.get(key) for key in ('status', 'elapsed', 'distance', 'min_dynamic_clearance',
                                          'frames', 'terminal_speed', 'runout_min_clearance')}
    kept['diagnostics'] = [{k: v for k, v in d.items() if k not in TIMING} for d in run['diagnostics']]
    return sha256(json.dumps(kept, sort_keys=True, default=list).encode()).hexdigest()


if __name__ == '__main__':
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps({name: digest(name) for name in CASES}, indent=2, sort_keys=True)+'\n')
    print(f'Wrote {len(CASES)} digests to {FIXTURE}')
```

- [ ] **Step 2: Write the regression test**

File: `tests/test_engine_parity.py`

```python
from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from engine_parity import CASES, FIXTURE, NEEDS_CHECKPOINT, SARL, digest  # noqa: E402

EXPECTED = json.loads(FIXTURE.read_text())


@pytest.mark.parametrize('name', sorted(set(CASES)-NEEDS_CHECKPOINT))
def test_episode_physics_unchanged(name):
    assert digest(name) == EXPECTED[name]


@pytest.mark.integration
@pytest.mark.parametrize('name', sorted(NEEDS_CHECKPOINT))
def test_sarl_episode_physics_unchanged(name):
    if not (SARL/'rl_model.pth').is_file():
        pytest.skip('Supplied SARL checkpoint not present')
    assert digest(name) == EXPECTED[name]
```

- [ ] **Step 3: Run the test to confirm it fails without the fixture**

Run: `python -m pytest tests/test_engine_parity.py -v`
Expected: collection error, `FileNotFoundError` for `engine_parity.json`.

- [ ] **Step 4: Capture on the unchanged source, then re-run twice**

Run: `git status --short src/` (must print nothing), then `python tools/engine_parity.py`, then `python -m pytest tests/test_engine_parity.py -v` twice.
Expected: all pass both times. A difference between the two runs means hidden non-determinism. Stop and fix it before refactoring.

- [ ] **Step 5: Commit**

```bash
git add tools/engine_parity.py tests/fixtures/engine_parity.json tests/test_engine_parity.py
git commit -m "test: freeze timing-free episode digests before step-engine refactor"
```

### Task 2: Share episode setup between service and training

**Files:**
- Create: `src/shipnav/episode.py`, `tests/test_episode.py`
- Modify: `src/shipnav/service.py:17-72`

**Interfaces:**
- Consumes: `planning.astar/smooth`, `theta.theta_star`, `scenarios.load_traffic`, `reactive.ReactiveTraffic`
- Produces: `plan_route(sea, start, goal, planner='astar_smooth', resolution=1., clearance=.7) -> list[Point]`; `build_traffic(scenario: dict, sea) -> list`; `episode_limit(map_data: dict, route) -> float`; `SYNTHETIC_LIMIT = 100.`

- [ ] **Step 1: Write the failing tests**

File: `tests/test_episode.py`

```python
import pytest

from shipnav.episode import SYNTHETIC_LIMIT, build_traffic, episode_limit, plan_route
from shipnav.maps import SeaMap
from shipnav.reactive import ReactiveTraffic
from shipnav.scenarios import make_scenario


def test_plan_route_matches_both_planners_and_rejects_unknown():
    sea = SeaMap((0, 0, 24, 24), ((9, 5, 14, 18),))
    assert plan_route(sea, (2, 2), (22, 22))[0] == (2, 2)
    assert plan_route(sea, (2, 2), (22, 22), 'theta')[-1] == (22, 22)
    with pytest.raises(ValueError, match='Unknown planner'):
        plan_route(sea, (2, 2), (22, 22), 'dijkstra')


def test_build_traffic_wraps_reactive_and_rejects_unknown_modes():
    sea = SeaMap((0, 0, 24, 24))
    scenario = make_scenario(sea, (2, 2), (22, 22), 2, 4)
    assert len(build_traffic(scenario, sea)) == 2
    reactive = {**scenario, 'traffic_mode': 'reactive'}
    assert all(isinstance(s, ReactiveTraffic) for s in build_traffic(reactive, sea))
    with pytest.raises(ValueError, match='Unknown traffic model'):
        build_traffic({**scenario, 'traffic_mode': 'learned'}, sea)


def test_reactive_mode_rejects_course_change_entries():
    sea = SeaMap((0, 0, 24, 24))
    scenario = make_scenario(sea, (2, 12), (22, 12), 1, 5, corridor=[(2, 12), (22, 12)])
    with pytest.raises(ValueError, match='course_change'):
        build_traffic({**scenario, 'traffic_mode': 'reactive'}, sea)


def test_episode_limit_is_fixed_for_synthetic_and_route_scaled_for_real_maps():
    route = [(0, 0), (60, 0)]
    assert episode_limit({'metadata': {}}, route) == SYNTHETIC_LIMIT
    assert episode_limit({'metadata': {'model_scale': {}}}, route) == 180.
    assert episode_limit({'metadata': {'model_scale': {}}}, [(0, 0), (1, 0)]) == SYNTHETIC_LIMIT
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_episode.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'shipnav.episode'`.

- [ ] **Step 3: Implement**

File: `src/shipnav/episode.py`

```python
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
```

In `src/shipnav/service.py`, import `from shipnav.episode import build_traffic, episode_limit, plan_route`, delete the now-unused `astar`, `smooth`, `theta_star`, `load_traffic` and `CourseChangeTraffic` imports, and replace the three blocks:

```python
    route = plan_route(sea, start, goal, planner, resolution, clearance)
    if scenario is None:
        # Corridor traffic is timed on the A*-smoothed reference route whichever planner
        # runs, so planner comparisons share identical traffic.
        reference = None if placement == 'uniform' else route if planner == 'astar_smooth' else \
            plan_route(sea, start, goal, 'astar_smooth', resolution, clearance)
        scenario = make_scenario(sea,start,goal,count,seed,corridor=reference)
    # (unchanged map/endpoint equality check)
    traffic = build_traffic(scenario, sea)
    ...
    if limit is None:
        limit = episode_limit(map_data, route)
```

`plan_route` raises `ValueError('Unknown planner')` exactly as before. The placement check stays before planning.

- [ ] **Step 4: Run new and regression tests**

Run: `python -m pytest tests/test_episode.py tests/test_service.py tests/test_engine_parity.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/shipnav/episode.py src/shipnav/service.py tests/test_episode.py
git commit -m "refactor: share route, traffic and limit setup between service and training"
```

### Task 3: Generator step engine with `run_episode` as its driver

**Files:**
- Modify: `src/shipnav/simulation.py:110-239`
- Create: `tests/test_episode_steps.py`

**Interfaces:**
- Consumes: Task 2 (nothing new from it). Task 1's parity test must stay green.
- Produces:
  - `episode_steps(sea, route, traffic, policy=None, dt=.25, limit=100, radius=.5, speed=1.0, cancel=lambda: False, *, observer=None, filtered=False, uncertainty=True, dynamics='holonomic', record=True, terminal_packet=False)`. This generator yields `packet: dict`, receives `(nominal: tuple[float, float], controller_ms: float)` through `send`, and returns `(result: dict, terminal: dict | None)`.
  - The packet keys are `position, velocity, goal, goal_index, final, neighbours, observed, radius, speed, dt, t, vessel, sea, last`. `last` is `None` before the first step, then `{'clearance': float | None, 'dt': float, 'collision': bool, 'land_collision': bool, 'ship_collision': bool}`. `clearance` is the truth swept clearance to the nearest traffic ship in model units (`None` with no traffic), the same quantity `metrics._exposure` integrates.
  - `run_episode(...)` keeps its signature and result. It calls `policy.act(packet)` when the policy defines `act`, and otherwise the 7-argument call.

This is an ordinary hand refactor. Do not generate it with string replacement.

- [ ] **Step 1: Write the failing tests**

File: `tests/test_episode_steps.py`

```python
from time import sleep

import pytest

from shipnav.maps import SeaMap
from shipnav.simulation import Traffic, episode_steps, run_episode


def toward(p, v, goal, neighbours, radius, speed, dt):
    from math import dist
    d = dist(p, goal)
    return tuple((b-a)/d*min(speed, d/dt) if d else 0 for a, b in zip(p, goal))


def drive(engine, pause=0.):
    try:
        packet = next(engine)
        while True:
            sleep(pause)
            action = toward(packet['position'], packet['velocity'], packet['goal'],
                            packet['neighbours'], packet['radius'], packet['speed'], packet['dt'])
            packet = engine.send((action, 0.))
    except StopIteration as ended:
        return ended.value


def test_learner_pause_changes_neither_latency_nor_physics():
    sea = SeaMap((0, 0, 20, 20))
    route, limit = [(2, 10), (18, 10)], 2.  # 8 steps keep the deliberate pauses short
    result, _ = drive(episode_steps(sea, route, [Traffic((10, 1), (10, 19))], filtered=True, limit=limit), pause=.2)
    reference = run_episode(sea, route, [Traffic((10, 1), (10, 19))], toward, filtered=True, limit=limit)
    assert result['frames'] == reference['frames'] and result['status'] == reference['status']
    assert max(d['decision_ms'] for d in result['diagnostics']) < 200
    assert result['inference_ms'] == [0.]*len(result['inference_ms'])


def test_initially_terminal_episode_returns_without_yielding():
    with pytest.raises(StopIteration) as ended:
        next(episode_steps(SeaMap((0, 0, 10, 10)), [(2, 2)], []))
    result, terminal = ended.value.value
    assert result['status'] == 'success' and terminal is None


def test_terminal_packet_reports_final_state_on_timeout():
    engine = episode_steps(SeaMap((0, 0, 10, 10)), [(2, 2), (8, 2)], [Traffic((5, 8), (5, 1))],
                           limit=.25, terminal_packet=True)
    result, terminal = drive(engine)
    assert result['status'] == 'timeout'
    assert terminal['t'] == .25 and terminal['goal_index'] == 1 and terminal['last']['dt'] == .25
    assert [o['id'] for o in terminal['observed']] == [0]


def test_land_collision_is_reported_in_last():
    sea = SeaMap((0, 0, 10, 10), ((4, 4, 6, 6),))
    result, terminal = drive(episode_steps(sea, [(2, 5), (8, 5)], [], terminal_packet=True))
    assert result['status'] == 'collision'
    assert terminal['last']['collision'] and terminal['last']['land_collision']


def test_close_releases_a_suspended_engine():
    engine = episode_steps(SeaMap((0, 0, 10, 10)), [(2, 2), (8, 2)], [])
    next(engine)
    engine.close()
    with pytest.raises(StopIteration):
        engine.send(((1., 0.), 0.))


def test_record_false_keeps_outcome_but_drops_traces():
    args = (SeaMap((0, 0, 20, 20)), [(2, 2), (8, 2), (8, 8)], [])
    full, _ = drive(episode_steps(*args))
    light, _ = drive(episode_steps(*args, record=False))
    assert (light['status'], light['elapsed']) == (full['status'], full['elapsed'])
    assert light['frames'] == [] and light['diagnostics'] == []


def test_run_episode_prefers_packet_policies():
    class PacketPolicy:
        seen = []

        def act(self, packet):
            self.seen.append(packet['goal_index'])
            return toward(packet['position'], packet['velocity'], packet['goal'], [], .5, 1., packet['dt'])

    policy = PacketPolicy()
    assert run_episode(SeaMap((0, 0, 20, 20)), [(2, 2), (8, 2)], [], policy)['status'] == 'success'
    assert policy.seen and set(policy.seen) == {1}
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_episode_steps.py -v`
Expected: FAIL, `ImportError: cannot import name 'episode_steps'`.

- [ ] **Step 3: Implement the engine**

Replace `run_episode` in `src/shipnav/simulation.py` with the following. The validation block, marine set-up, motion, clearance, target-validity and run-out code are moved unchanged.

```python
def episode_steps(sea: SeaMap, route: list[Point], traffic: list[Traffic], policy=None,
                  dt: float = .25, limit: float = 100, radius: float = .5, speed: float = 1.0,
                  cancel: Callable[[], bool] = lambda: False, *, observer=None, filtered=False,
                  uncertainty=True, dynamics='holonomic', record=True, terminal_packet=False):
    """One episode as a generator paused at every nominal-action boundary.

    Yields a decision packet and expects `(nominal_velocity, controller_ms)` back via
    `send`; time spent by the caller between yield and send is excluded from
    decision latency. Returns `(result, terminal)`, where `terminal` is the final
    packet when `terminal_packet` is set (one extra observation) and None otherwise.
    `policy` is used only for its optional `set_context` hook and `solver_failed`
    flag. `record=False` (training) keeps outcomes but drops frames/diagnostics.
    """
    # --- unchanged: parameter, route and traffic validation; imports; observer default;
    # --- unchanged: marine Vessel initialisation and 'Unknown dynamics' check.
    position, velocity, time, index = route[0], (0.0, 0.0), 0.0, 1
    minimum, latencies, frames, travelled = float('inf'), [], [], 0.0
    tick, last = 0, None

    def record_frame():
        if record:
            frames.append({'t': time, 'position': list(position), 'velocity': list(velocity),
                           'goal_index': min(index, len(route)-1),
                           'traffic': [list(ship.at(time)[0]) for ship in traffic]})

    def packet(perceived, step):
        goal_index = min(index, len(route)-1)
        return {'position': position, 'velocity': velocity, 'goal': route[goal_index],
                'goal_index': goal_index, 'final': goal_index == len(route)-1,
                'neighbours': [(s['position'], s['velocity'], s['radius']) for s in perceived],
                'observed': perceived, 'radius': radius, 'speed': speed, 'dt': step, 't': time,
                'vessel': vessel, 'sea': sea, 'last': last}

    record_frame()
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
        perceived = observer.observe(traffic, time, tick)
        steps = horizon_steps(step, dynamics, speed)
        predictions = predict(perceived, step, steps=steps, uncertainty=uncertainty)
        if hasattr(policy, 'set_context'):
            policy.set_context(sea, vessel, predictions, steps=steps, final=index == len(route)-1)
        perception_ms = (perf_counter()-began)*1000
        nominal, controller_ms = yield packet(perceived, step)
        resumed = perf_counter()
        nominal = tuple(nominal)
        latencies.append(float(controller_ms))
        if len(nominal) != 2 or not all(isfinite(x) for x in nominal) or hypot(*nominal) > speed+1e-6:
            raise ValueError('Policy returned invalid velocity')
        motion = motion_from(vessel, speed) if vessel is not None else None
        # Unfiltered runs never evaluate feasibility: no_feasible_action is None
        # ("not evaluated"), not False ("a feasible action existed").
        decision = choose(sea, position, nominal, predictions, radius, speed, step, steps=steps, motion=motion,
                          goal=route[index], final=index == len(route)-1) if filtered else {
            'executed': nominal, 'override': False, 'no_feasible_action': None, 'path': [], 'predicted_clearance': None}
        velocity = tuple(decision['executed'])
        decision_ms = perception_ms + float(controller_ms) + (perf_counter()-resumed)*1000
        # Soft wall-clock indicator only (see original comment).
        entry = {'t': time, 'observed': perceived, 'predictions': predictions, 'nominal': nominal, **decision,
                 'decision_ms': decision_ms, 'deadline_miss': decision_ms > step*1000,
                 'solver_failed': bool(getattr(policy, 'solver_failed', False))}
        if record:
            diagnostics.append(entry)
        # --- unchanged from here to `travelled += ...`, with every `diagnostics[-1]`
        # --- replaced by `entry` (heading/speed, collision flags, target validity).
        last = {'clearance': clearance if isfinite(clearance) else None, 'dt': step,
                'collision': collision, 'land_collision': land_hit, 'ship_collision': ship_hit}
        travelled += dist(position, next_position)
        position, time = next_position, time+step
        if collision:
            status = 'collision'
        elif dist(position, route[index]) < radius:
            index += 1
            if index == len(route):
                status = 'success'
        tick += 1
        record_frame()
    result = {'status': status, 'elapsed': time, 'distance': travelled,
              'min_dynamic_clearance': minimum if isfinite(minimum) else None,
              'inference_ms': latencies, 'frames': frames, 'diagnostics': diagnostics}
    # --- unchanged marine terminal_speed / runout_min_clearance block.
    terminal = packet(observer.observe(traffic, time, tick), dt) if terminal_packet else None
    return result, terminal


def run_episode(sea: SeaMap, route: list[Point], traffic: list[Traffic], policy,
                dt: float = .25, limit: float = 100, radius: float = .5, speed: float = 1.0,
                cancel: Callable[[], bool] = lambda: False, **kwargs) -> dict:
    """GUI/CLI driver: time each policy call and feed it to the step engine."""
    engine = episode_steps(sea, route, traffic, policy, dt, limit, radius, speed, cancel, **kwargs)
    try:
        packet = next(engine)
        while True:
            started = perf_counter()
            if hasattr(policy, 'act'):
                action = policy.act(packet)
            else:
                action = policy(packet['position'], packet['velocity'], packet['goal'],
                                packet['neighbours'], radius, speed, packet['dt'])
            packet = engine.send((action, (perf_counter()-started)*1000))
    except StopIteration as ended:
        return ended.value[0]
    finally:
        engine.close()
```

The observer tick is now an explicit counter. It equals the old `len(frames)-1` when recording and stays correct when `record=False`.

- [ ] **Step 4: Run new, parity and full simulation tests**

Run: `python -m pytest tests/test_episode_steps.py tests/test_engine_parity.py tests/test_simulation.py tests/test_safety_pipeline.py tests/test_service.py -v`, then the full suite `python -m pytest -q`.
Expected: all PASS. A changed digest means physics changed: fix the refactor, never the fixture.

- [ ] **Step 5: Commit**

```bash
git add src/shipnav/simulation.py tests/test_episode_steps.py
git commit -m "refactor: expose the episode loop as a step engine driven by run_episode"
```

### Task 4: Disjoint procedural training corpus

**Files:**
- Create: `src/shipnav/training/__init__.py` (empty), `src/shipnav/training/scenarios.py`, `tools/generate_training_scenarios.py`, `tests/test_training_scenarios.py`
- Create (generated, committed): `scenarios/training/manifest.json`, `train.jsonl.gz`, `select.jsonl.gz`

**Interfaces:**
- Consumes: `episode.plan_route`, `scenarios.make_scenario/validate_scenario/scenario_hash`
- Produces: `generate(seed: int, split: str) -> dict`; `write_corpus(directory, splits: dict[str, list[dict]], failures: list[dict]) -> dict` (returns manifest); `load_split(directory, split: str) -> list[dict]` (hash-verified); `frozen_identities(root) -> tuple[set[str], set[int]]`; `TRAIN_SEEDS = range(100_000, 105_000)`; `SELECT_SEEDS = range(200_000, 200_200)`

- [ ] **Step 1: Write the failing tests**

File: `tests/test_training_scenarios.py`

```python
from pathlib import Path
import gzip
import json

import pytest

from shipnav.scenarios import scenario_hash, validate_scenario
from shipnav.training.scenarios import (SELECT_SEEDS, TRAIN_SEEDS, frozen_identities, generate,
                                        load_split, write_corpus)

ROOT = Path(__file__).resolve().parents[1]


def test_generation_is_deterministic_valid_and_labelled():
    a, b = generate(100_000, 'train'), generate(100_000, 'train')
    assert a == b and a['split'] == 'train' and a['seed'] == 100_000
    validate_scenario(a)
    assert 24 <= a['map']['bounds'][2] <= 48


def test_seed_ranges_are_disjoint_from_each_other_and_every_frozen_manifest():
    hashes, seeds = frozen_identities(ROOT)
    assert not set(TRAIN_SEEDS) & set(SELECT_SEEDS)
    assert not (set(TRAIN_SEEDS) | set(SELECT_SEEDS)) & seeds
    assert scenario_hash(generate(100_001, 'train')) not in hashes


def test_corpus_round_trip_verifies_hashes(tmp_path):
    scenes = [generate(100_002, 'train')]
    write_corpus(tmp_path, {'train': scenes, 'select': [generate(200_000, 'select')]}, [])
    assert load_split(tmp_path, 'train') == scenes
    raw = gzip.decompress((tmp_path/'train.jsonl.gz').read_bytes())
    (tmp_path/'train.jsonl.gz').write_bytes(gzip.compress(raw+b'\n', mtime=0))
    with pytest.raises(ValueError, match='hash'):
        load_split(tmp_path, 'train')


def test_committed_corpus_matches_its_manifest():
    manifest = json.loads((ROOT/'scenarios/training/manifest.json').read_text())
    assert manifest['splits']['train']['count'] >= 4_900
    assert len(load_split(ROOT/'scenarios/training', 'select')) == manifest['splits']['select']['count']
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_training_scenarios.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'shipnav.training'`.

- [ ] **Step 3: Implement the generator and loader**

File: `src/shipnav/training/scenarios.py`

```python
"""Procedural training and checkpoint-selection scenarios.

Seeds are disjoint from every frozen manifest; maps are random rectangular
islands, never harbour.json, Ubin or Southern Islands. 70% of scenarios use
corridor traffic timed to meet the ego on its reference route (crossing,
head-on, overtaking); the rest use uniform traffic, a fifth of it reactive.
"""
from hashlib import sha256
from math import dist
from pathlib import Path
from random import Random
import gzip
import json

from shipnav.episode import plan_route
from shipnav.maps import SeaMap
from shipnav.scenarios import make_scenario, scenario_hash, validate_scenario

TRAIN_SEEDS = range(100_000, 105_000)
SELECT_SEEDS = range(200_000, 200_200)
SIZES = (24., 36., 48.)
ENDPOINT_MARGIN, ISLAND_MARGIN = 2., 1.5


def _endpoints(rng, size):
    while True:
        a = (rng.uniform(ENDPOINT_MARGIN, size-ENDPOINT_MARGIN), rng.uniform(ENDPOINT_MARGIN, size-ENDPOINT_MARGIN))
        b = (rng.uniform(ENDPOINT_MARGIN, size-ENDPOINT_MARGIN), rng.uniform(ENDPOINT_MARGIN, size-ENDPOINT_MARGIN))
        if min(12., size/2) <= dist(a, b) <= 45.:
            return a, b


def _islands(rng, size, endpoints, count):
    land = []
    for _ in range(count):
        for _ in range(100):
            w, h = rng.uniform(1.5, size/5), rng.uniform(1.5, size/5)
            x, y = rng.uniform(1., size-1.-w), rng.uniform(1., size-1.-h)
            if all(not (x-ISLAND_MARGIN < px < x+w+ISLAND_MARGIN and y-ISLAND_MARGIN < py < y+h+ISLAND_MARGIN)
                   for px, py in endpoints):
                land.append((x, y, x+w, y+h))
                break
    return tuple(land)


def generate(seed, split):
    """Raises NoPath/PlanningLimit/InvalidRoute/ValueError for an unusable seed."""
    rng = Random(seed)
    size = rng.choice(SIZES)
    start, goal = _endpoints(rng, size)
    sea = SeaMap((0., 0., size, size), _islands(rng, size, (start, goal), rng.randint(0, 4)))
    corridor = rng.random() < .7
    route = plan_route(sea, start, goal) if corridor else None
    scenario = make_scenario(sea, start, goal, rng.randint(1, 5), seed, corridor=route)
    reactive = not corridor and rng.random() < .2
    scenario = {**scenario, 'split': split, 'family': f'generated_{scenario["family"]}',
                'traffic_mode': 'reactive' if reactive else 'scripted'}
    plan_route(sea, start, goal)  # every kept scenario must be reachable
    validate_scenario(scenario)
    return json.loads(json.dumps(scenario, allow_nan=False))  # JSON-normal form: lists, not tuples


def _encode(scenarios):
    lines = (json.dumps(s, sort_keys=True, separators=(',', ':'), allow_nan=False) for s in scenarios)
    return gzip.compress(('\n'.join(lines)+'\n').encode(), mtime=0)


def write_corpus(directory, splits, failures):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 1, 'generator': 'shipnav.training.scenarios', 'splits': {}, 'failures': failures}
    for name, scenarios in splits.items():
        data = _encode(scenarios)
        (directory/f'{name}.jsonl.gz').write_bytes(data)
        manifest['splits'][name] = {'file': f'{name}.jsonl.gz', 'sha256': sha256(data).hexdigest(),
                                    'count': len(scenarios)}
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    return manifest


def load_split(directory, split):
    directory = Path(directory)
    entry = json.loads((directory/'manifest.json').read_text())['splits'][split]
    data = (directory/entry['file']).read_bytes()
    if sha256(data).hexdigest() != entry['sha256']:
        raise ValueError(f'Corpus hash mismatch for split {split}')
    scenarios = [json.loads(line) for line in gzip.decompress(data).decode().splitlines() if line]
    if len(scenarios) != entry['count']:
        raise ValueError(f'Corpus count mismatch for split {split}')
    return scenarios


def frozen_identities(root):
    """Scenario hashes and seeds of every frozen evaluation/development manifest."""
    root = Path(root)
    hashes, seeds = set(), set(range(1000, 1010))  # historical train split, unused
    for path in (root/'scenarios/splits.json', root/'scenarios/baseline/splits.json'):
        for entries in json.loads(path.read_text())['splits'].values():
            hashes |= {e['scenario_hash'] for e in entries}
            seeds |= {e['seed'] for e in entries if 'seed' in e}
    return hashes, seeds
```

File: `tools/generate_training_scenarios.py`

```python
"""Write scenarios/training/ deterministically; failed seeds are recorded, never retried."""
from pathlib import Path

from shipnav.planning import InvalidRoute, NoPath, PlanningLimit
from shipnav.scenarios import scenario_hash
from shipnav.training.scenarios import SELECT_SEEDS, TRAIN_SEEDS, frozen_identities, generate, write_corpus

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    hashes, seeds = frozen_identities(ROOT)
    if (set(TRAIN_SEEDS) | set(SELECT_SEEDS)) & seeds:
        raise SystemExit('Generated seed range overlaps a frozen manifest')
    splits, failures = {'train': [], 'select': []}, []
    for split, seed_range in (('train', TRAIN_SEEDS), ('select', SELECT_SEEDS)):
        for seed in seed_range:
            try:
                scenario = generate(seed, split)
            except (NoPath, PlanningLimit, InvalidRoute, ValueError) as error:
                failures.append({'seed': seed, 'split': split, 'error': f'{type(error).__name__}: {error}'})
                continue
            if scenario_hash(scenario) in hashes:
                raise SystemExit(f'Seed {seed} reproduces a frozen scenario')
            splits[split].append(scenario)
    manifest = write_corpus(ROOT/'scenarios/training', splits, failures)
    print({k: v['count'] for k, v in manifest['splits'].items()}, 'failures', len(failures))
```

- [ ] **Step 4: Generate twice, check determinism, run tests**

Run: `python tools/generate_training_scenarios.py`, record `shasum -a 256 scenarios/training/*`, run it again and compare. Then run `python -m pytest tests/test_training_scenarios.py -v`.
Expected: identical hashes, at least 4,900 train scenarios, and the tests PASS. If more than 5% of seeds fail, inspect the failure reasons. Adjust the generator only (not individual seeds), then regenerate everything.

- [ ] **Step 5: Commit**

```bash
git add src/shipnav/training tools/generate_training_scenarios.py tests/test_training_scenarios.py scenarios/training
git commit -m "feat: generate a disjoint procedural training and selection corpus"
```

### Task 5: Bounded observation schema 1 and action transform

**Files:**
- Create: `src/shipnav/training/observation.py`, `tests/test_observation_encoding.py`

**Interfaces:**
- Consumes: the Task 3 packet (`position, velocity, goal, final, observed, radius, speed, vessel, sea`)
- Produces: `SCHEMA = 1`, `ACTION_SCHEMA = 1`, `SIZE = 120`, `encode(packet) -> np.ndarray[float32]` with every value in `[-1, 1]`, `ray_lengths(sea, p, radius) -> list[float]`, `action_to_velocity(action, speed) -> tuple[float, float]`

All features use fixed analytic scaling in model units, so no fitted normaliser has to be saved. The manifest records `observation_schema` instead. Future target truth and the domain cost are never inputs.

| Slice | Features (scale) |
|---|---|
| 0–7 | goal unit vector x/y; goal distance `min(d,20)/20`; ego velocity x/y `/speed`; `sin`, `cos` heading; final-waypoint flag |
| 8–103 | 12 nearest perceived targets × (rel x, rel y clipped `/20`; vx, vy `/2`; radius `/2`; age `min(·,5)/5`; margin `min(·,5)/5`; mask) |
| 104–119 | 16 map-clearance rays, `/20` |

- [ ] **Step 1: Write the failing tests**

File: `tests/test_observation_encoding.py`

```python
import numpy as np
import pytest

from shipnav.dynamics import Vessel
from shipnav.maps import SeaMap
from shipnav.training.observation import SIZE, action_to_velocity, encode, ray_lengths


def packet(**overrides):
    base = {'position': (5., 5.), 'velocity': (.5, 0.), 'goal': (100., 5.), 'final': True,
            'observed': [{'id': 0, 'position': (8., 5.), 'velocity': (-1., 0.), 'radius': .6,
                          'age': 0., 'margin': 0.}],
            'radius': .5, 'speed': 1., 'vessel': Vessel(5., 5., 0., .5), 'sea': SeaMap((0, 0, 24, 24))}
    return {**base, **overrides}


def test_encoding_is_bounded_and_sized():
    values = encode(packet())
    assert values.shape == (SIZE,) and values.dtype == np.float32
    assert np.all(values >= -1) and np.all(values <= 1)
    assert values[2] == 1.  # far goal clipped
    assert values[15] == 1. and values[23] == 0.  # first target mask on, second off


def test_targets_sorted_nearest_first_and_capped_at_twelve():
    far = [{'id': i, 'position': (5.+i+2, 5.), 'velocity': (0., 0.), 'radius': .6, 'age': 0., 'margin': 0.}
           for i in range(15)]
    values = encode(packet(observed=list(reversed(far))))
    assert values[8] == pytest.approx(2/20)
    assert all(values[8+8*k+7] == 1. for k in range(12))


def test_rays_measure_land_and_edges():
    sea = SeaMap((0, 0, 24, 24), ((10, 0, 12, 24),))
    rays = ray_lengths(sea, (5., 5.), .5)
    assert rays[0] == pytest.approx(4.5, abs=.05)   # east, to the land face minus radius
    assert rays[8] == pytest.approx(4.5, abs=.05)   # west, to the map edge minus radius


def test_holonomic_packets_are_rejected():
    with pytest.raises(ValueError, match='marine'):
        encode(packet(vessel=None))


def test_action_transform_clips_to_the_unit_disk_and_rejects_bad_input():
    assert action_to_velocity([1., 1.], 1.) == pytest.approx((2**-.5, 2**-.5))
    assert action_to_velocity([.3, .4], 2.) == pytest.approx((.6, .8))
    assert action_to_velocity([0., 0.], 1.) == (0., 0.)
    with pytest.raises(ValueError):
        action_to_velocity([np.nan, 0.], 1.)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_observation_encoding.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

File: `src/shipnav/training/observation.py`

```python
"""Observation schema 1: bounded, unit-scaled features shared by NavigationEnv and
the ModernLearned adapter so training and deployment encode identically."""
from math import cos, dist, hypot, pi, sin

import numpy as np

SCHEMA, ACTION_SCHEMA = 1, 1
RANGE, VELOCITY_BOUND, RADIUS_BOUND, AGE_BOUND = 20., 2., 2., 5.
MAX_TARGETS, TARGET_FEATURES, RAYS, RAY_ITERATIONS, EGO_FEATURES = 12, 8, 16, 10, 8
SIZE = EGO_FEATURES + MAX_TARGETS*TARGET_FEATURES + RAYS


def _clip(value, bound):
    return max(-1., min(1., value/bound))


def ray_lengths(sea, p, radius):
    """Clear distance along 16 bearings (monotone segment test, bisection)."""
    lengths = []
    for k in range(RAYS):
        c, s = cos(2*pi*k/RAYS), sin(2*pi*k/RAYS)
        if sea.clear(p, (p[0]+c*RANGE, p[1]+s*RANGE), radius):
            lengths.append(RANGE)
            continue
        lo, hi = 0., RANGE
        for _ in range(RAY_ITERATIONS):
            mid = (lo+hi)/2
            lo, hi = (mid, hi) if sea.clear(p, (p[0]+c*mid, p[1]+s*mid), radius) else (lo, mid)
        lengths.append(lo)
    return lengths


def encode(packet):
    vessel = packet['vessel']
    if vessel is None:
        raise ValueError('Observation schema 1 requires marine dynamics (vessel heading)')
    p, v, g, speed = packet['position'], packet['velocity'], packet['goal'], packet['speed']
    dx, dy = g[0]-p[0], g[1]-p[1]
    d = hypot(dx, dy)
    values = [dx/d if d else 0., dy/d if d else 0., min(d, RANGE)/RANGE,
              _clip(v[0], speed), _clip(v[1], speed), sin(vessel.heading), cos(vessel.heading),
              float(packet['final'])]
    targets = sorted(packet['observed'], key=lambda o: dist(o['position'], p))[:MAX_TARGETS]
    for o in targets:
        values += [_clip(o['position'][0]-p[0], RANGE), _clip(o['position'][1]-p[1], RANGE),
                   _clip(o['velocity'][0], VELOCITY_BOUND), _clip(o['velocity'][1], VELOCITY_BOUND),
                   min(o['radius'], RADIUS_BOUND)/RADIUS_BOUND, min(o['age'], AGE_BOUND)/AGE_BOUND,
                   min(o['margin'], AGE_BOUND)/AGE_BOUND, 1.]
    values += [0.]*(TARGET_FEATURES*(MAX_TARGETS-len(targets)))
    values += [r/RANGE for r in ray_lengths(packet['sea'], p, packet['radius'])]
    return np.asarray(values, dtype=np.float32)


def action_to_velocity(action, speed):
    """Normalised 2-D desired velocity -> model-unit command inside the speed disk.
    Zero requests braking through the marine tracker; it never zeroes velocity."""
    a = np.asarray(action, dtype=float)
    if a.shape != (2,) or not np.isfinite(a).all():
        raise ValueError('Action must be two finite numbers')
    a = np.clip(a, -1., 1.)
    norm = hypot(*a)
    a = a/norm if norm > 1 else a
    return float(speed*a[0]), float(speed*a[1])
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_observation_encoding.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/shipnav/training/observation.py tests/test_observation_encoding.py
git commit -m "feat: add bounded observation schema and shared action transform"
```

### Task 6: Gymnasium environment with route progress and truth-based domain cost

**Files:**
- Create: `src/shipnav/training/env.py`, `tests/test_training_env.py`

**Interfaces:**
- Consumes: Task 2 `plan_route/build_traffic/episode_limit`, Task 3 `episode_steps`, Task 5 `encode/action_to_velocity/SIZE`, `scale.map_options/units`
- Produces:
  - `EnvConfig` (frozen dataclass) with fields `observation: dict = {}`, `filtered: bool = True`, `penalty: float = 0.`, `domain_m: float = 1.`, `progress_weight = 1.`, `time_penalty = .01`, `success_reward = 10.`, `collision_penalty = 10.`, `limit: float | None = None`, plus `to_dict()`.
  - `NavigationEnv(scenarios, config=EnvConfig())` with `set_penalty(value)`.
  - `remaining_route(position, route, index) -> float`
  - `domain_cost(last, length_m, time_s, domain_m) -> float`
  - `info` keys: `cost`, `progress`, `status`; terminal steps add `episode_cost`.

Contract:
- `reset` picks a scenario with `np_random`, and builds route, traffic and limit exactly as the service does (`map_options` resolution/clearance, `episode_limit`). It uses marine dynamics, `record=False` and `terminal_packet=True`.
- Reward is route progress (remaining waypoint-path length, not straight-line distance to the final goal), minus the time penalty, ±10 on success/collision, minus `penalty × cost`.
- Cost is the physical seconds the truth swept clearance to other ships stays below `domain_m` metres (the ship domain). It uses the same rule as `metrics._exposure`, so the training cost and the evaluation metric `domain_time_s` agree. Land contact is not a cost: it ends the episode as a collision.
- Success and collision terminate the episode; timeout truncates it.
- Advancing to the next waypoint can add at most one ego-radius-sized progress bonus. This is accepted and documented.

- [ ] **Step 1: Write the failing tests**

File: `tests/test_training_env.py`

```python
from math import dist

from gymnasium.utils.env_checker import check_env as gym_check_env
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from shipnav.maps import SeaMap
from shipnav.scenarios import make_scenario
from shipnav.training.env import EnvConfig, NavigationEnv, remaining_route


def scene(land=(), start=(2., 2.), goal=(20., 2.), count=0, seed=1):
    return make_scenario(SeaMap((0, 0, 24, 24), land), start, goal, count, seed)


def run(env, action, steps=400):
    out = []
    for _ in range(steps):
        observation, reward, terminated, truncated, info = env.step(action)
        out.append((observation, reward, terminated, truncated, info))
        if terminated or truncated:
            break
    return out


def test_checkers_pass_and_reset_is_seeded():
    gym_check_env(NavigationEnv([scene(count=2)], EnvConfig(filtered=False)), skip_render_check=True)
    sb3_check_env(NavigationEnv([scene(count=2)], EnvConfig(filtered=False)))
    env = NavigationEnv([scene(count=2, seed=s) for s in (1, 2, 3)], EnvConfig(filtered=False))
    a, _ = env.reset(seed=7)
    b, _ = env.reset(seed=7)
    assert np.array_equal(a, b) and env.observation_space.contains(a)


def test_time_limit_truncates():
    env = NavigationEnv([scene()], EnvConfig(filtered=False, limit=.25))
    env.reset(seed=1)
    _, _, terminated, truncated, info = env.step([1., 0.])
    assert truncated and not terminated and info['status'] == 'timeout' and 'episode_cost' in info
    with pytest.raises(RuntimeError, match='reset'):
        env.step([1., 0.])


def test_land_collision_terminates_with_a_valid_final_observation():
    env = NavigationEnv([scene(land=((0, 5, 24, 6),))], EnvConfig(filtered=False))
    env.reset(seed=1)
    observation, reward, terminated, truncated, info = run(env, [0., 1.])[-1]
    assert terminated and not truncated and info['status'] == 'collision'
    assert reward < -9 and env.observation_space.contains(observation)


def test_domain_cost_is_logged_penalised_and_never_observed():
    # A ship 1.3 m abeam: truth clearance 1.3-.5-.6 = .2 m < 1 m domain, but no collision.
    alongside = {**scene(), 'traffic': [{'start': [2., 3.3], 'goal': [22., 3.3], 'speed': .3, 'radius': .6}]}
    plain = NavigationEnv([alongside], EnvConfig(filtered=False))
    priced = NavigationEnv([alongside], EnvConfig(filtered=False, penalty=5.))
    plain.reset(seed=1), priced.reset(seed=1)
    (o1, r1, *_, i1), (o2, r2, *_, i2) = plain.step([1., 0.]), priced.step([1., 0.])
    assert i1['cost'] == pytest.approx(.25) and np.array_equal(o1, o2)
    assert r1-r2 == pytest.approx(5*.25)
    priced.set_penalty(0.)
    with pytest.raises(ValueError):
        priced.set_penalty(-1.)


def test_progress_follows_the_route_not_the_straight_line():
    env = NavigationEnv([scene(land=((6, 3.5, 8, 24),), start=(4., 20.), goal=(12., 20.))],
                        EnvConfig(filtered=False))
    env.reset(seed=1)
    total, before = 0., dist(env.packet['position'], (12., 20.))
    for _ in range(12):
        p, g = env.packet['position'], env.packet['goal']
        d = dist(p, g)
        *_, info = env.step([(g[0]-p[0])/d, (g[1]-p[1])/d])
        total += info['progress']
    assert total > 0 and dist(env.packet['position'], (12., 20.)) > before


def test_remaining_route_sums_the_unvisited_legs():
    assert remaining_route((0., 1.), [(0., 0.), (0., 4.), (3., 4.)], 1) == pytest.approx(6.)


def test_inputs_are_validated():
    with pytest.raises(ValueError):
        NavigationEnv([])
    with pytest.raises(ValueError):
        EnvConfig(observation={'seed': 1})
    env = NavigationEnv([scene()], EnvConfig(filtered=False))
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step([np.inf, 0.])
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_training_env.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'shipnav.training.env'`.

- [ ] **Step 3: Implement**

File: `src/shipnav/training/env.py`

```python
"""Gymnasium view of the shared marine step engine (observation schema 1)."""
from dataclasses import asdict, dataclass, field
from math import dist, isfinite

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from shipnav.episode import build_traffic, episode_limit, plan_route
from shipnav.maps import SeaMap
from shipnav.observations import Observer
from shipnav.scale import map_options, units
from shipnav.scenarios import scenario_hash, validate_observation, validate_scenario
from shipnav.simulation import episode_steps
from shipnav.training.observation import SIZE, action_to_velocity, encode


@dataclass(frozen=True)
class EnvConfig:
    observation: dict = field(default_factory=dict)
    filtered: bool = True
    penalty: float = 0.
    domain_m: float = 1.
    progress_weight: float = 1.
    time_penalty: float = .01
    success_reward: float = 10.
    collision_penalty: float = 10.
    limit: float | None = None

    def __post_init__(self):
        validate_observation(self.observation)
        numbers = (self.penalty, self.domain_m, self.progress_weight, self.time_penalty,
                   self.success_reward, self.collision_penalty)
        if not all(isfinite(x) and x >= 0 for x in numbers) or self.domain_m <= 0:
            raise ValueError('Reward/cost settings must be finite and nonnegative; domain_m positive')
        if self.limit is not None and not (isfinite(self.limit) and self.limit > 0):
            raise ValueError('Limit override must be positive and finite')

    def to_dict(self):
        return asdict(self)


def remaining_route(position, route, index):
    index = min(index, len(route)-1)
    return dist(position, route[index]) + sum(dist(a, b) for a, b in zip(route[index:], route[index+1:]))


def domain_cost(last, length_m, time_s, domain_m):
    """Physical seconds of the last step spent inside the truth ship domain."""
    if last is None or last['clearance'] is None:
        return 0.
    return last['dt']*time_s if last['clearance']*length_m < domain_m else 0.


class NavigationEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, scenarios, config=EnvConfig()):
        if not scenarios:
            raise ValueError('Empty training split')
        for scenario in scenarios:
            validate_scenario(scenario)
        self.scenarios, self.config, self.penalty = tuple(scenarios), config, config.penalty
        self.action_space = spaces.Box(-1., 1., (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1., 1., (SIZE,), dtype=np.float32)
        self.engine, self.done = None, True

    def set_penalty(self, value):
        value = float(value)
        if not isfinite(value) or value < 0:
            raise ValueError('Penalty multiplier must be finite and nonnegative')
        self.penalty = value

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        scenario = self.scenarios[int(self.np_random.integers(len(self.scenarios)))]
        sea = SeaMap.from_dict(scenario['map'])
        self.route = plan_route(sea, scenario['start'], scenario['goal'], **map_options(scenario['map']))
        self.length_m, self.time_s = units(scenario['map'])
        observer = Observer(seed=int(self.np_random.integers(2**31)), **self.config.observation)
        limit = self.config.limit or episode_limit(scenario['map'], self.route)
        self.engine = episode_steps(sea, self.route, build_traffic(scenario, sea), None, limit=limit,
                                    observer=observer, filtered=self.config.filtered, dynamics='marine',
                                    record=False, terminal_packet=True)
        try:
            self.packet = next(self.engine)
        except StopIteration as error:
            raise ValueError('Training split contains an initially terminal scenario') from error
        self.done, self.episode_cost = False, 0.
        self.remaining = remaining_route(self.packet['position'], self.route, self.packet['goal_index'])
        return encode(self.packet), {'scenario_hash': scenario_hash(scenario)}

    def step(self, action):
        if self.done:
            raise RuntimeError('Call reset() after an episode ends')
        command = action_to_velocity(action, self.packet['speed'])
        try:
            self.packet, status = self.engine.send((command, 0.)), 'running'
        except StopIteration as ended:
            result, self.packet = ended.value
            status = result['status']
        c = self.config
        cost = domain_cost(self.packet['last'], self.length_m, self.time_s, c.domain_m)
        remaining = 0. if status == 'success' else remaining_route(
            self.packet['position'], self.route, self.packet['goal_index'])
        progress = self.remaining-remaining
        reward = (c.progress_weight*progress - c.time_penalty + c.success_reward*(status == 'success')
                  - c.collision_penalty*(status == 'collision') - self.penalty*cost)
        self.remaining, self.episode_cost = remaining, self.episode_cost+cost
        terminated, truncated = status in ('success', 'collision'), status in ('timeout', 'cancelled')
        self.done = terminated or truncated
        info = {'cost': cost, 'progress': progress, 'status': status}
        if self.done:
            info['episode_cost'] = self.episode_cost
        return encode(self.packet), float(reward), terminated, truncated, info

    def close(self):
        if self.engine is not None:
            self.engine.close()
            self.engine = None
```

- [ ] **Step 4: Run tests and a throughput probe**

Run: `python -m pytest tests/test_training_env.py -v`
Expected: PASS. Then record single-env steps/s for filtered on and off on `scenarios/training` with `python tools/training_pilot.py --env-only` (added in Task 10). Until then, a quick `timeit` over 1,000 random-action steps is enough. Write the number in the commit message body.

- [ ] **Step 5: Commit**

```bash
git add src/shipnav/training/env.py tests/test_training_env.py
git commit -m "feat: add marine Gymnasium environment with route progress and domain cost"
```

### Task 7: Atomic, hash-verified full-state checkpoints

**Files:**
- Create: `src/shipnav/training/checkpoints.py`, `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: an SB3 model
- Produces: `FILES = ('policy.zip', 'state.pt', 'rng.json')`; `save_checkpoint(model, root, name, metadata: dict) -> Path`; `verify_checkpoint(directory) -> dict` (the manifest); `restore_rng(directory) -> dict`; `latest_checkpoint(root) -> Path | None`; `update_penalty(current, mean_episode_cost, budget, rate=.05) -> float`

A checkpoint is written into a hidden staging folder, hashed, given its manifest, and then renamed into place in one atomic step. A folder without a valid manifest never counts as a checkpoint. Resuming is **restart-equivalent**, not bitwise: model, optimizer and RNG state are restored, and the environments are reseeded at an episode boundary from `(seed, num_timesteps)`.

- [ ] **Step 1: Write the failing tests**

File: `tests/test_checkpoint.py`

```python
import json
import random

import numpy as np
import pytest
from stable_baselines3 import PPO
import torch

from shipnav.maps import SeaMap
from shipnav.scenarios import make_scenario
from shipnav.training.checkpoints import (latest_checkpoint, restore_rng, save_checkpoint,
                                          update_penalty, verify_checkpoint)
from shipnav.training.env import EnvConfig, NavigationEnv


def tiny_model():
    env = NavigationEnv([make_scenario(SeaMap((0, 0, 24, 24)), (2, 2), (20, 2), 1, 1)], EnvConfig(filtered=False))
    return PPO('MlpPolicy', env, seed=3, n_steps=64, batch_size=32, device='cpu')


def test_round_trip_preserves_deterministic_actions_and_rng(tmp_path):
    model = tiny_model()
    observations = np.random.default_rng(0).uniform(-1, 1, (5, 120)).astype(np.float32)
    expected = model.predict(observations, deterministic=True)[0]
    path = save_checkpoint(model, tmp_path, 'chunk-0000', {'penalty': 0.})
    next_torch, next_python = torch.rand(3), random.random()
    manifest = verify_checkpoint(path)
    loaded = PPO.load(path/'policy.zip', device='cpu')
    restore_rng(path)  # after PPO.load: _setup_model reseeds global RNGs
    assert np.array_equal(loaded.predict(observations, deterministic=True)[0], expected)
    assert torch.equal(torch.rand(3), next_torch) and random.random() == next_python
    assert set(manifest['files']) == {'policy.zip', 'state.pt', 'rng.json'}


def test_existing_names_are_never_overwritten_and_tampering_is_detected(tmp_path):
    model = tiny_model()
    path = save_checkpoint(model, tmp_path, 'chunk-0000', {})
    with pytest.raises(FileExistsError):
        save_checkpoint(model, tmp_path, 'chunk-0000', {})
    (path/'state.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash'):
        verify_checkpoint(path)


def test_latest_checkpoint_ignores_incomplete_folders(tmp_path):
    model = tiny_model()
    save_checkpoint(model, tmp_path, 'chunk-0000', {'chunk': 0})
    model.num_timesteps = 64
    save_checkpoint(model, tmp_path, 'chunk-0001', {'chunk': 1})
    (tmp_path/'chunk-0002').mkdir()  # no manifest: crashed upload or copy
    assert latest_checkpoint(tmp_path).name == 'chunk-0001'
    assert latest_checkpoint(tmp_path/'missing') is None


def test_penalty_rises_on_violation_and_stays_nonnegative():
    assert update_penalty(1., 2., 1.) > 1.
    assert update_penalty(0., 0., 1.) == 0.
    with pytest.raises(ValueError):
        update_penalty(0., float('nan'), 1.)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_checkpoint.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

File: `src/shipnav/training/checkpoints.py`

```python
"""Checkpoints: policy/optimizer (SB3 zip), RNG state and a manifest written last.

RNG state avoids pickle: Python/NumPy generators go to rng.json and torch
generator tensors to state.pt, loaded with `weights_only=True`. SB3's
policy.zip still deserialises objects on load, so only load hash-verified
checkpoints that this project produced; never load a policy.zip from elsewhere.
"""
from hashlib import sha256
from math import isfinite
from pathlib import Path
import json
import os
import random
import tempfile

import numpy as np
import torch

FILES = ('policy.zip', 'state.pt', 'rng.json')


def _rng_json():
    version, internal, gauss = random.getstate()
    name, key, pos, has_gauss, cached = np.random.get_state()
    return {'python': [version, list(internal), gauss],
            'numpy': [name, key.tolist(), int(pos), int(has_gauss), float(cached)]}


def save_checkpoint(model, root, name, metadata):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    final = root/name
    if final.exists():
        raise FileExistsError(final)
    staging = Path(tempfile.mkdtemp(prefix=f'.{name}-', dir=root))
    model.save(staging/'policy.zip')
    torch.save({'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
               staging/'state.pt')
    (staging/'rng.json').write_text(json.dumps(_rng_json()))
    files = {n: sha256((staging/n).read_bytes()).hexdigest() for n in FILES}
    manifest = {**metadata, 'num_timesteps': model.num_timesteps, 'files': files}
    (staging/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+'\n')
    os.rename(staging, final)
    return final


def verify_checkpoint(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    for name in FILES:
        if sha256((directory/name).read_bytes()).hexdigest() != manifest['files'][name]:
            raise ValueError(f'Checkpoint hash mismatch: {name}')
    return manifest


def restore_rng(directory):
    """Call after PPO.load, whose model setup reseeds the global RNGs."""
    directory = Path(directory)
    state = torch.load(directory/'state.pt', weights_only=True)
    rng = json.loads((directory/'rng.json').read_text())
    version, internal, gauss = rng['python']
    random.setstate((version, tuple(internal), gauss))
    name, key, pos, has_gauss, cached = rng['numpy']
    np.random.set_state((name, np.asarray(key, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state


def latest_checkpoint(root):
    root = Path(root)
    if not root.is_dir():
        return None
    complete = []
    for path in root.iterdir():
        if path.is_dir() and not path.name.startswith('.') and (path/'manifest.json').is_file():
            complete.append((verify_checkpoint(path)['num_timesteps'], path.name, path))
    return max(complete)[2] if complete else None


def update_penalty(current, mean_episode_cost, budget, rate=.05):
    if not all(isfinite(x) for x in (current, mean_episode_cost, budget, rate)):
        raise ValueError('Penalty update inputs must be finite')
    return max(0., current+rate*(mean_episode_cost-budget))
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_checkpoint.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/shipnav/training/checkpoints.py tests/test_checkpoint.py
git commit -m "feat: add atomic hash-verified training checkpoints with RNG state"
```

### Task 8: Chunked trainer with vector environments, Lagrangian update, SIGTERM and deadline

**Files:**
- Create: `src/shipnav/training/train.py`, `tests/test_train.py`

**Interfaces:**
- Consumes: Tasks 4, 6, 7, and `shipnav.provenance.provenance`
- Produces: `main(argv: list[str] | None) -> dict` (the final status). It writes `<output>/chunk-NNNN/` checkpoints, `<output>/curve.jsonl` and `<output>/status.json`. The CLI is `python -m shipnav.training.train --scenarios DIR --output DIR --seed N --variant {ppo,ppo_lagrangian} [--budget B] --chunks N [--chunk-steps 20480] [--envs 8] [--device cpu|cuda|mps] --max-seconds S [--resume] [--no-filter] [--observation JSON]`.

Behaviour:
- Each chunk trains `--chunk-steps` more steps (`learn(..., reset_num_timesteps=False)` adds to `num_timesteps`), then writes a checkpoint.
- `ppo_lagrangian` updates the multiplier from the **completed-episode domain costs of that chunk**. The checkpoint records both `penalty_used` and `penalty_next`.
- `ppo` keeps `penalty=0`.
- A SIGTERM or the deadline stops training at the next environment step. The trainer then saves a checkpoint marked `complete_chunk: false` and exits with status `stopped` or `deadline`.
- `--resume` continues from the latest verified checkpoint and requires the same `--envs` and configuration. It restores RNG state after `PPO.load` and reseeds the environments from `seed*1_000_003 + num_timesteps`.
- The Lagrangian variant is a pragmatic ablation, not a constrained-MDP solution or a certified safety method.

- [ ] **Step 1: Write the failing tests**

File: `tests/test_train.py`

```python
import json

import pytest

from shipnav.maps import SeaMap
from shipnav.scenarios import make_scenario
from shipnav.training import train
from shipnav.training.checkpoints import verify_checkpoint
from shipnav.training.scenarios import write_corpus


@pytest.fixture
def corpus(tmp_path):
    sea = SeaMap((0, 0, 24, 24), ((9, 9, 12, 12),))
    scenes = [{**make_scenario(sea, (2, 2), (20, 20), 1, s), 'split': 'train'} for s in (1, 2)]
    write_corpus(tmp_path/'corpus', {'train': scenes, 'select': scenes[:1]}, [])
    return tmp_path/'corpus'


def args(corpus, output, *extra):
    return ['--scenarios', str(corpus), '--output', str(output), '--seed', '0', '--chunk-steps', '128',
            '--envs', '1', '--max-seconds', '600', '--no-filter', *extra]


def test_chunks_checkpoint_and_resume(corpus, tmp_path):
    out = tmp_path/'run'
    assert train.main(args(corpus, out, '--variant', 'ppo', '--chunks', '2'))['state'] == 'complete'
    assert verify_checkpoint(out/'chunk-0001')['num_timesteps'] == 256
    status = train.main(args(corpus, out, '--variant', 'ppo', '--chunks', '3', '--resume'))
    manifest = verify_checkpoint(out/'chunk-0002')
    assert status['state'] == 'complete' and manifest['num_timesteps'] == 384
    assert manifest['resume_mode'] == 'restart-equivalent' and manifest['observation_schema'] == 1
    assert len((out/'curve.jsonl').read_text().splitlines()) == 3


def test_lagrangian_records_penalty_and_requires_budget(corpus, tmp_path):
    with pytest.raises(SystemExit):
        train.main(args(corpus, tmp_path/'a', '--variant', 'ppo_lagrangian', '--chunks', '1'))
    train.main(args(corpus, tmp_path/'b', '--variant', 'ppo_lagrangian', '--budget', '0', '--chunks', '1'))
    manifest = verify_checkpoint(tmp_path/'b'/'chunk-0000')
    assert manifest['penalty_used'] == 0. and manifest['penalty_next'] >= 0.


def test_stop_request_saves_a_partial_checkpoint(corpus, tmp_path):
    train.STOP.set()
    try:
        status = train.main(args(corpus, tmp_path/'c', '--variant', 'ppo', '--chunks', '2'))
    finally:
        train.STOP.clear()
    assert status['state'] == 'stopped'
    assert json.loads((tmp_path/'c'/'status.json').read_text())['state'] == 'stopped'


def test_resume_rejects_a_changed_configuration(corpus, tmp_path):
    out = tmp_path/'d'
    train.main(args(corpus, out, '--variant', 'ppo', '--chunks', '1'))
    with pytest.raises(SystemExit):
        train.main(args(corpus, out, '--variant', 'ppo_lagrangian', '--budget', '1', '--chunks', '2', '--resume'))
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_train.py -v`
Expected: FAIL, `ImportError: cannot import name 'train'`.

- [ ] **Step 3: Implement**

File: `src/shipnav/training/train.py`

```python
"""Bounded-chunk PPO / Lagrangian-penalty training with full-state checkpoints."""
from pathlib import Path
from statistics import mean
from threading import Event
import argparse
import json
import signal
import time

import gymnasium
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from shipnav.provenance import provenance
from shipnav.training.checkpoints import (latest_checkpoint, restore_rng, save_checkpoint,
                                          update_penalty, verify_checkpoint)
from shipnav.training.env import EnvConfig, NavigationEnv
from shipnav.training.observation import ACTION_SCHEMA, SCHEMA
from shipnav.training.scenarios import load_split

STOP = Event()
ROLLOUT, BATCH, CHECKPOINT_GRACE_S = 2048, 64, 120.
ROOT = Path(__file__).resolve().parents[4]  # repository root, as service._ROOT
IDENTITY = ('variant', 'seed', 'envs', 'chunk_steps', 'env_config', 'scenario_manifest_sha256')


class EpisodeStats(BaseCallback):
    def __init__(self, deadline):
        super().__init__()
        self.deadline, self.costs, self.statuses, self.interrupted = deadline, [], [], False

    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode_cost' in info:
                self.costs.append(info['episode_cost'])
                self.statuses.append(info['status'])
        self.interrupted = STOP.is_set() or time.monotonic() >= self.deadline
        return not self.interrupted


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenarios', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--variant', choices=('ppo', 'ppo_lagrangian'), required=True)
    p.add_argument('--budget', type=float)
    p.add_argument('--chunks', type=int, required=True)
    p.add_argument('--chunk-steps', type=int, default=20480)
    p.add_argument('--envs', type=int, default=8)
    p.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    p.add_argument('--max-seconds', type=float, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--no-filter', action='store_true')
    p.add_argument('--observation', type=json.loads, default={})
    a = p.parse_args(argv)
    if a.variant == 'ppo_lagrangian' and a.budget is None:
        p.error('--budget (physical domain seconds per episode) is required for ppo_lagrangian')
    if min(a.chunks, a.chunk_steps, a.envs) < 1 or a.max_seconds <= 0:
        p.error('chunks, chunk-steps, envs and max-seconds must be positive')
    rollout = min(ROLLOUT, a.chunk_steps)
    if rollout % a.envs or a.chunk_steps % rollout:
        p.error('chunk-steps must be a multiple of the rollout, and the rollout divisible by envs')
    a.n_steps = rollout//a.envs
    return a


def identity(a, config):
    manifest = json.loads((a.scenarios/'manifest.json').read_text())
    return {'variant': a.variant, 'seed': a.seed, 'envs': a.envs, 'chunk_steps': a.chunk_steps,
            'env_config': config.to_dict(), 'scenario_manifest_sha256': manifest['splits']['train']['sha256']}


def write_status(output, state, **fields):
    status = {'state': state, **fields}
    (output/'status.json').write_text(json.dumps(status, indent=2, allow_nan=False)+'\n')
    return status


def main(argv=None):
    a = parse(argv)
    previous = signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    deadline = time.monotonic()+a.max_seconds-CHECKPOINT_GRACE_S
    a.output.mkdir(parents=True, exist_ok=True)
    last = latest_checkpoint(a.output) if a.resume else None
    if not a.resume and latest_checkpoint(a.output):
        raise SystemExit('Output already has checkpoints; use --resume or a new directory')
    penalty = verify_checkpoint(last)['penalty_next'] if last else 0.
    config = EnvConfig(observation=a.observation, filtered=not a.no_filter, penalty=penalty)
    ident = identity(a, EnvConfig(observation=a.observation, filtered=not a.no_filter))
    if last and {k: verify_checkpoint(last)[k] for k in IDENTITY} != ident:
        raise SystemExit('Resume configuration differs from the checkpoint')
    scenarios = load_split(a.scenarios, 'train')
    env = make_vec_env(lambda: NavigationEnv(scenarios, config), n_envs=a.envs, seed=a.seed,
                       vec_env_cls=SubprocVecEnv if a.envs > 1 else DummyVecEnv)
    try:
        if last:
            model = PPO.load(last/'policy.zip', env=env, device=a.device)
            restore_rng(last)
            env.seed(a.seed*1_000_003+model.num_timesteps)
            first = verify_checkpoint(last)['chunk']+1
        else:
            model = PPO('MlpPolicy', env, seed=a.seed, device=a.device, n_steps=a.n_steps, batch_size=BATCH)
            first = 0
        state = 'complete'
        for chunk in range(first, a.chunks):
            began, stats = time.monotonic(), EpisodeStats(deadline)
            if STOP.is_set() or time.monotonic() >= deadline:
                stats.interrupted = True  # stop before learning: save a zero-step partial checkpoint
            else:
                env.env_method('set_penalty', penalty)
                model.learn(a.chunk_steps, reset_num_timesteps=False, callback=stats)
            done = not stats.interrupted
            used = penalty
            if a.variant == 'ppo_lagrangian' and stats.costs:
                penalty = update_penalty(penalty, mean(stats.costs), a.budget)
            save_checkpoint(model, a.output, f'chunk-{chunk:04d}', {
                **ident, 'chunk': chunk, 'complete_chunk': done, 'penalty_used': used, 'penalty_next': penalty,
                'budget': a.budget, 'observation_schema': SCHEMA, 'action_schema': ACTION_SCHEMA,
                'dynamics': 'marine', 'device': a.device, 'resume_mode': 'restart-equivalent',
                'versions': {'torch': torch.__version__, 'stable_baselines3': stable_baselines3.__version__,
                             'gymnasium': gymnasium.__version__},
                'provenance': provenance(ident, ROOT)})
            with (a.output/'curve.jsonl').open('a') as stream:
                stream.write(json.dumps({'chunk': chunk, 'num_timesteps': model.num_timesteps,
                    'episodes': len(stats.costs), 'mean_episode_cost': mean(stats.costs) if stats.costs else None,
                    'success_rate': stats.statuses.count('success')/len(stats.statuses) if stats.statuses else None,
                    'collision_rate': stats.statuses.count('collision')/len(stats.statuses) if stats.statuses else None,
                    'penalty_used': used, 'penalty_next': penalty, 'wall_s': time.monotonic()-began})+'\n')
            if not done:
                state = 'stopped' if STOP.is_set() else 'deadline'
                break
        return write_status(a.output, state, num_timesteps=model.num_timesteps)
    finally:
        env.close()
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(0 if main()['state'] == 'complete' else 3)
```

The stop test sets `STOP` before the first chunk. In that case the loop saves a zero-step partial checkpoint and writes status `stopped`, which is the expected safe-boundary behaviour.

- [ ] **Step 4: Run tests, then a local 20,480-step smoke**

Run: `python -m pytest tests/test_train.py -v`, then:

```bash
python -m shipnav.training.train --scenarios scenarios/training --output results/train-smoke --seed 0 --variant ppo --chunks 1 --envs 4 --max-seconds 3600
```

Expected: the tests PASS; the smoke writes `chunk-0000` with `num_timesteps` 20480 and `status.json` `complete`. Send SIGTERM during a second smoke run (`kill -TERM <pid>`) and confirm a partial checkpoint plus status `stopped`.

- [ ] **Step 5: Commit**

```bash
git add src/shipnav/training/train.py tests/test_train.py
git commit -m "feat: add chunked PPO and Lagrangian trainer with safe stop and resume"
```

### Task 9: `ppo` policy in the service, CLI and GUI

**Files:**
- Create: `src/shipnav/training/adapter.py`, `tests/test_modern_learned.py`
- Modify: `src/shipnav/service.py` (signature, policy selection, CLI choices), `src/shipnav/gui.py:86-91,285-290`, `tests/test_gui.py`

**Interfaces:**
- Consumes: Task 3 `run_episode` `act` dispatch, Task 5 `encode/action_to_velocity`, Task 7 `verify_checkpoint`
- Produces: `ModernLearned(directory, device='cpu')` with `.act(packet)` and `.hashes`. `execute(..., learned_dir=None)`: `policy_name='ppo'` loads `learned_dir or model_dir`, requires `dynamics='marine'`, and records `model_hashes = manifest['files']`.

- [ ] **Step 1: Write the failing tests**

File: `tests/test_modern_learned.py`

```python
import json

import numpy as np
import pytest
from stable_baselines3 import PPO

from shipnav.maps import SeaMap
from shipnav.scenarios import make_scenario
from shipnav.service import execute
from shipnav.training.adapter import ModernLearned
from shipnav.training.checkpoints import save_checkpoint
from shipnav.training.env import EnvConfig, NavigationEnv
from shipnav.training.observation import ACTION_SCHEMA, SCHEMA, action_to_velocity


@pytest.fixture
def checkpoint(tmp_path):
    scene = make_scenario(SeaMap((0, 0, 24, 24)), (2, 2), (20, 2), 2, 4)
    model = PPO('MlpPolicy', NavigationEnv([scene], EnvConfig(filtered=False)), seed=1, n_steps=64, device='cpu')
    meta = {'observation_schema': SCHEMA, 'action_schema': ACTION_SCHEMA, 'dynamics': 'marine'}
    return scene, model, save_checkpoint(model, tmp_path, 'chunk-0000', meta)


def test_service_and_gym_issue_identical_actions(checkpoint):
    scene, model, path = checkpoint
    run = execute(scene['map'], scene['start'], scene['goal'], policy_name='ppo', learned_dir=str(path),
                  scenario=scene, dynamics='marine', filtered=False)
    env = NavigationEnv([scene], EnvConfig(filtered=False))
    observation, _ = env.reset(seed=0)
    for diagnostic in run['diagnostics'][:20]:
        expected = action_to_velocity(model.predict(observation, deterministic=True)[0], 1.)
        assert diagnostic['nominal'] == pytest.approx(expected, abs=1e-6)
        observation, *_ = env.step(model.predict(observation, deterministic=True)[0])
    assert run['model_hashes'] == json.loads((path/'manifest.json').read_text())['files']


def test_incompatible_or_tampered_checkpoints_are_rejected(checkpoint):
    _, _, path = checkpoint
    manifest = json.loads((path/'manifest.json').read_text())
    (path/'manifest.json').write_text(json.dumps({**manifest, 'observation_schema': 0}))
    with pytest.raises(ValueError, match='schema'):
        ModernLearned(path)
    (path/'manifest.json').write_text(json.dumps(manifest))
    (path/'policy.zip').write_bytes(b'not a model')
    with pytest.raises(ValueError, match='hash'):
        ModernLearned(path)


def test_ppo_requires_marine_dynamics(checkpoint):
    scene, _, path = checkpoint
    with pytest.raises(ValueError, match='marine'):
        execute(scene['map'], scene['start'], scene['goal'], policy_name='ppo', learned_dir=str(path),
                scenario=scene, dynamics='holonomic')
```

In `tests/test_gui.py`, add:

```python
def test_policy_menu_offers_trained_ppo(qtbot):
    window = Window(SeaMap((0, 0, 24, 24)))
    qtbot.addWidget(window)
    assert 'ppo' in [window.policy.itemText(i) for i in range(window.policy.count())]
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_modern_learned.py tests/test_gui.py::test_policy_menu_offers_trained_ppo -v`
Expected: FAIL (`ModuleNotFoundError`; `'ppo'` not in the menu).

- [ ] **Step 3: Implement**

File: `src/shipnav/training/adapter.py`

```python
"""Deterministic trained-PPO controller for the service and GUI.

It consumes the engine's decision packet and encodes it with the same function
as NavigationEnv, so a saved policy acts identically in training and deployment.
"""
from pathlib import Path

from shipnav.training.checkpoints import verify_checkpoint
from shipnav.training.observation import ACTION_SCHEMA, SCHEMA, SIZE, action_to_velocity, encode


class ModernLearned:
    def __init__(self, directory, device='cpu'):
        directory = Path(directory)
        manifest = verify_checkpoint(directory)
        found = (manifest.get('observation_schema'), manifest.get('action_schema'), manifest.get('dynamics'))
        if found != (SCHEMA, ACTION_SCHEMA, 'marine'):
            raise ValueError(f'Incompatible checkpoint manifest {found}: needs observation schema {SCHEMA}, '
                             f'action schema {ACTION_SCHEMA}, marine dynamics')
        from stable_baselines3 import PPO
        self.model = PPO.load(directory/'policy.zip', device=device)
        if self.model.observation_space.shape != (SIZE,) or self.model.action_space.shape != (2,):
            raise ValueError('Checkpoint spaces do not match observation/action schema 1')
        self.manifest, self.hashes = manifest, dict(manifest['files'])

    def act(self, packet):
        action, _ = self.model.predict(encode(packet), deterministic=True)
        return action_to_velocity(action, packet['speed'])
```

In `src/shipnav/service.py`:
- Add `learned_dir=None` to the `execute` keyword-only parameters.
- Insert before the `else` of the policy chain:

```python
    elif policy_name == 'ppo':
        if dynamics != 'marine':
            raise ValueError('Trained PPO policies require marine dynamics')
        from shipnav.training.adapter import ModernLearned
        policy = ModernLearned(Path(learned_dir or model_dir))
        hashes = policy.hashes
```

- Change the final error to `'Policy must be sarl, orca, mpc, direct or ppo'`, and add `'ppo'` to the CLI `--policy` choices.

In `src/shipnav/gui.py`, change line 91 to `self.policy.addItems(['sarl', 'orca', 'mpc', 'direct', 'ppo'])`. Change the dialog title in `load_model` to `'Model folder (SARL: policy.config + rl_model.pth; PPO checkpoint: manifest.json + policy.zip)'`. The service's `ValueError` already reaches the status bar through the existing worker error path, so a holonomic or invalid PPO run fails visibly.

- [ ] **Step 4: Run tests, including parity and GUI**

Run: `python -m pytest tests/test_modern_learned.py tests/test_gui.py tests/test_service.py tests/test_engine_parity.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/shipnav/training/adapter.py src/shipnav/service.py src/shipnav/gui.py tests/test_modern_learned.py tests/test_gui.py
git commit -m "feat: run trained PPO checkpoints through the service and GUI"
```

### Task 10: Throughput pilot and preregistered training configuration

**Files:**
- Create: `tools/training_pilot.py`, `tools/freeze_training_config.py`
- Create (generated, committed **before any full training run**): `evidence/retraining/training_config.json`

**Interfaces:**
- Consumes: Tasks 4, 6, 8, and `service.execute` with `marine_mpc` settings
- Produces: a pilot JSON (`results/pilot-<host>.json`) and `training_config.json` with keys `seeds, variants, chunks, chunk_steps, envs, filtered, observation, budget_domain_s, selection, corpus_sha256, source_commit`

- [ ] **Step 1: Write the pilot tool**

File: `tools/training_pilot.py`

```python
"""Measure environment and learner throughput for EC2 sizing. Not a performance claim."""
from pathlib import Path
import argparse
import json
import os
import platform
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from shipnav.training.env import EnvConfig, NavigationEnv
from shipnav.training.scenarios import load_split


def env_rate(scenarios, filtered, steps=1000):
    env, rng = NavigationEnv(scenarios, EnvConfig(filtered=filtered)), np.random.default_rng(0)
    env.reset(seed=0)
    began = time.perf_counter()
    for _ in range(steps):
        *_, terminated, truncated, _ = env.step(rng.uniform(-1, 1, 2))
        if terminated or truncated:
            env.reset()
    return steps/(time.perf_counter()-began)


def learn_rate(scenarios, envs, device, steps):
    env = make_vec_env(lambda: NavigationEnv(scenarios, EnvConfig()), n_envs=envs, seed=0,
                       vec_env_cls=SubprocVecEnv if envs > 1 else DummyVecEnv)
    try:
        model = PPO('MlpPolicy', env, seed=0, device=device, n_steps=2048//envs, batch_size=64)
        began = time.perf_counter()
        model.learn(steps)
        return steps/(time.perf_counter()-began)
    finally:
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scenarios', type=Path, default=Path('scenarios/training'))
    p.add_argument('--devices', nargs='+', default=['cpu'])
    p.add_argument('--envs', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--steps', type=int, default=8192)
    p.add_argument('--env-only', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    scenarios = load_split(a.scenarios, 'train')[:500]
    result = {'host': platform.node(), 'platform': platform.platform(), 'cpus': os.cpu_count(),
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'cuda_available': torch.cuda.is_available(),
              'env_steps_per_s': {f: env_rate(scenarios, f) for f in (False, True)}, 'learn_steps_per_s': {}}
    if not a.env_only:
        for device in a.devices:
            for envs in a.envs:
                result['learn_steps_per_s'][f'{device}/envs={envs}'] = learn_rate(scenarios, envs, device, a.steps)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
```

- [ ] **Step 2: Run the local pilot**

Run: `python tools/training_pilot.py --devices cpu mps --output results/pilot-local.json`
Expected: JSON with filtered and unfiltered env rates plus learn rates. Use it to estimate the hours for the planned total steps (below). Treat MPS numbers as unvalidated until a separate parity check passes, per the global constraints.

- [ ] **Step 3: Write the configuration freezer**

File: `tools/freeze_training_config.py`

```python
"""Preregister training settings and the domain-cost budget before full training.

The budget comes from marine_mpc on the generated *selection* split, never the
held-out corpus. Commit the output before launching any full training run.
"""
from pathlib import Path
from statistics import mean
import argparse
import json
import subprocess

from shipnav.metrics import metrics
from shipnav.scale import map_options
from shipnav.service import execute
from shipnav.training.scenarios import load_split

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--chunks', type=int, required=True)
    p.add_argument('--envs', type=int, required=True)
    p.add_argument('--seeds', type=int, default=5)
    a = p.parse_args()
    if a.seeds < 3:
        raise SystemExit('At least three independent training seeds are required')
    if subprocess.check_output(['git', 'status', '--porcelain', 'src', 'scenarios/training'], cwd=ROOT).strip():
        raise SystemExit('Commit source and corpus before freezing the training configuration')
    exposures = []
    for scenario in load_split(ROOT/'scenarios/training', 'select'):
        run = execute(scenario['map'], scenario['start'], scenario['goal'], policy_name='mpc', scenario=scenario,
                      dynamics='marine', filtered=True, **map_options(scenario['map']))
        exposures.append(metrics(run)['domain_time_s'] or 0.)
    corpus = json.loads((ROOT/'scenarios/training/manifest.json').read_text())
    config = {
        'schema': 1, 'seeds': list(range(a.seeds)), 'variants': ['ppo', 'ppo_lagrangian'],
        'chunks': a.chunks, 'chunk_steps': 20480, 'envs': a.envs, 'filtered': True, 'observation': {},
        'budget_domain_s': round(mean(exposures), 3), 'budget_source': 'marine_mpc mean domain_time_s on select split',
        'selection': {'split': 'select', 'every_chunks': 5, 'include_last': True, 'collision_max': .05,
                      'rule': 'max success among collision<=max; else min collision then max success; '
                              'ties: lower mean domain_time_s, then earlier chunk'},
        'corpus_sha256': {k: v['sha256'] for k, v in corpus['splits'].items()},
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    out = ROOT/'evidence/retraining/training_config.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2, allow_nan=False)+'\n')
    print(json.dumps(config, indent=2))
```

- [ ] **Step 4: Choose the step budget, freeze and commit**

Set `SHIPNAV_CHUNKS` and `SHIPNAV_ENVS` so that the total environment steps (`chunks × 20,480 × 5 seeds × 2 variants`) fit the approved spending limit at the measured rate, using `cloud/budget.py` from Task 13. Then run:

```bash
python tools/freeze_training_config.py --chunks "$SHIPNAV_CHUNKS" --envs "$SHIPNAV_ENVS"
git add tools/training_pilot.py tools/freeze_training_config.py evidence/retraining/training_config.json
git commit -m "chore: preregister retraining configuration and domain-cost budget"
```

After this commit, the config changes only through a new, explained commit made **before** any held-out run. It never changes because of results.

### Task 11: Checkpoint selection on the generated selection split

**Files:**
- Create: `tools/select_checkpoints.py`, `tests/test_select_checkpoints.py`

**Interfaces:**
- Consumes: Task 7 checkpoints, Task 9 `ppo` service path, the `selection` block of `training_config.json`
- Produces: `choose(rows: list[dict], collision_max: float) -> dict` (pure), and `<run>/selection.json` holding `{chunk, path, files, success_rate, collision_rate, mean_domain_s, candidates}`

- [ ] **Step 1: Write the failing test for the pure rule**

File: `tests/test_select_checkpoints.py`

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from select_checkpoints import choose  # noqa: E402


def row(chunk, success, collision, domain):
    return {'chunk': chunk, 'success_rate': success, 'collision_rate': collision, 'mean_domain_s': domain}


def test_prefers_success_among_safe_candidates_then_lower_domain_then_earlier():
    rows = [row(4, .9, .10, 1.), row(9, .8, .04, 2.), row(14, .8, .04, 1.), row(19, .8, .04, 1.)]
    assert choose(rows, .05)['chunk'] == 14


def test_falls_back_to_lowest_collision_when_none_are_safe():
    assert choose([row(4, .9, .2, 0.), row(9, .5, .1, 0.)], .05)['chunk'] == 9
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_select_checkpoints.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'select_checkpoints'`.

- [ ] **Step 3: Implement**

File: `tools/select_checkpoints.py`

```python
"""Pick one checkpoint per training run using only the generated selection split."""
from pathlib import Path
from statistics import mean
import argparse
import json

from shipnav.metrics import metrics
from shipnav.scale import map_options
from shipnav.service import execute
from shipnav.training.checkpoints import verify_checkpoint
from shipnav.training.scenarios import load_split

ROOT = Path(__file__).resolve().parents[1]


def choose(rows, collision_max):
    safe = [r for r in rows if r['collision_rate'] <= collision_max]
    if safe:
        return min(safe, key=lambda r: (-r['success_rate'], r['mean_domain_s'], r['chunk']))
    return min(rows, key=lambda r: (r['collision_rate'], -r['success_rate'], r['mean_domain_s'], r['chunk']))


def evaluate(path, scenarios, filtered):
    outcomes = []
    for scenario in scenarios:
        run = execute(scenario['map'], scenario['start'], scenario['goal'], policy_name='ppo',
                      learned_dir=str(path), scenario=scenario, dynamics='marine', filtered=filtered,
                      **map_options(scenario['map']))
        m = metrics(run)
        outcomes.append((run['status'], m['domain_time_s'] or 0.))
    n = len(outcomes)
    return {'success_rate': sum(s == 'success' for s, _ in outcomes)/n,
            'collision_rate': sum(s == 'collision' for s, _ in outcomes)/n,
            'mean_domain_s': mean(d for _, d in outcomes)}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    a = p.parse_args()
    config = json.loads((ROOT/'evidence/retraining/training_config.json').read_text())
    rule = config['selection']
    checkpoints = sorted(d for d in a.run.iterdir() if (d/'manifest.json').is_file())
    manifests = {d: verify_checkpoint(d) for d in checkpoints}
    complete = [d for d in checkpoints if manifests[d]['complete_chunk']]
    picked = [d for d in complete if manifests[d]['chunk'] % rule['every_chunks'] == rule['every_chunks']-1]
    if rule['include_last'] and complete and complete[-1] not in picked:
        picked.append(complete[-1])
    scenarios = load_split(ROOT/'scenarios/training', rule['split'])
    rows = [{'chunk': manifests[d]['chunk'], 'path': str(d), 'files': manifests[d]['files'],
             **evaluate(d, scenarios, config['filtered'])} for d in picked]
    best = choose(rows, rule['collision_max'])
    (a.run/'selection.json').write_text(json.dumps({**best, 'candidates': rows}, indent=2)+'\n')
    print(json.dumps(best, indent=2))
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_select_checkpoints.py -v`, then `python tools/select_checkpoints.py results/train-smoke` on the Task 8 smoke run.
Expected: PASS; the smoke run writes `selection.json`.

- [ ] **Step 5: Commit**

```bash
git add tools/select_checkpoints.py tests/test_select_checkpoints.py
git commit -m "feat: select training checkpoints on the generated selection split"
```

### Task 12: Generalised runner and retraining preregistration

**Files:**
- Modify: `tools/baseline_run.py` (functions `run_unit`, `summary`, `verify_frozen_identities`, `bind_output`, `validate_rows`, `execute_matrix`), `tests/test_baseline_guards.py`
- Create: `tools/freeze_retraining_protocol.py`, `tools/retrain_eval.py`, `tests/test_retrain_eval.py`
- Create (generated, committed before held-out execution): `evidence/retraining/protocol.json`

**Interfaces:**
- Consumes: Task 11 `selection.json` per run, the baseline manifest `scenarios/baseline/splits.json` (same 110 test scenarios), `metrics.hierarchical_interval`
- Produces:
  - Every listed `baseline_run` function gains a keyword `variants=VARIANTS`, and `summary` gains `pairs=PAIRS`. Defaults keep baseline behaviour.
  - `expected_model_hashes(data, name, variants) -> dict`
  - `seed_matched(fixed: dict, seeds) -> dict` replicates a fixed comparator under each training-seed key.
  - `retrain_eval.py` takes the same `heldout|audit|summarize` modes and mandatory `--protocol-commit/--protocol-sha256`.

Fix found in review: `validate_rows` currently compares **every** executed arm's `model_hashes` with the SARL checkpoint, but ORCA, MPC and direct arms record `{}`, so a protocol containing them can never pass audit. The expected hashes must be per variant.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_baseline_guards.py`:

```python
def test_expected_model_hashes_are_per_variant():
    sys.path.insert(0, str(ROOT/'tools'))
    from baseline_run import expected_model_hashes
    data = {'identities': {'model': {'rl_model.pth': 'a'}, 'learned': {'ppo_s0': {'policy.zip': 'b'}}}}
    variants = {'marine_sarl': {'policy_name': 'sarl'}, 'marine_mpc': {'policy_name': 'mpc'},
                'ppo_s0': {'policy_name': 'ppo'}}
    assert expected_model_hashes(data, 'marine_sarl', variants) == {'rl_model.pth': 'a'}
    assert expected_model_hashes(data, 'marine_mpc', variants) == {}
    assert expected_model_hashes(data, 'ppo_s0', variants) == {'policy.zip': 'b'}
```

File: `tests/test_retrain_eval.py`

```python
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from retrain_eval import learned_variants, seed_matched  # noqa: E402


def test_seed_matched_replicates_a_fixed_comparator():
    fixed = {'s1': 1., 's2': 0.}
    assert seed_matched(fixed, [0, 1]) == {0: fixed, 1: fixed}


def test_learned_variants_cover_every_seed_variant_and_filter_state():
    selections = {('ppo', 0): 'a', ('ppo', 1): 'b', ('ppo_lagrangian', 0): 'c', ('ppo_lagrangian', 1): 'd'}
    variants = learned_variants(selections)
    assert variants['ppo_s0_filtered'] == {'policy_name': 'ppo', 'learned_dir': 'a',
                                           'dynamics': 'marine', 'filtered': True}
    assert len(variants) == 8 and not variants['ppo_lagrangian_s1_unfiltered']['filtered']
    with pytest.raises(ValueError):
        learned_variants({('ppo', 0): 'a'})
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_baseline_guards.py tests/test_retrain_eval.py -v`
Expected: FAIL (`ImportError` for `expected_model_hashes` and `retrain_eval`).

- [ ] **Step 3: Generalise `baseline_run.py`**

Make these changes:

```python
def expected_model_hashes(data, name, variants=VARIANTS):
    policy = variants[name].get('policy_name')
    if policy == 'sarl':
        return data['identities']['model']
    if policy == 'ppo':
        return data['identities']['learned'][name]
    return {}
```

- Thread `variants=VARIANTS` through `run_unit` (it uses `variants[name]`), `verify_frozen_identities` (`if data['variants'] != variants`), `bind_output` (`'variants': variants`), `validate_rows` (`expected` grid and `variant_config` check), `execute_matrix` and `summary` (`for name in variants`, plus `pairs=PAIRS` in its comparison loop).
- In `validate_rows`, replace `run.get('model_hashes') != data['identities']['model']` with `run.get('model_hashes') != expected_model_hashes(data, pair[1], variants)`.
- In `verify_frozen_identities`, additionally verify each `identities.learned[name]` file hash against the variant's `learned_dir`.

Run `python -m pytest tests/test_baseline_guards.py tests/test_baseline_evidence.py -v` and confirm all pass with the defaults.

- [ ] **Step 4: Implement the protocol freezer and runner**

File: `tools/retrain_eval.py`

```python
"""Held-out evaluation of selected PPO checkpoints against re-run matched comparators.

Never run heldout before committing evidence/retraining/protocol.json; its commit
and SHA256 are verified exactly as for the pre-retraining baseline.
"""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import subprocess

from baseline_run import (ROOT, bind_output, dump, execute_matrix, summary, validate_rows,
                          verify_frozen_identities)
from shipnav.benchmark import VARIANTS
from shipnav.metrics import hierarchical_interval

COMPARATORS = {name: VARIANTS[name] for name in ('marine_mpc', 'marine_mpc_unfiltered', 'marine_sarl')}
LEARNED = ('ppo', 'ppo_lagrangian')


def learned_variants(selections):
    """selections: {(variant, seed): checkpoint directory} -> runner variant table."""
    seeds = {seed for _, seed in selections}
    if {(v, s) for v in LEARNED for s in seeds} != set(selections):
        raise ValueError('Every learned variant needs a selected checkpoint for every seed')
    return {f'{v}_s{s}_{"filtered" if f else "unfiltered"}':
            {'policy_name': 'ppo', 'learned_dir': selections[v, s], 'dynamics': 'marine', 'filtered': f}
            for v in LEARNED for s in sorted(seeds) for f in (True, False)}


def seed_matched(fixed, seeds):
    """A fixed comparator has no training-seed variance: replicate it under each seed key."""
    return {seed: fixed for seed in seeds}


def outcome(row, field):
    if field == 'success':
        return float(row['status'] == 'success')
    return float(bool(row['ship_collision'] or row['land_collision']))


def per_scenario(rows, variant, field):
    return {r['scenario_hash']: outcome(r, field) for r in rows if r['variant'] == variant}


def hierarchical(rows, data):
    seeds, out = data['training']['seeds'], {}
    for a, b, label in data['comparisons']:
        for field in ('success', 'any_collision'):
            def table(name):
                if name in COMPARATORS:
                    return seed_matched(per_scenario(rows, name, field), seeds)
                return {s: per_scenario(rows, name.format(s=s), field) for s in seeds}
            out[f'{label}/{field}'] = hierarchical_interval(table(a), table(b), seed=9127)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['heldout', 'audit', 'summarize'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='../CrowdNav-20250813-DIP/crowd_nav/data/output_trained')
    p.add_argument('--protocol', type=Path, default=ROOT/'evidence/retraining/protocol.json')
    p.add_argument('--protocol-commit', required=True)
    p.add_argument('--protocol-sha256', required=True)
    a = p.parse_args()
    raw = a.protocol.read_bytes()
    if sha256(raw).hexdigest() != a.protocol_sha256:
        raise ValueError('Protocol hash mismatch')
    path = a.protocol.resolve().relative_to(ROOT.parent).as_posix()
    if raw != subprocess.check_output(['git', 'show', f'{a.protocol_commit}:{path}'], cwd=ROOT):
        raise ValueError('Protocol is not the committed preregistration')
    data = json.loads(raw)
    variants = data['variants']
    verify_frozen_identities(data, a.model, variants=variants)
    entries = data['scenarios']['test']
    protocol = {'sha256': a.protocol_sha256, 'commit': a.protocol_commit}
    pairs = [tuple(x) for x in data['comparisons'] if '{s}' not in x[0] and '{s}' not in x[1]]
    if a.mode == 'heldout':
        execute_matrix(entries, ROOT/'scenarios/baseline', a.model, a.output, protocol, data, variants=variants)
    bind_output(a.output, protocol, entries, create=False, variants=variants)
    rows = [json.loads(line) for line in (a.output/'rows.jsonl').read_text().splitlines()]
    validate_rows(rows, entries, a.output, protocol, data, complete=True, variants=variants)
    if a.mode != 'audit':
        dump(a.output/'summary.json', {**summary(rows, variants=variants, pairs=pairs),
                                       'hierarchical': hierarchical(rows, data), 'protocol': protocol})


if __name__ == '__main__':
    main()
```

`execute_matrix` already writes the shared per-scenario limit. Copy `shared_limits_model` from the baseline protocol into the new protocol unchanged, so episode budgets are identical to the baseline's.

File: `tools/freeze_retraining_protocol.py`. Build and write `evidence/retraining/protocol.json`:
- `schema: 1`, a `name`, `frozen_utc`, and `source_commit` (HEAD, which must be clean).
- `source_sha256` over `pyproject.toml`, every `src/**/*.py`, and `tools/{baseline_run,baseline_evidence,retrain_eval}.py`.
- `manifest`/`manifest_sha256` = `scenarios/baseline/splits.json`, and `original_manifest`/`original_manifest_sha256` copied from the baseline protocol (both are checked by `verify_frozen_identities`).
- `scenarios.test` = that manifest's test entries.
- `shared_limits_model` copied from `evidence/baseline/benchmark_protocol.json`.
- `variants = {**COMPARATORS, **learned_variants(selections read from each results/train/<variant>/seed<k>/selection.json)}`.
- `identities.lock_sha256` (from `uv.lock`), `identities.model` (SARL hashes, re-checked against the files), `identities.learned` (`{arm: selection files}`).
- `training` = `training_config.json` contents plus `seeds`.
- `comparisons`:
  - `["ppo_s{s}_filtered", "marine_mpc", "learned_vs_mpc"]`
  - `["ppo_lagrangian_s{s}_filtered", "ppo_s{s}_filtered", "constraint_penalty"]`
  - `["ppo_s{s}_filtered", "ppo_s{s}_unfiltered", "PPO_filter"]`
  - `["ppo_s{s}_filtered", "marine_sarl", "new_vs_pretrained"]`
  - `["marine_mpc", "marine_mpc_unfiltered", "MPC_filter"]`
- `targets` = `{success_rate_minimum: .8, collision_rate_maximum: .05, constraint_cost_budget: <budget_domain_s>}`.
- `inference` = the baseline's method, plus: `"learned arms: hierarchical_interval over training seeds then scenarios; fixed comparators replicated per seed key; pairs of independently trained variants are matched by seed index (arbitrary pairing, exploratory)"`.
- `multiplicity: "exploratory, unadjusted"`.
- `limitations`: "Southern Islands transfer is zero-shot (no real-map training data); the 99 canonical-family cases were inspected during development; this corpus previously evaluated fixed policies."

Before writing, the freezer refuses to run if any selection file is missing, any checkpoint fails `verify_checkpoint`, or `git status --porcelain` is not clean.

- [ ] **Step 5: Run tests, freeze and commit the protocol**

Run: `python -m pytest tests/test_baseline_guards.py tests/test_retrain_eval.py tests/test_baseline_evidence.py -v` (expected PASS). After Task 14 delivers verified checkpoints and Task 11 selection has run on every run, freeze:

```bash
git add tools/baseline_run.py tools/retrain_eval.py tools/freeze_retraining_protocol.py tests/test_baseline_guards.py tests/test_retrain_eval.py
git commit -m "feat: generalise the frozen runner for preregistered retraining evaluation"
python tools/freeze_retraining_protocol.py
git add evidence/retraining/protocol.json
git commit -m "chore: preregister held-out retraining evaluation"
```

No held-out run happens before this second commit exists.

### Task 13: Cloud launch tooling

**Files:**
- Create: `cloud/prepare_launch.py`, `cloud/ssm_commands.py`, `cloud/budget.py`, `cloud/run-training.sh`, `tests/test_cloud.py`

**Interfaces:**
- Consumes: Task 8 CLI and exit codes (0 complete, 3 stopped/deadline)
- Produces:
  - `request(ami, instance_type, subnet, group, profile, root, run_id, hours, volume_gib) -> dict` (the `run-instances` JSON)
  - `commands(kind, settings) -> dict` (the SSM `--parameters` JSON, with `executionTimeout`)
  - `estimate(hours, hourly, volume_gib, ebs_gib_month, s3_gib, s3_gib_month, extra) -> dict`
  - `run-training.sh`, which publishes `policy.zip`, `state.pt` and `rng.json` before `manifest.json` for every checkpoint and always writes `status.json` to S3

AWS-RunShellScript's default `executionTimeout` is 3600 s, with a maximum of 172800 s ([AWS re:Post](https://repost.aws/knowledge-center/systems-manager-run-command)). Training therefore runs as a detached `systemd-run` unit. SSM only performs setup and checks, and S3 `status.json` is the source of truth.

- [ ] **Step 1: Write the failing tests**

File: `tests/test_cloud.py`

```python
from pathlib import Path
import json
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'cloud'))
from budget import estimate  # noqa: E402
from prepare_launch import request  # noqa: E402
from ssm_commands import commands  # noqa: E402
from shlex import quote  # noqa: E402

SETTINGS = {'bucket': 'example-bucket', 'prefix': 'shipnav/run-1', 'seed': 0, 'variant': 'ppo',
            'chunks': 40, 'envs': 8, 'device': 'cpu', 'seconds': 36000, 'run_id': 'run-1', 'budget': None,
            'uv_version': '0.0.0'}


def test_launch_request_is_private_encrypted_and_bounded():
    r = request('ami-0', 'c7i.4xlarge', 'subnet-0', 'sg-0', 'shipnav-profile', '/dev/xvda', 'run-1', 10., 60)
    assert (r['MinCount'], r['MaxCount']) == (1, 1) and 'KeyName' not in r
    assert r['MetadataOptions']['HttpTokens'] == 'required'
    ebs = r['BlockDeviceMappings'][0]['Ebs']
    assert ebs['Encrypted'] and ebs['DeleteOnTermination'] and ebs['VolumeSize'] == 60
    assert r['InstanceInitiatedShutdownBehavior'] == 'stop' and r['ClientToken'] == 'run-1'
    with pytest.raises(ValueError):
        request('ami-0', 't', 's', 'g', 'p', '/dev/xvda', 'run-1', 0., 60)


def test_ssm_documents_set_timeouts_and_quote_values():
    setup = commands('setup', SETTINGS)
    assert setup['executionTimeout'] == ['3600']
    start = commands('start', {**SETTINGS, 'prefix': 'a b; rm -rf /'})
    script = '\n'.join(start['commands'])
    assert 'systemd-run' in script and 'shutdown -h +' in script
    assert quote('s3://example-bucket/a b; rm -rf /') in script and '\nrm -rf' not in script
    with pytest.raises(ValueError):
        commands('start', {**SETTINGS, 'seconds': 0})
    with pytest.raises(ValueError):
        commands('start', {**SETTINGS, 'variant': 'ppo_lagrangian'})  # missing budget


def test_budget_includes_storage_and_extras():
    e = estimate(hours=10, hourly=1., volume_gib=60, ebs_gib_month=.08, s3_gib=5, s3_gib_month=.025, extra=2.)
    assert e['compute'] == 10. and e['total'] > 12.


def test_runner_script_parses_and_requires_settings(tmp_path):
    subprocess.run(['bash', '-n', str(ROOT/'cloud/run-training.sh')], check=True)
    result = subprocess.run(['bash', str(ROOT/'cloud/run-training.sh')], capture_output=True, text=True,
                            env={'PATH': '/usr/bin:/bin'})
    assert result.returncode != 0 and 'required' in result.stderr
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_cloud.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

File: `cloud/prepare_launch.py`

```python
"""Write cloud/local/launch.json from discovered identifiers. Launches nothing."""
from pathlib import Path
import argparse
import json


def request(ami, instance_type, subnet, group, profile, root, run_id, hours, volume_gib):
    if not hours > 0 or not volume_gib > 0:
        raise ValueError('Positive wall time and volume size required')
    tags = [{'Key': 'Project', 'Value': 'shipnav'}, {'Key': 'RunId', 'Value': run_id},
            {'Key': 'MaxHours', 'Value': str(hours)}]
    return {'ImageId': ami, 'InstanceType': instance_type, 'MinCount': 1, 'MaxCount': 1,
            'SubnetId': subnet, 'SecurityGroupIds': [group], 'IamInstanceProfile': {'Name': profile},
            'MetadataOptions': {'HttpTokens': 'required', 'HttpEndpoint': 'enabled', 'HttpPutResponseHopLimit': 1},
            'BlockDeviceMappings': [{'DeviceName': root, 'Ebs': {'VolumeSize': int(volume_gib), 'VolumeType': 'gp3',
                                                                 'Encrypted': True, 'DeleteOnTermination': True}}],
            'InstanceInitiatedShutdownBehavior': 'stop', 'ClientToken': run_id,
            'TagSpecifications': [{'ResourceType': r, 'Tags': tags} for r in ('instance', 'volume')]}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('ami', 'instance-type', 'subnet', 'group', 'profile', 'root', 'run-id'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--hours', type=float, required=True)
    p.add_argument('--volume-gib', type=int, required=True)
    a = p.parse_args()
    Path('cloud/local').mkdir(parents=True, exist_ok=True)
    Path('cloud/local/launch.json').write_text(json.dumps(request(
        a.ami, a.instance_type, a.subnet, a.group, a.profile, a.root, a.run_id, a.hours, a.volume_gib), indent=2))
```

File: `cloud/ssm_commands.py`

```python
"""SSM AWS-RunShellScript parameter documents. Values are shell-quoted; nothing is concatenated raw."""
from pathlib import Path
from shlex import quote
import argparse
import json

WORK = '/opt/shipnav'
UV_SYNC = 'uv sync --locked --extra train --extra maps'


def _check(s):
    if not s['seconds'] > 0 or s['chunks'] < 1 or s['envs'] < 1:
        raise ValueError('Positive seconds, chunks and envs required')
    if s['variant'] == 'ppo_lagrangian' and s.get('budget') is None:
        raise ValueError('ppo_lagrangian needs the preregistered budget')


def commands(kind, s):
    _check(s)
    uri = quote(f"s3://{s['bucket']}/{s['prefix']}")
    if kind == 'setup':
        body = ['set -euo pipefail', f'mkdir -p {WORK} && cd {WORK}',
                f'aws s3 cp {uri}/source.tar.gz source.tar.gz', f'aws s3 cp {uri}/source.sha256 source.sha256',
                'sha256sum -c source.sha256', 'tar -xzf source.tar.gz', 'ldd --version | head -1',
                f"command -v uv || curl -LsSf {quote('https://astral.sh/uv/'+s['uv_version']+'/install.sh')} | sh",
                f'cd rebuild && $HOME/.local/bin/uv python install 3.14 && $HOME/.local/bin/{UV_SYNC}']
        timeout = 3600
    elif kind == 'smoke':
        body = ['set -euo pipefail', f'cd {WORK}/rebuild', 'nvidia-smi || true',
                '.venv/bin/python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"',
                'MPLBACKEND=Agg .venv/bin/python -m pytest -p no:pytestqt -q tests/test_training_env.py '
                'tests/test_checkpoint.py tests/test_train.py tests/test_modern_learned.py']
        timeout = 3600
    elif kind == 'start':
        minutes = int(s['seconds']//60) + 30  # training deadline + upload grace
        env = {'SHIPNAV_PYTHON': f'{WORK}/rebuild/.venv/bin/python', 'SHIPNAV_SCENARIOS': f'{WORK}/rebuild/scenarios/training',
               'SHIPNAV_OUTPUT': f'{WORK}/out', 'SHIPNAV_S3_URI': f"s3://{s['bucket']}/{s['prefix']}",
               'SHIPNAV_SEED': s['seed'], 'SHIPNAV_VARIANT': s['variant'], 'SHIPNAV_CHUNKS': s['chunks'],
               'SHIPNAV_ENVS': s['envs'], 'SHIPNAV_DEVICE': s['device'], 'SHIPNAV_SECONDS': int(s['seconds']),
               'SHIPNAV_BUDGET': '' if s.get('budget') is None else s['budget']}
        setenv = ' '.join(f'--setenv={k}={quote(str(v))}' for k, v in env.items())
        body = ['set -euo pipefail', f'sudo shutdown -h +{minutes}',
                f"sudo systemd-run --unit={quote('shipnav-'+s['run_id'])} --property=RuntimeMaxSec={minutes*60} "
                f'--working-directory={WORK}/rebuild {setenv} /bin/bash {WORK}/rebuild/cloud/run-training.sh']
        timeout = 600
    else:
        raise ValueError('kind must be setup, smoke or start')
    return {'commands': body, 'executionTimeout': [str(timeout)]}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('kind', choices=('setup', 'smoke', 'start'))
    p.add_argument('--settings', type=Path, required=True)
    a = p.parse_args()
    out = Path(f'cloud/local/{a.kind}-commands.json')
    out.write_text(json.dumps(commands(a.kind, json.loads(a.settings.read_text())), indent=2))
    print(out)
```

File: `cloud/budget.py`

```python
"""Upper-bound cost estimate from a saved Pricing API quote. A budget alert is not a cap."""
import argparse
import json


def estimate(hours, hourly, volume_gib, ebs_gib_month, s3_gib, s3_gib_month, extra):
    compute = hours*hourly
    ebs = volume_gib*ebs_gib_month*max(hours, 24)/730  # include stopped-disk days before teardown
    s3 = s3_gib*s3_gib_month
    return {'compute': compute, 'ebs': ebs, 's3': s3, 'extra': extra, 'total': compute+ebs+s3+extra}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('hours', 'hourly', 'volume-gib', 'ebs-gib-month', 's3-gib', 's3-gib-month', 'extra'):
        p.add_argument('--'+key, type=float, required=True)
    a = p.parse_args()
    print(json.dumps(estimate(a.hours, a.hourly, a.volume_gib, a.ebs_gib_month, a.s3_gib, a.s3_gib_month, a.extra), indent=2))
```

File: `cloud/run-training.sh`

```bash
#!/usr/bin/env bash
# One process per chunk (resume between chunks) so every chunk is a publish boundary.
set -euo pipefail
: "${SHIPNAV_PYTHON:?SHIPNAV_PYTHON required}"
: "${SHIPNAV_SCENARIOS:?SHIPNAV_SCENARIOS required}"
: "${SHIPNAV_OUTPUT:?SHIPNAV_OUTPUT required}"
: "${SHIPNAV_S3_URI:?SHIPNAV_S3_URI required}"
: "${SHIPNAV_SEED:?SHIPNAV_SEED required}"
: "${SHIPNAV_VARIANT:?SHIPNAV_VARIANT required}"
: "${SHIPNAV_CHUNKS:?SHIPNAV_CHUNKS required}"
: "${SHIPNAV_ENVS:?SHIPNAV_ENVS required}"
: "${SHIPNAV_DEVICE:?SHIPNAV_DEVICE required}"
: "${SHIPNAV_SECONDS:?SHIPNAV_SECONDS required}"
export MPLBACKEND=Agg
mkdir -p "$SHIPNAV_OUTPUT"
state=failed
publish() {  # payload first, manifest last: a manifest in S3 means a complete checkpoint
    local dir=$1 name; name=$(basename "$dir")
    aws s3 cp "$dir/policy.zip" "$SHIPNAV_S3_URI/checkpoints/$name/policy.zip" --only-show-errors
    aws s3 cp "$dir/state.pt" "$SHIPNAV_S3_URI/checkpoints/$name/state.pt" --only-show-errors
    aws s3 cp "$dir/rng.json" "$SHIPNAV_S3_URI/checkpoints/$name/rng.json" --only-show-errors
    aws s3 cp "$dir/manifest.json" "$SHIPNAV_S3_URI/checkpoints/$name/manifest.json" --only-show-errors
}
finish() {
    local code=$?
    printf '{"state":"%s","exit_code":%d}\n' "$state" "$code" > "$SHIPNAV_OUTPUT/run-status.json"
    aws s3 cp "$SHIPNAV_OUTPUT/curve.jsonl" "$SHIPNAV_S3_URI/curve.jsonl" --only-show-errors || true
    aws s3 cp "$SHIPNAV_OUTPUT/run-status.json" "$SHIPNAV_S3_URI/status.json" --only-show-errors
}
trap finish EXIT
start=$SECONDS
budget=()
if [[ -n "${SHIPNAV_BUDGET:-}" ]]; then budget=(--budget "$SHIPNAV_BUDGET"); fi
for ((target=1; target<=SHIPNAV_CHUNKS; target++)); do
    remaining=$((SHIPNAV_SECONDS-(SECONDS-start)))
    if ((remaining<=300)); then state=deadline; exit 0; fi
    code=0  # --resume is a no-op without checkpoints and continues after a stop/start
    timeout --signal=TERM --kill-after=120s "$remaining" "$SHIPNAV_PYTHON" -m shipnav.training.train \
        --scenarios "$SHIPNAV_SCENARIOS" --output "$SHIPNAV_OUTPUT" --seed "$SHIPNAV_SEED" \
        --variant "$SHIPNAV_VARIANT" --chunks "$target" --envs "$SHIPNAV_ENVS" --device "$SHIPNAV_DEVICE" \
        --max-seconds "$remaining" --resume ${budget[@]+"${budget[@]}"} || code=$?
    newest=$(ls -d "$SHIPNAV_OUTPUT"/chunk-* 2>/dev/null | sort | tail -1)
    if [[ -n "$newest" && -f "$newest/manifest.json" ]]; then publish "$newest"; fi
    if ((code==3)); then state=stopped; exit 0; fi
    if ((code!=0)); then exit "$code"; fi
done
state=complete
```

`set -e` makes a failed `publish` or a failed final status upload exit non-zero, and the systemd unit then records the failure. If `status.json` is missing from S3 after the deadline, treat the run as failed.

- [ ] **Step 4: Run tests**

Run `python -m pytest tests/test_cloud.py -v`.
Expected: PASS. The runner's chunk publishing, SIGTERM and stop/start behaviour need GNU `timeout` and systemd, so they are verified on Linux in the Task 14 stop/resume pilot, before any multi-seed launch.

- [ ] **Step 5: Commit**

```bash
git add cloud tests/test_cloud.py
git commit -m "feat: add EC2 launch, SSM and checkpoint-publishing tooling"
```

### Task 14: EC2 execution runbook (AWS CLI only)

This task launches paid resources. Before running any command marked **paid**, record these in `cloud/local/run.json` and confirm them explicitly: AWS profile/account, region, total spending limit, maximum wall time, S3 bucket/prefix, instance type, and volume size. Singapore is the user's timezone, not a chosen AWS region. Every command below uses the selected existing `AWS_PROFILE` and `AWS_REGION`.

**Discover and price (read-only)**

- [ ] Verify the session and save the regional candidates, offerings and quotas:

```bash
aws --version
aws sts get-caller-identity
aws ec2 describe-instance-types --filters Name=current-generation,Values=true --output json > cloud/local/instance-types.json
aws ec2 describe-instance-type-offerings --location-type availability-zone --output json > cloud/local/offerings.json
aws service-quotas list-service-quotas --service-code ec2 --output json > cloud/local/ec2-quotas.json
```

- [ ] Shortlist two CPU instance types (x86_64, enough vCPUs for `--envs` + 1) and, only if the local pilot suggests a learner-bound workload, one single-GPU type. Quote On-Demand prices and save each response with a timestamp:

```bash
aws pricing get-products --region us-east-1 --service-code AmazonEC2 --output json \
  --filters Type=TERM_MATCH,Field=instanceType,Value="$SHIPNAV_INSTANCE_TYPE" \
            Type=TERM_MATCH,Field=regionCode,Value="$AWS_REGION" \
            Type=TERM_MATCH,Field=operatingSystem,Value=Linux Type=TERM_MATCH,Field=tenancy,Value=Shared \
            Type=TERM_MATCH,Field=capacitystatus,Value=Used Type=TERM_MATCH,Field=preInstalledSw,Value=NA \
            Type=TERM_MATCH,Field=licenseModel,Value="No License required" \
  > "cloud/local/price-$SHIPNAV_INSTANCE_TYPE.json"
```

Expect exactly one product per type. Then run `python cloud/budget.py ...` for the planned hours. The estimate must stay under the approved limit, including stopped-EBS days and any NAT or endpoint charges.

- [ ] Resolve the AMI. For CPU runs, use the current official Amazon Linux 2023 AMI SSM parameter. For GPU runs, look up the current Base OSS NVIDIA Driver DLAMI parameter path in the [DLAMI ID discovery docs](https://docs.aws.amazon.com/dlami/latest/devguide/find-dlami-id.html) and store it as `SHIPNAV_AMI_PARAMETER`:

```bash
aws ssm get-parameter --name "$SHIPNAV_AMI_PARAMETER" --query Parameter.Value --output text
aws ec2 describe-images --image-ids "$SHIPNAV_AMI_ID" --query 'Images[0].{Id:ImageId,Owner:OwnerId,Arch:Architecture,Root:RootDeviceName,Name:Name,Created:CreationDate}'
```

Check the owner, x86_64 architecture, root device name, and minimum volume size, plus glibc ≥ 2.28 (needed for the locked manylinux_2_28 wheels). For a GPU AMI, also check that the driver supports the CUDA version reported by the locked torch wheel (the smoke step prints `torch.version.cuda`).

- [ ] Inspect and reuse a subnet, security group and instance profile, or create run-specific ones and record them:
  - SSM access, **no inbound rules**, IMDSv2, encrypted EBS.
  - The instance role has `AmazonSSMManagedInstanceCore` plus `s3:GetObject/PutObject/ListBucket` limited to `arn:aws:s3:::$SHIPNAV_BUCKET/$SHIPNAV_RUN_PREFIX/*` (and its KMS key if one is used).
  - HTTPS egress to SSM, S3, PyPI and astral.sh.
  - No access keys in user data, files or S3.
- [ ] Create a scheduler execution role trusted by `scheduler.amazonaws.com`, allowed only `ec2:StopInstances` on instances tagged `Project=shipnav`.

**Launch (paid)**

- [ ] Package the source. Confirm the commit exists, then upload only `rebuild/` implementation, locks and `scenarios/training` (no venvs, results, docs, PDFs or credentials):

```bash
git archive --format=tar.gz -o cloud/local/source.tar.gz HEAD rebuild/src rebuild/cloud rebuild/tests rebuild/scenarios/training rebuild/pyproject.toml rebuild/uv.lock
(cd cloud/local && sha256sum source.tar.gz > source.sha256)
aws s3 cp cloud/local/source.tar.gz "s3://$SHIPNAV_BUCKET/$SHIPNAV_RUN_PREFIX/source.tar.gz"
aws s3 cp cloud/local/source.sha256 "s3://$SHIPNAV_BUCKET/$SHIPNAV_RUN_PREFIX/source.sha256"
```

The `git archive` paths are repository-relative (run it from the repository root, or use `git -C ..` from `rebuild/`). The tests need `tests/fixtures/engine_parity.json`, which this includes.

- [ ] Generate the request, check permission with `--dry-run`, and launch. `--dry-run` exits non-zero with `DryRunOperation` when permission is sufficient; treat `UnauthorizedOperation` or any other error as a stop.

```bash
python cloud/prepare_launch.py --ami "$SHIPNAV_AMI_ID" --instance-type "$SHIPNAV_INSTANCE_TYPE" --subnet "$SHIPNAV_SUBNET" \
  --group "$SHIPNAV_GROUP" --profile "$SHIPNAV_PROFILE" --root "$SHIPNAV_ROOT_DEVICE" --run-id "$SHIPNAV_RUN_ID" \
  --hours "$SHIPNAV_HOURS" --volume-gib "$SHIPNAV_VOLUME_GIB"
aws ec2 run-instances --cli-input-json file://cloud/local/launch.json --dry-run
aws ec2 run-instances --cli-input-json file://cloud/local/launch.json --output json > cloud/local/launched.json
aws ec2 wait instance-status-ok --instance-ids "$SHIPNAV_INSTANCE_ID"
aws ssm describe-instance-information --filters Key=InstanceIds,Values="$SHIPNAV_INSTANCE_ID"
```

If a call times out, retry with the same `ClientToken`, which never launches a duplicate.

- [ ] Create the independent cost cap immediately after launch. Set the stop time to the wall time plus upload grace, in UTC:

```bash
aws scheduler create-schedule --name "shipnav-stop-$SHIPNAV_RUN_ID" \
  --schedule-expression "at($SHIPNAV_STOP_AT_UTC)" --flexible-time-window Mode=OFF \
  --target "{\"Arn\":\"arn:aws:scheduler:::aws-sdk:ec2:stopInstances\",\"RoleArn\":\"$SHIPNAV_SCHEDULER_ROLE\",\"Input\":\"{\\\"InstanceIds\\\":[\\\"$SHIPNAV_INSTANCE_ID\\\"]}\"}"
```

Universal-target format: [EventBridge Scheduler docs](https://docs.aws.amazon.com/scheduler/latest/UserGuide/managing-targets-universal.html). Tags and budget alarms never stop an instance; this schedule and the on-host `shutdown` do.

**Set up, verify, pilot (paid)**

- [ ] Write `cloud/local/settings.json` (keys as in `tests/test_cloud.py::SETTINGS`; `uv_version` is the local `uv --version` number, so the host installs the same uv). For `KIND` in `setup`, then `smoke`, run `python cloud/ssm_commands.py "$KIND" --settings cloud/local/settings.json`, then:

```bash
aws ssm send-command --document-name AWS-RunShellScript --instance-ids "$SHIPNAV_INSTANCE_ID" \
  --parameters file://"cloud/local/$KIND-commands.json" \
  --output-s3-bucket-name "$SHIPNAV_BUCKET" --output-s3-key-prefix "$SHIPNAV_RUN_PREFIX/logs/$KIND" \
  --query Command.CommandId --output text
aws ssm get-command-invocation --command-id "$SHIPNAV_COMMAND_ID" --instance-id "$SHIPNAV_INSTANCE_ID" \
  --query '{Status:Status,Code:ResponseCode}'
```

An SSM status of `Success` means the step succeeded only when `ResponseCode` is 0.
- [ ] Run `tools/training_pilot.py` through a one-off SSM command (executionTimeout 3600). Download the JSON, then compute cost per million env steps = `price / (learn_steps_per_s × 3600) × 1e6` for each shortlisted type. Choose the cheapest type that finishes inside the wall-time limit. If it differs from the pilot instance, stop and terminate the pilot instance, and repeat the launch with the chosen type.
- [ ] **Stop/resume pilot:**
  1. Start a 3-chunk run (`start` document).
  2. After `chunk-0000` appears in S3, `aws ec2 stop-instances`, wait for `instance-stopped`, then `start-instances`.
  3. Re-send the `start` document with the same output prefix. The runner resumes from the latest local verified checkpoint, and the EBS volume persisted.
  4. Confirm `status.json` says `complete` and the timesteps continue without a gap.
  5. Download one checkpoint and run `ModernLearned` locally on CPU through the service.

  Only after this, consider Spot for later runs.

**Train every seed (paid)**

- [ ] For each `variant × seed` in `training_config.json`, start one run (sequentially on one instance, or one instance per run within the approved budget), each with its own `--prefix .../<variant>/seed<k>`. Monitor only `status.json` and `curve.jsonl` in S3. A successful process exit is not a performance claim.
- [ ] Record per run: instance type, region, AMI, GPU/driver/CUDA (if any), lock hash, code commit, corpus hash, seed, steps/s, runtime, estimated cost, SSM command IDs and checkpoint hashes, all in `cloud/local/runs.jsonl`.

**Download, verify and clean up**

- [ ] For each run, download the checkpoints and verify them; folders without `manifest.json` are incomplete and are listed, not used:

```bash
aws s3 sync "s3://$SHIPNAV_BUCKET/$SHIPNAV_RUN_PREFIX/checkpoints" "results/train/$VARIANT/seed$SEED"
python -c "import sys;from pathlib import Path;from shipnav.training.checkpoints import verify_checkpoint;[print(d.name, verify_checkpoint(d)['num_timesteps']) for d in sorted(Path(sys.argv[1]).iterdir()) if (d/'manifest.json').is_file()]" "results/train/$VARIANT/seed$SEED"
```

Never compare S3 ETags with SHA256; the manifest hashes are authoritative.

- [ ] Stop the instance (`aws ec2 stop-instances`, then `wait instance-stopped`). If any run is incomplete, keep the stopped disk until recovery is decided.
- [ ] After verified recovery, terminate only the recorded instance and wait, then delete the stop schedule:

```bash
aws ec2 terminate-instances --instance-ids "$SHIPNAV_INSTANCE_ID"
aws ec2 wait instance-terminated --instance-ids "$SHIPNAV_INSTANCE_ID"
aws scheduler delete-schedule --name "shipnav-stop-$SHIPNAV_RUN_ID"
```

- [ ] Check for residual charges with tag filters: volumes (`aws ec2 describe-volumes --filters Name=tag:RunId,Values=$SHIPNAV_RUN_ID`), snapshots, Elastic IPs, run-specific NAT gateways and endpoints. Delete only resources created for this run. Keep the selected S3 checkpoints, curves and logs, and delete source archives under the run prefix once no longer needed. Save the cleanup evidence and the actual runtime/cost estimate in `cloud/local/cleanup-$SHIPNAV_RUN_ID.json`.

### Task 15: Held-out evaluation and documentation

**Files:**
- Create: `evidence/retraining/README.md`, `evidence/retraining/summary.json`, `evidence/retraining/curves/` (copied `curve.jsonl` per run), `evidence/retraining/selections/` (copied `selection.json` per run)
- Modify: `README.md`

- [ ] **Step 1: Select checkpoints for every run**

Run `python tools/select_checkpoints.py results/train/<variant>/seed<k>` for every run, and copy each `selection.json` and `curve.jsonl` into `evidence/retraining/`.

- [ ] **Step 2: Freeze the protocol** (Task 12 Step 5), and commit it before any held-out command.

- [ ] **Step 3: Run, audit and summarise on CPU with the baseline thread settings**

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv-modern/bin/python tools/retrain_eval.py heldout --output results/retraining-heldout \
  --protocol-commit "$PROTOCOL_COMMIT" --protocol-sha256 "$PROTOCOL_SHA256"
.venv-modern/bin/python tools/retrain_eval.py audit --output results/retraining-heldout --protocol-commit "$PROTOCOL_COMMIT" --protocol-sha256 "$PROTOCOL_SHA256"
.venv-modern/bin/python tools/retrain_eval.py summarize --output results/retraining-heldout --protocol-commit "$PROTOCOL_COMMIT" --protocol-sha256 "$PROTOCOL_SHA256"
```

Set `PROTOCOL_COMMIT`/`PROTOCOL_SHA256` from the Task 12 freeze commit. Expected: the complete grid (110 × arms), the audit passes, and `summary.json` has per-arm targets and hierarchical intervals. Copy `summary.json` into `evidence/retraining/`.

- [ ] **Step 4: Write `evidence/retraining/README.md` and the README section**

Report every arm, every seed, failures and timeouts. Report Southern Islands results separately as zero-shot transfer, and give the constraint-budget violation rate for both learned variants even where reward improved. Label the intervals exploratory. In `README.md`, add short user instructions: generating the corpus, a local smoke training run, loading a PPO checkpoint in the GUI (marine dynamics required), and where the evidence lives.

- [ ] **Step 5: Commit**

```bash
git add evidence/retraining README.md
git commit -m "docs: record preregistered retraining results and usage"
```

## Self-review against the spec

- **Spec item 7:** the Gymnasium API and observation/action contract are Tasks 3, 5 and 6. The prediction-aware PPO baseline uses perceived age/margin inputs (Task 5). The constrained-reward ablation with a logged budget is covered by Task 6 (cost), Task 8 (update) and Task 10 (budget). Multiple seeds come from Task 10 (≥ 3, five by default), resumable EC2 jobs via AWS CLI from Tasks 7, 8, 13 and 14, held-out evaluation from Tasks 12 and 15, and GUI model loading from Task 9.
- **Review fixes:**
  - stale extraction script → hand refactor plus digest gate (Tasks 1, 3)
  - invalid map fixture → `make_scenario`/`SeaMap.to_dict` (Task 6)
  - 10-scenario training set → disjoint 5,000-scenario corpus (Task 4)
  - divergent setup → `shipnav.episode` (Task 2)
  - baseline protocol invalidated → new protocol re-running comparators (Task 12)
  - degenerate cost → truth domain exposure (Tasks 3, 6)
  - missing `maps` extra → `--extra maps` (Task 13)
  - SSM 1-hour limit → detached systemd unit, explicit timeouts (Task 13)
  - GPU default → CPU-first measured pilot (Tasks 10, 14)
  - non-atomic upload → manifest-last publish and status file (Task 13)
  - state not saved → full checkpoints on every chunk (Tasks 7, 8)
  - reward/observation mismatch → route progress (Task 6)
  - hardcoded degraded sensing → `EnvConfig.observation`, exact for the primary runs (Tasks 6, 10)
  - unbounded box → `[-1, 1]` schema (Task 5)
  - heavy info and traces → `record=False` (Tasks 3, 6)
  - duplicate lock → removed
  - no firm cap → EventBridge Scheduler (Task 14)
  - pricing filters → completed (Task 14)
  - extra terminal observe → only with `terminal_packet` (Task 3)
  - per-arm hash validation bug → `expected_model_hashes` (Task 12)
- **Still open (set before the held-out run, not after):** the account/region/instance/spending limit (Task 14) and the total step budget (Task 10). The success/collision targets carry over from the baseline protocol; the constraint budget is fixed in Task 10.

References: [EC2 run-instances](https://docs.aws.amazon.com/cli/latest/reference/ec2/run-instances.html), [regional offerings](https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-instance-type-offerings.html), [SSM send-command](https://docs.aws.amazon.com/cli/latest/reference/ssm/send-command.html), [Run Command timeouts](https://repost.aws/knowledge-center/systems-manager-run-command), [EventBridge Scheduler universal targets](https://docs.aws.amazon.com/scheduler/latest/UserGuide/managing-targets-universal.html), [DLAMI ID discovery](https://docs.aws.amazon.com/dlami/latest/devguide/find-dlami-id.html), [SB3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), [Gymnasium migration](https://gymnasium.farama.org/introduction/migration_guide/). Refresh instance, AMI and pricing information at execution time; this revision queried no account or cloud resources.
