# 03b — Marine dynamics and external benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare controllers under bounded vessel motion and connect external maritime scenarios.

**Architecture:** Keep the holonomic reference and introduce a separately labelled marine tier. Share dynamics across policy tracking, safety rollouts and MPC; test coordinate adapters before importing benchmarks.

**Tech Stack:** Modern environment from 01b; standard-library kinematics; optional isolated CommonOcean and NTNU simulator environments

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

All implementation paths below are relative to `rebuild/`. Run commands there. These are instructions and reference snippets, not installed software or benchmark results.

## File map and model limits

Create `src/shipnav/dynamics.py`, `guidance.py`, `controllers/__init__.py`, `controllers/mpc.py`, `reactive.py`, `adapters/__init__.py`, `adapters/coordinates.py`, and corresponding tests. Extend `simulation.py` and `service.py`. `guidance.py` exposes the velocity-reference tracker implemented by `advance`; keep `from shipnav.dynamics import advance` there until a richer tracker is needed.

The first marine model is a discrete heading/speed model with 0.2 m/s² acceleration and 0.35 rad/s yaw-rate limits in this 24 m synthetic scene. These are demo parameters, not calibrated ship data. Expose them in scenario/model metadata; circular radius .5 m conservatively represents the test vessel. Record the discrete integration scheme. A Fossen/NTNU model replaces this tier only after units, step size, hull and control limits are validated. Do not claim hydrodynamic realism from these two bounds.

### D1 — Use the same reachable dynamics in motion and filtering

**Files:** `dynamics.py`, `guidance.py`, `tests/test_dynamics.py`

**Interfaces:** `Vessel(x,y,heading,speed)`; `advance(state,desired,dt,max_speed=1.,acceleration=.2,yaw_rate=.35)->Vessel`; `motion_from(state,max_speed)->callable(desired,dt,steps)->list[Point]`. Desired is a velocity reference; actual speed/heading are bounded.

- [ ] Write the behavior test:

File: `rebuild/tests/test_dynamics.py`

```python
from math import hypot
from shipnav.dynamics import Vessel,advance,motion_from
from shipnav.maps import SeaMap
from shipnav.simulation import run_episode
from shipnav.policies import Direct

def test_heading_and_acceleration_cannot_jump():
    a=Vessel(2,2,0,1)
    b=advance(a,(-1,0),.25)
    assert abs(b.heading-a.heading)<=.35*.25+1e-12
    assert abs(b.speed-a.speed)<=.2*.25+1e-12
    assert b.x>a.x  # inertia prevents instantaneous reversal
    assert motion_from(a)((-1,0),.25,1)[1]==(b.x,b.y)

def test_marine_episode_uses_reachable_motion():
    r=run_episode(SeaMap((0,0,24,24)),[(2,2),(20,2)],[],Direct(),dynamics='marine',limit=2)
    assert r['frames'][1]['position'][0]-2 < .25
    assert all(d['speed']<=1 for d in r['diagnostics'])
```

- [ ] Run `python -m pytest tests/test_dynamics.py tests/test_safety_pipeline.py -v`. Expect a missing module or unmet behavior, not a missing third-party dependency.
- [ ] Implement the following content:

File: `rebuild/src/shipnav/dynamics.py`

```python
from dataclasses import dataclass
from math import atan2, hypot, sin, cos, pi

@dataclass(frozen=True)
class Vessel:
    x: float
    y: float
    heading: float
    speed: float

def advance(state, desired, dt, max_speed=1., acceleration=.2, yaw_rate=.35):
    if dt<=0 or min(max_speed,acceleration,yaw_rate)<=0:
        raise ValueError('Positive dynamics limits required')
    target = atan2(desired[1],desired[0]) if hypot(*desired)>1e-9 else state.heading
    error = (target-state.heading+pi)%(2*pi)-pi
    heading = state.heading+max(-yaw_rate*dt,min(yaw_rate*dt,error))
    preferred = min(max_speed,hypot(*desired))*max(0.,cos(error))
    speed = state.speed+max(-acceleration*dt,min(acceleration*dt,preferred-state.speed))
    return Vessel(state.x+speed*cos(heading)*dt,state.y+speed*sin(heading)*dt,heading,speed)

def motion_from(state,max_speed=1.):
    def motion(desired,dt,steps):
        current=state; path=[(current.x,current.y)]
        for _ in range(steps):
            current=advance(current,desired,dt,max_speed)
            path.append((current.x,current.y))
        return path
    return motion
```

File: `rebuild/src/shipnav/guidance.py`

```python
from shipnav.dynamics import advance
```

- [ ] Run `python -m pytest tests/test_dynamics.py tests/test_safety_pipeline.py -v` again. Inspect failures before proceeding.
- [ ] Review and commit only the implementation/test files named above after the check passes; keep `docs/` ignored.

- [ ] Add near-shore braking and high-speed turning fixtures; test both filtered and unfiltered controllers. The filter must use `motion_from` rather than pretending marine motion can instantaneously stop. Refine the integration timestep and compare collision/clearance outputs; when adding continuous curved dynamics, bound chord approximation or use substeps before interpreting clearance.

### D2 — Add a reproducible MPC comparator

**Files:** `controllers/mpc.py`, `tests/test_mpc.py`, modify `simulation.py`

**Interfaces:** `MPC.set_context(sea,vessel,predictions)` then the same policy callable as SARL; `solver_failed` records no feasible candidate.

- [ ] Write the behavior test:

File: `rebuild/tests/test_mpc.py`

```python
from shipnav.controllers.mpc import MPC
from shipnav.dynamics import Vessel
from shipnav.maps import SeaMap

def test_mpc_has_explicit_infeasibility_and_bounded_command():
    m=MPC(); m.set_context(SeaMap((0,0,24,24)),Vessel(2,2,0,0),[])
    u=m((2,2),(0,0),(20,2),[],.5,1,.25)
    assert u[0]>0 and not m.solver_failed
    bad=[{'radius':100.,'points':[(2,2)]*13,'margins':[0.]*13}]
    m.set_context(SeaMap((0,0,24,24)),Vessel(2,2,0,0),bad)
    m((2,2),(0,0),(20,2),[],.5,1,.25)
    assert m.solver_failed
```

- [ ] Run `python -m pytest tests/test_mpc.py -v`. Expect a missing module or unmet behavior, not a missing third-party dependency.
- [ ] Implement the following content:

File: `rebuild/src/shipnav/controllers/mpc.py`

```python
from math import sin,cos,pi,dist
from shipnav.dynamics import motion_from
from shipnav.safety import assess,rollout

class MPC:
    def set_context(self,sea,vessel,predictions):
        self.sea,self.vessel,self.predictions=sea,vessel,predictions

    def __call__(self,p,v,goal,neighbours,radius,speed,dt):
        # Finite-control-set MPC: constant velocity references over a finite horizon.
        # The tracker integrates bounded acceleration and yaw rate, then only step 1 executes.
        motion=motion_from(self.vessel,speed) if self.vessel is not None else None
        commands=[(0.,0.)]+[(s*cos(k*pi/12),s*sin(k*pi/12)) for s in (speed*.5,speed) for k in range(24)]
        ranked=[]
        for command in commands:
            path=rollout(p,command,dt,12,motion)
            clearance=assess(self.sea,path,self.predictions,radius)
            cost=dist(path[-1],goal)+.1*dist(command,v)
            ranked.append((clearance, cost, command))
        feasible=[r for r in ranked if r[0]>0]
        self.solver_failed=not bool(feasible)
        if feasible:
            return min(feasible,key=lambda r:r[1])[2]
        return max(ranked,key=lambda r:(r[0],-r[1]))[2]
```

- [ ] Run `python -m pytest tests/test_mpc.py -v` again. Inspect failures before proceeding.
- [ ] Review and commit only the implementation/test files named above after the check passes; keep `docs/` ignored.

Insert these lines immediately before `inference_start = perf_counter()` in Task 6c's simulator. Add `solver_failed=bool(getattr(policy, 'solver_failed', False))` to each diagnostic record after the policy call. The decision timer includes MPC/context/prediction/filter; inference timer measures controller evaluation only.

```python
if hasattr(policy, 'set_context'):
    policy.set_context(sea, vessel, predictions)
```

This is finite-control-set receding-horizon MPC, not the RMPC-CBF research implementation or a nonlinear solver guarantee. To compare an upstream NMPC solver, first run its authors' example in isolation, then implement the same callable/context interface and log status, residual, solve time and fallback. Candidate exhaustion and deadline miss remain failures even if a fallback subsequently finishes the episode. Record deadline misses; do not claim a Python worker thread enforces hard real time.

### D3 — Add reactive and course-change traffic without changing pair inputs

**Files:** `src/shipnav/reactive.py`, `tests/test_reactive.py`, scenario fixtures. Reactive targets use a simple collision-responsive heading change as a controlled test. Do not label this handcrafted rule COLREGs compliant or ORCA.

File: `rebuild/src/shipnav/reactive.py`

```python
from math import dist,hypot
from shipnav.simulation import Traffic

class ReactiveTraffic:
    def __init__(self,ship,sea):
        self.start,self.goal,self.speed,self.radius=ship.start,ship.goal,ship.speed,ship.radius
        self.sea,self.history=sea,[(0.,self.start,(0.,0.))]
        self.arrival=float('inf')

    def at(self,t):
        for (ta,pa,va),(tb,pb,vb) in zip(self.history,self.history[1:]):
            if ta<=t<tb:
                f=(t-ta)/(tb-ta)
                return tuple(a+f*(b-a) for a,b in zip(pa,pb)),vb,self.radius
        _,p,v=self.history[-1]
        return p,v,self.radius

    def advance(self,t,dt,ego):
        p=self.history[-1][1]
        d=dist(p,self.goal)
        v=tuple((b-a)/d*min(self.speed,d/dt) if d else 0. for a,b in zip(p,self.goal))
        if dist(p,ego)<3.:
            away=(p[0]-ego[0],p[1]-ego[1]); norm=hypot(*away)
            if norm:
                v=tuple(self.speed*x/norm for x in away)
        q=tuple(x+dt*y for x,y in zip(p,v))
        if not self.sea.clear(p,q,self.radius):
            q,v=p,(0.,0.)
        self.history.append((t+dt,q,v))
```

File: `rebuild/tests/test_reactive.py`

```python
from shipnav.reactive import ReactiveTraffic
from shipnav.simulation import Traffic
from shipnav.maps import SeaMap

def test_reaction_changes_path_but_preserves_history():
    s=ReactiveTraffic(Traffic((5,5),(10,5)),SeaMap((0,0,24,24)))
    s.advance(0,.25,(6,5))
    assert s.at(.25)[0][0]<5
    assert s.at(0)[0]==(5,5)
```

- [ ] Run `python -m pytest tests/test_reactive.py -v` failing before adding the class and passing after it. For `traffic_mode='reactive'`, wrap the scenario's fixed initial voyages in `ReactiveTraffic` when the service starts. Truth advances only after ego control selection; delayed observations interpolate saved history, not a regenerated trajectory.
- [ ] Add a scripted course-change variant by storing a list of `(time, position)` points and using linear interpolation in `at(t)`; split swept truth checks at **every** course-change time inside a step, just as fixed voyages split at arrival. Keep that extension's test red until a crossing at the turn is detected. For the required first course-change benchmark, the reactive fixture above supplies an actual change of course; explicitly label it reactive rather than scripted.
- [ ] Paired reactive experiments compare the same initial scenario/hash, target rule and exogenous noise. Do not assert identical realized target paths when the ego differs. Score target/target and target/land invalidity separately from ego performance.

### D4 — Verify maritime adapter conventions and installation

Create `adapters/coordinates.py` with the functions below and `tests/test_coordinates.py`. For NTNU North–East positions/headings use an explicit conversion; CommonOcean inputs already in Cartesian coordinates require their own verified metadata, not this transform by default.

File: `rebuild/src/shipnav/adapters/coordinates.py`

```python
from math import pi

def north_east_to_xy(north,east,heading_from_north):
    return east,north,(pi/2-heading_from_north+pi)%(2*pi)-pi

def conservative_radius(length,width):
    from math import hypot
    if min(length,width)<=0:
        raise ValueError('Positive hull dimensions required')
    return hypot(length,width)/2
```

File: `rebuild/tests/test_coordinates.py`

```python
from math import pi
import pytest
from shipnav.adapters.coordinates import north_east_to_xy,conservative_radius

def test_north_east_heading_and_hull():
    assert north_east_to_xy(10,20,0)==pytest.approx((20,10,pi/2))
    assert north_east_to_xy(10,20,pi/2)==pytest.approx((20,10,0))
    assert conservative_radius(4,3)==2.5
```

- [ ] Run `python -m pytest tests/test_coordinates.py -v` before/after the adapter. Add round-trip points, radians/degrees, timestep and hull-corner tests for each imported source.
- [ ] In disposable environments under `migration/external/`, install the current [CommonOcean tooling](https://commonocean.cps.cit.tum.de/) and [NTNU colav-simulator](https://github.com/ntnu-itk-autonomous-ship-lab/colav-simulator) using their checked-out, pinned revision instructions. Record licenses, dependency locks, native libraries and exact sample scenario IDs. Read installed APIs before writing bindings; an install failure is a recorded compatibility result, not permission to constrain the entire modern app to an old interpreter.
- [ ] Export one upstream scenario into the canonical schema, saving original waters/polygons, actor shape/dynamics, control limits and coordinate transform under `external_metadata`. Rectangular land approximations must be conservative and labelled; reject unsupported time-varying geometry rather than silently dropping it. Keep exact upstream data for the upstream feasibility checker even when the local view uses circular hulls.
- [ ] Add `adapters/commonocean.py` with `import_scenario(path)->dict` and `check_trajectory(original_path, result)->dict` against the actual installed API; test a known feasible sample and deliberate water-boundary/collision/yaw-rate violations. Run the upstream checker as the independent oracle. Add `adapters/ntnu.py` with the same canonical export boundary if the spike supports this platform. External adapter implementation is an API-verification task: acceptance requires the real upstream example and round-trip check, not a fabricated mock API.
- [ ] Decide at this gate whether NTNU replaces the synthetic marine backend or remains an external evaluator. Prefer reuse when its model/observations fit; keep unit-test geometry independent. Save the decision with evidence in `migration/external/decision.md`. Full COLREGs legality is outside the meaning of either feasibility checker.

#### Concrete CommonOcean collision-oracle starting point

The following binding uses the documented reader/trajectory/collision interfaces. Its upstream checker evaluates discrete time points; keep local swept checks and test the gap between samples. This is only the collision component of `check_trajectory`; water-boundary and model-feasibility checks remain separately named task requirements, and must not be represented as already passed by this result. [Official CommonOcean interface tutorial](https://commonocean-documentation.readthedocs.io/en/latest/commonocean-dc/doc/docs/source/02_commonocean_interface.html).

File: `rebuild/src/shipnav/adapters/commonocean.py (collision oracle)`

```python
def check_collision(original_path,result,length,width):
    import numpy as np
    from commonocean.common.file_reader import CommonOceanFileReader
    from commonocean.scenario.state import GeneralState
    from commonocean.scenario.trajectory import Trajectory
    from commonocean.prediction.prediction import TrajectoryPrediction
    from commonroad.geometry.shape import Rectangle
    from commonocean_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_checker,create_collision_object
    scenario,_=CommonOceanFileReader(str(original_path)).open()
    dt=float(scenario.dt)
    frames=result['frames']; ds=result['diagnostics']
    states=[]
    for k,frame in enumerate(frames):
        if abs(frame['t']-k*dt)>1e-6:
            raise ValueError('Resample to the upstream timestep before checking')
        orientation=0. if k==0 else ds[k-1]['heading']
        states.append(GeneralState(time_step=k,position=np.asarray(frame['position']),orientation=orientation))
    predicted=TrajectoryPrediction(Trajectory(0,states),Rectangle(length=length,width=width))
    return {'discrete_collision':bool(create_collision_checker(scenario).collide(create_collision_object(predicted))),
            'continuous_collision_checked':False}
```

- [ ] Run this function on the tutorial's known colliding trajectory and a deliberately clear trajectory in the installed external environment. Test a mismatched dt raises the explicit error. Save fixture provenance and upstream revision with expected results.
