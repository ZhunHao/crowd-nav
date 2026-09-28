# 02 — Land maps and global planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute clearance-aware routes and intermediate goals on a loadable land map.

**Architecture:** Use immutable rectangular map geometry and an eight-neighbour grid A* search. Reduce redundant waypoints only when the complete connecting segment is clear at the chosen vessel clearance.

**Tech Stack:** Python standard library, pytest; stage 1 package installation

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

## File structure and execution order

Create `rebuild/src/shipnav/maps.py`, `rebuild/src/shipnav/planning.py`, `rebuild/maps/harbour.json`, `rebuild/tests/test_maps.py`, and `rebuild/tests/test_planning.py`. Neither module imports PyTorch, Gym or Qt. Complete Task 2 before Task 3.

Map rectangles use `[xmin, ymin, xmax, ymax]`; values are metres. The map is a deliberately synthetic harbour, not imported geospatial data. Clearance 0.7 m means the 0.5 m robot radius plus a proposed 0.2 m planning margin. These numbers are design choices, not briefing requirements.

### Task 2: Load and validate land maps

**Files:** `rebuild/src/shipnav/maps.py`, `rebuild/tests/test_maps.py`, `rebuild/maps/harbour.json`.

**Interfaces:** Produces `SeaMap(bounds: Rect, land: tuple[Rect, ...])`, `clear(a: Point, b: Point, clearance: float) -> bool`, `load(Path) -> SeaMap`, `save(Path) -> None`, `to_dict() -> dict`, `from_dict(dict) -> SeaMap`. Touching inflated land or the inset boundary is blocked.

- [ ] **Step 1: Write the behaviour tests.**

Create `rebuild/tests/test_maps.py` with:

```python
import pytest
from shipnav.maps import SeaMap


def test_swept_segment_cannot_jump_over_land():
    sea = SeaMap((0, 0, 10, 10), ((4, 4, 6, 6),))
    assert sea.clear((2, 2), (8, 2), .5)
    assert not sea.clear((2, 5), (8, 5), .5)
    assert not sea.clear((3.5, 3), (3.5, 7), .5)
    assert not sea.clear((.5, 2), (2, 2), .5)


def test_map_roundtrip_and_bad_rectangles(tmp_path):
    sea = SeaMap((0, 0, 10, 10), ((4, 4, 6, 6),))
    path = tmp_path / 'map.json'
    sea.save(path)
    assert SeaMap.load(path) == sea
    with pytest.raises(ValueError):
        SeaMap((0, 0, 10, 10), ((8, 8, 7, 9),))
    with pytest.raises(ValueError):
        SeaMap((0, 0, float('nan'), 10))
```

- [ ] **Step 2: Run the tests before implementation.**

```bash
python -m pytest tests/test_maps.py -v
```

Expected: failure because the new module or behaviour does not exist. Missing environment dependencies must be resolved before treating a test failure as evidence about application behaviour.

- [ ] **Step 3: Implement the following files.**

Create `rebuild/src/shipnav/maps.py` with:

```python
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import json

Point = tuple[float, float]
Rect = tuple[float, float, float, float]


def intersects(a: Point, b: Point, rect: Rect) -> bool:
    low, high = 0.0, 1.0
    for p, q, lo, hi in zip(a, b, rect[:2], rect[2:]):
        delta = q - p
        if abs(delta) < 1e-12:
            if p < lo or p > hi:
                return False
        else:
            enter, leave = sorted(((lo - p) / delta, (hi - p) / delta))
            low, high = max(low, enter), min(high, leave)
            if low > high:
                return False
    return True


@dataclass(frozen=True)
class SeaMap:
    bounds: Rect
    land: tuple[Rect, ...] = ()

    def __post_init__(self):
        for rect in (self.bounds, *self.land):
            if len(rect) != 4 or not all(isfinite(v) for v in rect):
                raise ValueError('Rectangles require four finite coordinates')
            if rect[0] >= rect[2] or rect[1] >= rect[3]:
                raise ValueError('Rectangles require positive width and height')
        x0, y0, x1, y1 = self.bounds
        for a, b, c, d in self.land:
            if not (x0 <= a < c <= x1 and y0 <= b < d <= y1):
                raise ValueError('Land must lie within map bounds')

    def clear(self, a: Point, b: Point, clearance: float) -> bool:
        if not isfinite(clearance) or clearance < 0:
            raise ValueError('Clearance must be finite and non-negative')
        if not all(isfinite(v) for v in (*a, *b)):
            return False
        x0, y0, x1, y1 = self.bounds
        for x, y in (a, b):
            if not (x0 + clearance < x < x1 - clearance
                    and y0 + clearance < y < y1 - clearance):
                return False
        for x0, y0, x1, y1 in self.land:
            inflated = (x0-clearance, y0-clearance, x1+clearance, y1+clearance)
            if intersects(a, b, inflated):
                return False
        return True

    def to_dict(self) -> dict:
        return {'bounds': list(self.bounds), 'land': [list(r) for r in self.land]}

    @classmethod
    def from_dict(cls, data: dict):
        if set(data) != {'bounds', 'land'}:
            raise ValueError('Map requires exactly bounds and land')
        return cls(tuple(map(float, data['bounds'])),
                   tuple(tuple(map(float, r)) for r in data['land']))

    @classmethod
    def load(cls, path: Path):
        return cls.from_dict(json.loads(path.read_text()))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False))
```

Create `rebuild/maps/harbour.json`:

```json
{"bounds": [0, 0, 24, 24], "land": [[9, 5, 14, 18]]}
```

- [ ] **Step 4: Run the same test command and inspect the result.**

```bash
python -m pytest tests/test_maps.py -v
```

Expected: PASS. Do not weaken assertions to conceal an algorithm or interface failure.

- [ ] **Step 5: Review the task's diff and save a checkpoint.**

From the workspace root, if it is a Git repository:

```bash
git add rebuild/src/shipnav/maps.py rebuild/tests/test_maps.py rebuild/maps/harbour.json
git commit -m "feat: load and validate land maps"
```

### Task 3: Find routes and intermediate goals

**Files:** `rebuild/src/shipnav/planning.py`, `rebuild/tests/test_planning.py`.

**Interfaces:** Consumes `SeaMap.clear`. Produces `astar(sea, start, goal, clearance=.7, resolution=1.0) -> list[Point]`, `smooth(sea, route, clearance=.7) -> list[Point]`, and `NoPath(ValueError)`. Both route lists include start and final goal; the simulator uses `route[1:]`.

- [ ] **Step 1: Write the behaviour tests.**

Create `rebuild/tests/test_planning.py` with:

```python
import pytest
from shipnav.maps import SeaMap
from shipnav.planning import astar, smooth, NoPath


def test_route_and_smoothing_preserve_clearance():
    sea = SeaMap((0, 0, 24, 24), ((9, 5, 14, 18),))
    route = astar(sea, (2, 2), (22, 22))
    goals = smooth(sea, route)
    assert goals[0] == (2, 2) and goals[-1] == (22, 22)
    assert 2 < len(goals) <= len(route)
    assert all(sea.clear(a, b, .7) for a, b in zip(goals, goals[1:]))


def test_blocked_invalid_and_zero_length_routes():
    sea = SeaMap((0, 0, 10, 10), ((4, 0, 6, 10),))
    with pytest.raises(NoPath):
        astar(sea, (2, 5), (8, 5))
    with pytest.raises(NoPath):
        astar(sea, (5, 5), (8, 5))
    assert astar(sea, (2, 2), (2, 2)) == [(2, 2)]
    with pytest.raises(ValueError):
        astar(sea, (2, 2), (8, 8), resolution=0)


def test_diagonal_cannot_cut_between_touching_land():
    sea = SeaMap((0, 0, 6, 6), ((0, 3, 3, 6), (3, 0, 6, 3)))
    with pytest.raises(NoPath):
        astar(sea, (1, 1), (5, 5), clearance=.2)
```

- [ ] **Step 2: Run the tests before implementation.**

```bash
python -m pytest tests/test_planning.py -v
```

Expected: failure because the new module or behaviour does not exist. Missing environment dependencies must be resolved before treating a test failure as evidence about application behaviour.

- [ ] **Step 3: Implement the following files.**

Create `rebuild/src/shipnav/planning.py` with:

```python
from heapq import heappop, heappush
from itertools import count
from math import ceil, dist, floor, isfinite
from shipnav.maps import Point, SeaMap


class NoPath(ValueError):
    pass


def astar(sea: SeaMap, start: Point, goal: Point,
          clearance: float = .7, resolution: float = 1.0) -> list[Point]:
    start, goal = tuple(start), tuple(goal)
    if not isfinite(resolution) or resolution <= 0:
        raise ValueError('Resolution must be positive and finite')
    if not sea.clear(start, start, clearance) or not sea.clear(goal, goal, clearance):
        raise NoPath('Start or goal is blocked')
    if start == goal:
        return [start]
    if sea.clear(start, goal, clearance):
        return [start, goal]
    x0, y0, x1, y1 = sea.bounds
    nx, ny = ceil((x1-x0)/resolution), ceil((y1-y0)/resolution)
    if nx * ny > 250_000:
        raise ValueError('Grid exceeds 250000 cells; increase resolution')
    first, last = (-1, -1), (-2, -2)

    def point(node):
        if node == first:
            return start
        if node == last:
            return goal
        return (x0+(node[0]+.5)*resolution, y0+(node[1]+.5)*resolution)

    def neighbours(node):
        if node == first:
            ix = floor((start[0]-x0)/resolution)
            iy = floor((start[1]-y0)/resolution)
            offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        else:
            ix, iy = node
            offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        candidates = [(ix+dx, iy+dy) for dx, dy in offsets
                      if 0 <= ix+dx < nx and 0 <= iy+dy < ny]
        if dist(point(node), goal) <= 2*resolution:
            candidates.append(last)
        return [n for n in candidates if sea.clear(point(node), point(n), clearance)]

    serial = count()
    queue = [(dist(start, goal), next(serial), first)]
    costs, parent, closed = {first: 0.0}, {}, set()
    while queue:
        _, _, node = heappop(queue)
        if node in closed:
            continue
        if node == last:
            route = [goal]
            while node != first:
                node = parent[node]
                route.append(point(node))
            return route[::-1]
        closed.add(node)
        for other in neighbours(node):
            cost = costs[node] + dist(point(node), point(other))
            if cost < costs.get(other, float('inf')):
                costs[other], parent[other] = cost, node
                heappush(queue, (cost+dist(point(other), goal), next(serial), other))
    raise NoPath('No route at this grid resolution')


def smooth(sea: SeaMap, route: list[Point], clearance: float = .7) -> list[Point]:
    if not route:
        raise ValueError('Route is empty')
    result, i = [route[0]], 0
    while i < len(route)-1:
        j = len(route)-1
        while j > i and not sea.clear(route[i], route[j], clearance):
            j -= 1
        if j == i:
            raise NoPath('Input route contains an unsafe segment')
        result.append(route[j])
        i = j
    return result
```

- [ ] **Step 4: Run the same test command and inspect the result.**

```bash
python -m pytest tests/test_planning.py -v
```

Expected: PASS. Do not weaken assertions to conceal an algorithm or interface failure.

- [ ] **Step 5: Review the task's diff and save a checkpoint.**

From the workspace root, if it is a Git repository:

```bash
git add rebuild/src/shipnav/planning.py rebuild/tests/test_planning.py
git commit -m "feat: find routes and intermediate goals"
```

### Completion gate

- [ ] Run `python -m pytest tests/test_maps.py tests/test_planning.py -v`.
- [ ] Inspect the route numerically with this headless command:

```bash
python - <<'PYTHON'
from pathlib import Path
from shipnav.maps import SeaMap
from shipnav.planning import astar, smooth
sea = SeaMap.load(Path('maps/harbour.json'))
print(smooth(sea, astar(sea, (2, 2), (22, 22))))
PYTHON
```

Expected: start, one or more corner waypoints, final goal. Exact waypoint coordinates are not an acceptance requirement. This stage is independently usable without the GUI or learned policy. Coarse-grid failure means no route in this graph; it is not proof no continuous route exists.

### Task 3b — Compare Basic Theta* on the same map

**Files:** `src/shipnav/theta.py`, `tests/test_theta.py`

**Interfaces:** `theta_star(sea, start, goal, clearance=.7, resolution=1.) -> list[Point]`; same endpoints, grid and clearance as A*. LOS relaxation uses the current node’s parent; it is not post-smoothing.

- [ ] Write the behavior test:

File: `rebuild/tests/test_theta.py`

```python
import pytest
from math import dist
from shipnav.maps import SeaMap
from shipnav.theta import theta_star
from shipnav.planning import NoPath

def test_theta_segments_and_endpoints():
    sea = SeaMap((0, 0, 24, 24), ((9, 5, 14, 18),))
    route = theta_star(sea, (2, 2), (22, 22))
    assert route[0] == (2, 2) and route[-1] == (22, 22)
    assert all(sea.clear(a, b, .7) for a, b in zip(route, route[1:]))
    assert sum(dist(a, b) for a, b in zip(route, route[1:])) >= dist(route[0], route[-1])

def test_theta_rejects_disconnected_water():
    with pytest.raises(NoPath):
        theta_star(SeaMap((0, 0, 10, 10), ((4, 0, 6, 10),)), (2, 5), (8, 5))
```

- [ ] Run `python -m pytest tests/test_planning.py tests/test_theta.py -v`. Expect a missing module or unmet behavior, not a missing third-party dependency.
- [ ] Implement the following content:

File: `rebuild/src/shipnav/theta.py`

```python
from heapq import heappop, heappush
from itertools import count
from math import ceil, dist, floor, isfinite
from shipnav.maps import Point, SeaMap


from shipnav.planning import NoPath


def theta_star(sea: SeaMap, start: Point, goal: Point,
          clearance: float = .7, resolution: float = 1.0) -> list[Point]:
    start, goal = tuple(start), tuple(goal)
    if not isfinite(resolution) or resolution <= 0:
        raise ValueError('Resolution must be positive and finite')
    if not sea.clear(start, start, clearance) or not sea.clear(goal, goal, clearance):
        raise NoPath('Start or goal is blocked')
    if start == goal:
        return [start]
    if sea.clear(start, goal, clearance):
        return [start, goal]
    x0, y0, x1, y1 = sea.bounds
    nx, ny = ceil((x1-x0)/resolution), ceil((y1-y0)/resolution)
    if nx * ny > 250_000:
        raise ValueError('Grid exceeds 250000 cells; increase resolution')
    first, last = (-1, -1), (-2, -2)

    def point(node):
        if node == first:
            return start
        if node == last:
            return goal
        return (x0+(node[0]+.5)*resolution, y0+(node[1]+.5)*resolution)

    def neighbours(node):
        if node == first:
            ix = floor((start[0]-x0)/resolution)
            iy = floor((start[1]-y0)/resolution)
            offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        else:
            ix, iy = node
            offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        candidates = [(ix+dx, iy+dy) for dx, dy in offsets
                      if 0 <= ix+dx < nx and 0 <= iy+dy < ny]
        if dist(point(node), goal) <= 2*resolution:
            candidates.append(last)
        return [n for n in candidates if sea.clear(point(node), point(n), clearance)]

    serial = count()
    queue = [(dist(start, goal), next(serial), first)]
    costs, parent, closed = {first: 0.0}, {}, set()
    while queue:
        _, _, node = heappop(queue)
        if node in closed:
            continue
        if node == last:
            route = [goal]
            while node != first:
                node = parent[node]
                route.append(point(node))
            return route[::-1]
        closed.add(node)
        for other in neighbours(node):
            ancestor = parent.get(node, node)
            via = ancestor if sea.clear(point(ancestor), point(other), clearance) else node
            cost = costs[via] + dist(point(via), point(other))
            if cost < costs.get(other, float('inf')):
                costs[other], parent[other] = cost, via
                closed.discard(other)
                heappush(queue, (cost+dist(point(other), goal), next(serial), other))
    raise NoPath('No route at this grid resolution')
```

- [ ] Run `python -m pytest tests/test_planning.py tests/test_theta.py -v` again. Inspect failures before proceeding.
- [ ] Review and commit only the implementation/test files named above after the check passes; keep `docs/` ignored.

- [ ] Benchmark both planners on the same frozen maps/endpoints with `perf_counter_ns`; save resolution, clearance, expanded nodes (add a counter to each search), route length and minimum sampled body-to-land distance. Validate every segment continuously with `SeaMap.clear`. Do not assume Theta* always beats smoothed A* or that coarse-grid failure proves continuous-space infeasibility. The service's `planner` field records `astar_smooth` or `theta`.
