# 04 — GUI, exports and evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide the interactive maritime-navigation demo and reproducible experiment outputs, and record the pretrained/classical baselines that plan 05 retraining is compared against.

**Architecture:** Use the same headless service for CLI, GUI and evaluation. Execute simulations in a worker thread with cooperative cancellation; display and replay results on the Qt main thread, then export the same recorded trace.

**Tech Stack:** PySide6, Matplotlib QtAgg, Shapely, FFmpeg, Python concurrent.futures, pytest-qt, standard-library CSV/JSON

**Spec:** [../specs/2026-09-27-solo-rebuild-design.md](../specs/2026-09-27-solo-rebuild-design.md)

## Global Constraints

- Never use `codex` in branch names or worktree directory names.
- Preserve `CrowdNav-20250813-DIP/` and the supplied briefing unchanged.
- Implement only inside `rebuild/`; keep planning documents inside `docs/superpowers/` (now tracked in Git).
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

All file paths below are relative to the workspace root. Run commands from `rebuild/` unless explicitly stated otherwise. A code step can be worked through function by function in 2–5 minute increments; commit only after the task's tests pass. Commit only explicitly named implementation files.

## Revision 2026-09-28 — aligned with the implemented 02/03/03b code

The first version of this plan was written before plans 02–03b landed. Its snippets no longer matched the code, and several were wrong in ways the tests exposed. This revision rewrites the code blocks against the current `execute()` schema-2 result and polygon `SeaMap`.

**Verification of this revision.** Every code block and diff below was applied to a scratch copy of `rebuild/` and run: **all 26 new tests pass**, and the full suite gives 208 passed. The 14 remaining failures need torch or the RVO2 native wheel, which were not installed; they fail identically on the unmodified code. The Task 6 changes to existing modules were checked for behavioural parity: all 31 frozen scenarios × {direct, MPC} × {filtered, unfiltered} × {holonomic, marine} (248 runs) produce identical statuses, frames, routes and filter decisions before and after. That run used Python 3.11 with current Shapely/Matplotlib/PySide6/pyogrio wheels and FFmpeg, **not** the locked 3.14 `.venv-modern`, and did not load the SARL checkpoint (it is not in the cloud checkout). Rerun every step in `.venv-modern`; these results are not a substitute for it.

What changed and why:

| Area | Old plan | Now |
|---|---|---|
| Land drawing (Task 7) | Read `result['map']['land']` as rectangles; KeyError on schema-1 maps | Draws polygon `features`, including holes and MultiPolygons |
| Diagnostics overlay and per-step CSV | Deferred to Task 10 | In Task 7: `execute()` already records `diagnostics` |
| GUI planner/dynamics/filter/perception controls | Deferred to Task 10 | In Task 8: `execute()` already accepts these keywords |
| Stop button (Task 8) | `clicked.connect(self.cancel_event.set)` bound the *first* Event; `start_run` replaces it, so **Stop never cancelled a run** | Connects through a lambda; a test fails on the old wiring |
| Land editing (Task 8) | Test compared `sea.land` with a tuple; the rebuilt map dropped metadata | Compares Shapely geometry; keeps `metadata` |
| Map scale | Assumed the 24 m harbour everywhere; `execute()` plans on a fixed 1 m grid with a 1 m/s, 100 s episode, which cannot cross a 12 × 8 km map | **Decision (user, 2026-09-28): keep the real Singapore maps.** New Task 6 runs them through the model frame by geometric similarity (vessel profiles), with scale-aware planning, route-timed traffic and a faster spatial index |
| Default endpoints (Task 8) | Fixed `(2,2)`/`(22,22)` | 2 model units inside the map corners if clear, else unset and Run asks for them |
| Default filter (Task 8) | Unfiltered | Predictive filter. The supplied SARL never observed land or map edges: unfiltered on `harbour.json` with 5 ships it hit "land" at 4.5 s. From `(2,2)` the island is ≥7 m away, so that was most likely the map edge (which `SeaMap.clear` treats as land). |
| `metrics()` (Task 11) | `sum(d['no_feasible_action'])` crashed on every unfiltered run (`None` = not evaluated) and aborted the benchmark | Reports `None` for unfiltered runs |
| `land_clearance` (Task 11) | Unpacked polygons as rectangles | Signed polygon distance (negative inside land) |
| `geometry_metrics` (Task 11) | Assumed marine runs start at heading 0; a run whose first leg is not due east counted a false heading violation | Starts on the first route leg's bearing, matching `run_episode`; returns `{}` for runs with no frames |
| Benchmark rows | Only `metrics()` | Merges `geometry_metrics()`; adds `orca_filtered` so filter claims have a matched pair |

## File structure and prerequisites

Create `rebuild/src/shipnav/scale.py`, `export.py`, `gui.py`, `evaluate.py`, `metrics.py`, `benchmark.py`; create `rebuild/tests/test_scale.py`, `test_export.py`, `test_gui.py`, `test_evaluate.py`, `test_metrics.py`, `test_geometric_metrics.py`; modify `rebuild/src/shipnav/maps.py`, `safety.py`, `controllers/mpc.py`, `scenarios.py`, `service.py` (Task 6); extend `rebuild/README.md`.

Plans 01b, 02, 03 and 03b are complete (see `git log`). Run all commands in `.venv-modern` with `uv sync --locked --all-extras` (the `gui` extra supplies PySide6; the `dev` group supplies pytest-qt). FFmpeg is a system binary (`brew install ffmpeg`). Run Qt tests with `QT_QPA_PLATFORM=offscreen`, then perform the native-window acceptance checklist. Headless Linux needs the system EGL/GL/xkbcommon/fontconfig libraries for PySide6 to import.

The GUI replays a completed simulation. It shows a running status during inference, remains responsive, and allows cancellation. Live streaming during inference is a possible extension, not needed to meet the briefing's visualization requirement.

### Task 6: Real-map support through the model frame

**Files:** create `rebuild/src/shipnav/scale.py`, `rebuild/tests/test_scale.py`; modify `rebuild/src/shipnav/maps.py`, `safety.py`, `controllers/mpc.py`, `scenarios.py`, `service.py`.

**Problem.** Every controller, filter, predictor and dynamics limit is tuned for the canonical ego: radius .5, speed 1, dt .25. Those are the supplied SARL's training units, and SARL's inputs are raw distances and speeds, so a 5 m vessel fed to it directly is out of distribution. The Singapore maps are 12 × 8 km (Ubin) in metres. `execute()` also planned on a fixed 1 m grid (100 M cells against a 250 k budget), capped episodes at 100 s, and scattered traffic uniformly, so ships never met the ego.

**Solution: geometric similarity.** A vessel profile with radius *r* m and speed *V* m/s maps onto the canonical ego with length scale *L = 2r* and time scale *L/V*. `scale.to_model` divides the map by *L* and records `model_scale` in its metadata. The unchanged service then runs in model units, and presentation (Tasks 7–8) converts back to metres and seconds.

- Collision geometry is exact under scaling. Everything tuned for the canonical ego keeps working, and SARL sees in-distribution inputs.
- The implied real parameters follow from the profile: dt = .25 *L/V* s; traffic .3–1 model units/s = .3*V*–*V* m/s; marine limits .2 *V²/L* m/s² and .35 *V/L* rad/s. For `harbour_craft` (5 m radius, 5 m/s, *L* = 10 m) that is dt .5 s, traffic 1.5–5 m/s, .5 m/s² and .175 rad/s. These are demo similarity values, not calibrated ship data.
- Synthetic maps use the identity `model` profile and are left byte-identical, so every existing scenario, hash and test is unaffected.
- Results and JSON exports stay in model units (the exact reproducibility artifact); `model_scale` in the stored map says how to convert.

| Profile | Radius | Speed | Margin | Planner grid | Use |
|---|---|---|---|---|---|
| `model` | .5 m | 1 m/s | .2 m | 1 m | Synthetic maps up to 250,000 m² (default there) |
| `harbour_craft` | 5 m | 5 m/s | 10 m | 50 m | Default for larger maps; matches the `map_demo` route settings |
| `coastal_ship` | 25 m | 7.5 m/s | 25 m | 100 m | Coarser, larger vessel |

The service gains keywords, all defaulting to the old behaviour:
- `resolution` and `clearance`: planner grid and land clearance in model units. `Profile.service_options()` supplies them; `scale.map_options(map)` recovers them from a frozen scenario's map.
- `limit=None`: the larger of 100 s and three times the nominal route time.
- `placement='uniform'|'corridor'`: corridor ships are scripted `course_change` voyages timed so they reach a point on the A*-smoothed reference route when a nominal 1 m/s ego does. The mix is crossing (60–120°), head-on and overtaking. Ships wait at their start before departing, and the timing uses the same reference route whichever planner runs, so planner comparisons face identical traffic.

**Speed.** On the scaled Ubin map, a filtered run took 229 s. Two exact optimisations bring it to 16 s:
- `SeaMap` indexes land split into about 48 tiles across the map. `land` stays the merged canonical geometry, so `clear`/`minimum_clearance` answers, map hashes and `to_dict` bytes are unchanged; a test compares 300 random queries with Shapely directly.
- The filter and MPC skip land checks when the nearest land is beyond the distance any speed-bounded rollout can travel within the horizon.

Measured on the scaled Ubin map with the harbour-craft profile, the 10.2 km route (same length as `map_demo`'s) and 5 corridor ships, seed 0:

| Run | Outcome | Simulated time | Closest ship | Wall time | JSON |
|---|---|---|---|---|---|
| Direct, unfiltered | collision | 595 s | −1.9 m | 2 s | 7 MB |
| Direct, predictive filter | success | 2090 s | 2.3 m | 16 s | 25 MB |
| MPC, marine, filtered | success | 2095 s | 5.1 m | 77 s | 36 MB |

The direct collision is expected: it has no avoidance, and it shows that corridor traffic produces real encounters. Uniform traffic kept every ship at least 410 m away. Marine MPC remains the slowest (Python rollouts of 53 candidates × 21 steps); it runs in the GUI's worker thread.

- [ ] **Step 1: Write the behaviour tests.**

Create `rebuild/tests/test_scale.py` with:

```python
import csv
import json
from math import dist
from random import Random
import pytest
from shapely import affinity
from shapely.geometry import LineString, Point, box
from shipnav.maps import SeaMap
from shipnav.scale import PROFILES, Profile, check_grid, scaled, to_model, units
from shipnav.scenarios import load_traffic
from shipnav.service import execute
from shipnav.export import export_run, video_frames

HARBOUR = SeaMap.load('maps/harbour.json')


def enlarged(sea, k):
    return SeaMap(tuple(v*k for v in sea.bounds), [affinity.scale(p, k, k, origin=(0, 0)) for p in sea.land],
                  sea.to_dict()['metadata'])


def test_a_ten_times_larger_harbour_runs_identically_in_model_units(tmp_path):
    profile = Profile('ten', 5., 1., 2., 10.)   # L = 10 m, time scale 10 s
    real = enlarged(HARBOUR, 10)
    model = to_model(real, profile)
    assert model.bounds == HARBOUR.bounds and units(model.to_dict()) == (10., 10.)
    options = dict(policy_name='direct', seed=3, count=4, filtered=True, **profile.service_options())
    a = execute(HARBOUR.to_dict(), (2, 2), (22, 22), **options)
    b = execute(model.to_dict(), (2, 2), (22, 22), **options)
    assert a['status'] == b['status'] and a['traffic_definitions'] == b['traffic_definitions']
    for f, g in zip(a['frames'], b['frames'], strict=True):
        assert f['t'] == g['t'] and f['position'] == pytest.approx(g['position'], abs=1e-9)
    export_run(b, tmp_path/'b.csv')
    with (tmp_path/'b.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[-1]['time_s']) == pytest.approx(10*b['elapsed'])
    assert float(rows[-1]['x_m']) == pytest.approx(10*b['frames'][-1]['position'][0])
    assert float(rows[0]['executed_vx']) == pytest.approx(b['diagnostics'][0]['executed'][0])  # V = 1 m/s


def test_scaling_is_explicit_and_not_repeated():
    assert scaled(HARBOUR, PROFILES['model']) is HARBOUR
    model = to_model(enlarged(HARBOUR, 10), PROFILES['harbour_craft'])
    with pytest.raises(ValueError):
        to_model(model, PROFILES['harbour_craft'])
    with pytest.raises(ValueError):
        check_grid(SeaMap.load('maps/singapore-ubin.json'), PROFILES['model'])
    check_grid(SeaMap.load('maps/singapore-ubin.json'), PROFILES['harbour_craft'])


def test_tiled_index_answers_exactly_like_the_merged_coastline():
    sea = to_model(SeaMap.load('maps/singapore-ubin.json'), PROFILES['harbour_craft'])
    land = sea.land
    x0, y0, x1, y1 = sea.bounds
    rng = Random(0)
    for _ in range(300):
        a = (rng.uniform(x0, x1), rng.uniform(y0, y1))
        b = (a[0]+rng.uniform(-30, 30), a[1]+rng.uniform(-30, 30))
        segment = LineString([a, b])
        inside = all(x0+1.5 < x < x1-1.5 and y0+1.5 < y < y1-1.5 for x, y in (a, b))
        assert sea.clear(a, b, 1.5) == (inside and not any(p.dwithin(segment, 1.5) for p in land))
        edge = min(min(x-x0, x1-x, y-y0, y1-y) for x, y in (a, b))
        expected = 0. if edge <= 0 else min([edge] + [p.distance(segment) for p in land])
        assert sea.minimum_clearance(a, b) == pytest.approx(expected, abs=1e-9)


def test_corridor_traffic_meets_the_reference_route_on_time_for_every_planner():
    sea = to_model(SeaMap.load('maps/singapore-ubin.json'), PROFILES['harbour_craft'])
    start, goal = (-444.98, 138.39), (445.17, 303.81)
    options = dict(policy_name='direct', seed=0, count=5, placement='corridor', limit=1.,
                   **PROFILES['harbour_craft'].service_options())
    a = execute(sea.to_dict(), start, goal, planner='astar_smooth', **options)
    b = execute(sea.to_dict(), start, goal, planner='theta', **options)
    assert a['traffic_definitions'] == b['traffic_definitions'] and len(a['traffic_definitions']) == 5
    line = LineString(a['route'])
    for ship in load_traffic(a['scenario']):
        # Some waypoint lies on the reference route and is reached exactly when a 1 m/s
        # ego following it arrives there.
        assert any(line.distance(Point(p)) < 1e-6 and abs(t-line.project(Point(p))) < 1e-6
                   for t, p in ship.waypoints)
        assert sea.clear(ship.start, ship.goal, ship.radius)


def test_video_speedup_keeps_terminal_frame_and_caps_fps():
    result = {'frames': [{}]*4191, 'settings': {'dt': .25}, 'elapsed': 1047.5,
              'map': {'metadata': {'model_scale': {'length_m': 10., 'speed_mps': 5.}}}}
    indices, fps = video_frames(result, speedup=18.)
    assert indices[0] == 0 and indices[-1] == 4190 and fps <= 30
    assert len(indices)/fps == pytest.approx(1047.5*2/18, rel=.01)


def test_cli_runs_a_real_map_in_metres(tmp_path):
    import subprocess, sys
    out = tmp_path/'ubin.json'
    done = subprocess.run([sys.executable, '-m', 'shipnav.service', '--map', 'maps/singapore-ubin.json',
                           '--policy', 'direct', '--count', '0', '--start', '-2000', '500', '--goal', '-2500', '500',
                           '--output', str(out)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    result = json.loads(out.read_text())
    assert result['status'] == 'success' and result['settings']['placement'] == 'corridor'
    assert units(result['map']) == (10., 2.)
    assert result['route'][0] == [-200., 50.]


def test_frozen_real_map_scenario_reuses_its_recorded_planner_settings(tmp_path):
    from shipnav.benchmark import benchmark
    from shipnav.scenarios import make_scenario
    sea = to_model(SeaMap.load('maps/singapore-ubin.json'), PROFILES['harbour_craft'])
    scene = make_scenario(sea, (-200, 50), (-250, 50), 0, 0)
    rows = benchmark([scene], '', tmp_path, {'direct': dict(policy_name='direct')})
    run = json.loads(next(tmp_path.glob('*-direct.json')).read_text())
    assert rows[0]['status'] == 'success'
    assert run['settings']['resolution'] == 5. and run['settings']['clearance'] == 1.5


def test_open_water_shortcut_bounds_reach_by_the_fastest_candidate():
    from shipnav import safety
    calls = []
    real = safety.local_sea
    safety.local_sea = lambda sea, p, reach, radius: calls.append(reach) or real(sea, p, reach, radius)
    try:
        safety.choose(SeaMap((0, 0, 24, 24)), (12, 12), (1+1e-6, 0), [], .5, 1., .25)
    finally:
        safety.local_sea = real
    assert calls == [pytest.approx((1+1e-6)*.25*12, abs=0)]
```

- [ ] **Step 2: Run `python -m pytest tests/test_scale.py -v`.** Expect a missing `shipnav.scale` module.

- [ ] **Step 3: Record parity before touching existing modules.** Save statuses, frames, routes and filter decisions for every frozen scenario × {direct, mpc} × {filtered, unfiltered} × {holonomic, marine} to `results/parity-before.json`. (Add sarl/orca in `.venv-modern`, where the checkpoint and RVO2 are available.)

- [ ] **Step 4: Implement.**

Create `rebuild/src/shipnav/scale.py` with:

```python
"""Run real-world maps through the canonical model frame by geometric similarity.

Every controller, filter, prediction and dynamics setting in ShipNav is tuned for the
canonical ego: radius .5, speed 1, dt .25 (the supplied SARL's training units). A real
vessel profile with radius r metres and speed V m/s maps onto it with length scale
L = 2r and time scale L/V. Dividing map coordinates by L lets the unchanged service run
the vessel, and results convert back to metres, m/s and seconds for display and export.

Consequences recorded in the map metadata as `model_scale`, stated here once:
dt = .25 L/V seconds; traffic speed .3-1 model units = .3V-V m/s; marine limits
.2 V²/L m/s² and .35 V/L rad/s; the planner grid is `resolution_m`.
"""
from dataclasses import dataclass
from math import isfinite
from shapely import affinity
from shapely.geometry import box
from shipnav.maps import SeaMap

INTERACTIVE_AREA_M2 = 250_000  # the canonical 1 m planner's max_cells


@dataclass(frozen=True)
class Profile:
    name: str
    radius_m: float
    speed_mps: float
    margin_m: float
    resolution_m: float

    def __post_init__(self):
        if not all(isfinite(v) and v > 0 for v in (self.radius_m, self.speed_mps, self.resolution_m)) \
                or not isfinite(self.margin_m) or self.margin_m < 0:
            raise ValueError('Profile needs positive radius, speed, resolution and nonnegative margin')

    @property
    def length(self):
        return 2*self.radius_m

    @property
    def speed(self):
        return self.speed_mps

    def service_options(self):
        """Planner settings in model units for `execute(**options)`."""
        return {'resolution': self.resolution_m/self.length,
                'clearance': (self.radius_m+self.margin_m)/self.length}


# 'model' is the identity used by synthetic maps and every existing test. The harbour
# craft matches the offline map_demo route settings (5 m radius, 10 m margin, 50 m grid).
PROFILES = {'model': Profile('model', .5, 1., .2, 1.),
            'harbour_craft': Profile('harbour_craft', 5., 5., 10., 50.),
            'coastal_ship': Profile('coastal_ship', 25., 7.5, 25., 100.)}


def default_profile(sea: SeaMap) -> Profile:
    x0, y0, x1, y1 = sea.bounds
    return PROFILES['model'] if (x1-x0)*(y1-y0) <= INTERACTIVE_AREA_M2 else PROFILES['harbour_craft']


def to_model(sea: SeaMap, profile: Profile) -> SeaMap:
    """Scale a metric map into model units. Idempotent guard: refuses an already scaled map."""
    metadata = sea.to_dict()['metadata']
    if 'model_scale' in metadata:
        raise ValueError('Map is already in model units')
    L = profile.length
    bounds = tuple(v/L for v in sea.bounds)
    crop = box(*bounds)
    # Scaling can move land that touches the crop edge outside it by rounding; clip it back.
    land = [affinity.scale(p, 1/L, 1/L, origin=(0, 0)).intersection(crop) for p in sea.land]
    metadata = {**metadata, 'model_scale': {'profile': profile.name, 'length_m': L,
                                            'speed_mps': profile.speed_mps,
                                            'radius_m': profile.radius_m, 'margin_m': profile.margin_m,
                                            'resolution_m': profile.resolution_m}}
    return SeaMap(bounds, [p for p in land if not p.is_empty], metadata)


def units(map_data: dict) -> tuple[float, float]:
    """(metres per model unit, seconds per model second) for a map dict; (1, 1) if unscaled."""
    scale = map_data['metadata'].get('model_scale')
    if scale is None:
        return 1., 1.
    return scale['length_m'], scale['length_m']/scale['speed_mps']


def scaled(sea: SeaMap, profile: Profile) -> SeaMap:
    """`to_model`, except the canonical profile leaves synthetic maps byte-identical."""
    return sea if profile.name == 'model' else to_model(sea, profile)


def check_grid(sea: SeaMap, profile: Profile, max_cells: int = 250_000) -> None:
    """Reject a map/profile pair whose planning grid exceeds the planner's cell budget."""
    x0, y0, x1, y1 = sea.bounds
    cells = ((x1-x0)/profile.resolution_m)*((y1-y0)/profile.resolution_m)
    if cells > max_cells:
        raise ValueError(f'{profile.name} plans on a {profile.resolution_m:g} m grid: {cells:,.0f} cells '
                         f'exceed {max_cells:,}. Choose a larger vessel profile')


def map_options(map_data: dict) -> dict:
    """Planner settings recorded in a scaled map (e.g. a frozen real-map scenario), else {}."""
    scale = map_data['metadata'].get('model_scale')
    if scale is None:
        return {}
    L = scale['length_m']
    return {'resolution': scale['resolution_m']/L, 'clearance': (scale['radius_m']+scale['margin_m'])/L}
```

Apply these changes to existing modules (`git apply` from the workspace root accepts each block):

`rebuild/src/shipnav/maps.py` — tiled spatial index:

```diff
--- a/rebuild/src/shipnav/maps.py
+++ b/rebuild/src/shipnav/maps.py
@@ -1,7 +1,7 @@
 """Metric polygon maps. Vector distance is authoritative; unknown depth is blocked."""
 from dataclasses import dataclass, field
 import json
-from math import floor, isfinite
+from math import ceil, floor, isfinite
 from pathlib import Path
 
 from pyproj import CRS, Transformer
@@ -53,12 +53,36 @@
         return {'origin_lonlat': [self.lon, self.lat], 'epsg': self.epsg, 'units': 'metres', 'axes': 'east,north'}
 
 
+TILES_ACROSS = 48
+
+
+def tile(polygons, size):
+    """Split land into grid-aligned pieces for the spatial index only.
+
+    `land` stays the canonical merged geometry. The union of the pieces equals it, so
+    `clear` and `minimum_clearance` are unchanged, but each query now tests a few small
+    pieces instead of a coastline with thousands of vertices."""
+    pieces = []
+    for polygon in polygons:
+        a, b, c, d = polygon.bounds
+        if c-a <= size and d-b <= size:
+            pieces.append(polygon)
+            continue
+        for i in range(floor(a/size), ceil(c/size)):
+            for j in range(floor(b/size), ceil(d/size)):
+                part = polygon.intersection(box(i*size, j*size, (i+1)*size, (j+1)*size))
+                pieces.extend(q for q in getattr(part, 'geoms', [part])
+                              if isinstance(q, Polygon) and not q.is_empty)
+    return tuple(pieces)
+
+
 @dataclass(frozen=True, init=False)
 class SeaMap:
     bounds: tuple
     land: tuple
     _metadata: str = field(repr=False)
     _tree: STRtree = field(repr=False, compare=False)
+    _pieces: tuple = field(repr=False, compare=False)
 
     def __init__(self, bounds, land=(), metadata=None):
         bounds = tuple(float(v) for v in bounds)
@@ -84,7 +108,11 @@
         parts.sort(key=lambda p: p.wkb_hex)
         object.__setattr__(self, 'bounds', bounds)
         object.__setattr__(self, 'land', tuple(parts))
-        object.__setattr__(self, '_tree', STRtree(parts))
+        # About 48 tiles across the longer side (at least 1 unit): ~25 units on the scaled
+        # Ubin map. Bounded tile count keeps loading fast at any coordinate scale.
+        pieces = tile(parts, max(1., max(bounds[2]-bounds[0], bounds[3]-bounds[1])/TILES_ACROSS))
+        object.__setattr__(self, '_pieces', pieces)
+        object.__setattr__(self, '_tree', STRtree(pieces))
         object.__setattr__(self, '_metadata', canonical_json(metadata or {}))
 
     @staticmethod
@@ -101,7 +129,7 @@
             return 0.
         segment = self._segment(a, b)
         nearest = self._tree.nearest(segment)
-        return min(edge, float(segment.distance(self.land[nearest]))) if nearest is not None else edge
+        return min(edge, float(segment.distance(self._pieces[nearest]))) if nearest is not None else edge
 
     def clear(self, a, b, clearance=0., *, mode='coastline'):
         if not isfinite(clearance) or clearance < 0:
```

`rebuild/src/shipnav/safety.py` — exact open-water shortcut:

```diff
--- a/rebuild/src/shipnav/safety.py
+++ b/rebuild/src/shipnav/safety.py
@@ -10,6 +10,21 @@
     return hypot(x+f*u, y+f*v)-radius
 
 
+class _OpenWater:
+    def clear(self, *args, **kwargs):
+        return True
+
+
+_OPEN_WATER = _OpenWater()
+
+
+def local_sea(sea, position, reach, radius):
+    """Exact shortcut for rollouts from `position`: every candidate is speed-bounded, so
+    no rollout moves the disk further than `reach`. When the nearest land or map edge is
+    beyond reach+radius no rollout can touch it, and land checks are skipped."""
+    return _OPEN_WATER if sea.minimum_clearance(position, position) > reach+radius+1e-9 else sea
+
+
 def rollout(position, desired, dt, steps, motion=None, goal=None, arrival=0.):
     """Roll a constant command out `steps` steps (bounded `motion` if given).
 
@@ -53,6 +68,9 @@
     if steps is None:
         steps = horizon_steps(dt, 'holonomic' if motion is None else 'marine', speed)
     hold = goal if final else None
+    # run_episode accepts a nominal up to speed+1e-6, and holonomic rollouts move at the
+    # command's own speed, so bound reach by the fastest candidate actually tried.
+    sea = local_sea(sea, position, max(speed, hypot(*nominal))*dt*steps, radius)
     candidates = [tuple(nominal), (0., 0.)]
     candidates += [(s*cos(k*pi/8), s*sin(k*pi/8)) for s in (speed*.5, speed) for k in range(16)]
     scored = []
```

`rebuild/src/shipnav/controllers/mpc.py` — same shortcut for MPC rollouts:

```diff
--- a/rebuild/src/shipnav/controllers/mpc.py
+++ b/rebuild/src/shipnav/controllers/mpc.py
@@ -29,7 +29,7 @@
 
 from shipnav.dynamics import motion_from
 from shipnav.horizon import horizon_steps
-from shipnav.safety import assess, rollout
+from shipnav.safety import assess, local_sea, rollout
 
 TIME_WEIGHT = .05
 EFFORT_WEIGHT = .1
@@ -78,10 +78,11 @@
         motion = motion_from(self.vessel, speed) if self.vessel is not None else None
         steps = self.steps or horizon_steps(dt, 'holonomic' if self.vessel is None else 'marine', speed)
         hold = goal if self.final else None
+        sea = local_sea(self.sea, p, speed*dt*steps, radius)
         ranked = []
         for command in candidates(p, goal, speed, dt):
             path = rollout(p, command, dt, steps, motion, hold, radius)
-            clearance = assess(self.sea, path, self.predictions, radius)
+            clearance = assess(sea, path, self.predictions, radius)
             cost = progress_cost(path, goal, radius, dt) + EFFORT_WEIGHT*dist(command, v)
             ranked.append((clearance, cost, command))
         feasible = [r for r in ranked if r[0] > 0]
```

`rebuild/src/shipnav/scenarios.py` — corridor traffic:

```diff
--- a/rebuild/src/shipnav/scenarios.py
+++ b/rebuild/src/shipnav/scenarios.py
@@ -1,7 +1,7 @@
 from dataclasses import asdict
 from hashlib import sha256
 from random import Random
-from math import dist, isfinite
+from math import atan2, cos, dist, isfinite, pi, sin
 import json
 from shipnav.simulation import CourseChangeTraffic, Traffic
 
@@ -94,9 +94,16 @@
     return sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
 
 
-def make_scenario(sea, start, goal, count, seed):
+def make_scenario(sea, start, goal, count, seed, *, corridor=None):
+    """Random traffic. With `corridor` (a reference route) ships are timed to meet the
+    nominal 1 m/s ego along it (see `corridor_traffic`); otherwise uniform over the map."""
     if count < 0:
         raise ValueError('Traffic count cannot be negative')
+    if corridor is not None:
+        ships = corridor_traffic(sea, corridor, count, Random(seed))
+        return {'schema': 1, 'seed': seed, 'map': sea.to_dict(), 'start': list(start),
+                'goal': list(goal), 'traffic': [traffic_to_dict(s) for s in ships],
+                'traffic_mode': 'scripted', 'family': 'corridor', 'split': 'smoke'}
     rng, ships = Random(seed), []
     x0, y0, x1, y1 = sea.bounds
     for _ in range(count):
@@ -117,6 +124,60 @@
             'traffic_mode': 'scripted', 'family': 'random', 'split': 'smoke'}
 
 
+def _along(route, s):
+    """Point and unit tangent at arc length `s` of a polyline."""
+    for a, b in zip(route, route[1:]):
+        length = dist(a, b)
+        if s <= length or b == route[-1]:
+            f = min(1., s/length) if length else 0.
+            return (a[0]+f*(b[0]-a[0]), a[1]+f*(b[1]-a[1])), ((b[0]-a[0])/length, (b[1]-a[1])/length)
+        s -= length
+    raise ValueError('Route needs at least two distinct points')
+
+
+def corridor_traffic(sea, route, count, rng, radius=.6, speeds=(.3, 1.), run_up=(3., 15.)):
+    """Scripted ships that reach a point of the reference route when a nominal 1 m/s ego
+    does. Each waits at its start, departs `run_up` model units before the meeting point
+    and continues the same distance beyond it, so encounters happen on maps far larger
+    than the ships' voyages. Kinds: crossing (±60-120°), head-on and overtaking.
+    The reference route is frozen into the timing, not into the scenario, so runs with
+    another planner or controller face identical traffic."""
+    total = sum(dist(a, b) for a, b in zip(route, route[1:]))
+    if total <= 0:
+        return []
+    ships = []
+    for _ in range(count):
+        for attempt in range(500):
+            arc = rng.uniform(.1, .95)*total
+            point, (tx, ty) = _along(route, arc)
+            heading = atan2(ty, tx)
+            kind = rng.choice(('crossing', 'crossing', 'head_on', 'overtake'))
+            speed = rng.uniform(*speeds)
+            if kind == 'crossing':
+                heading += rng.choice((1, -1))*rng.uniform(pi/3, 2*pi/3)
+            elif kind == 'head_on':
+                heading += pi + rng.uniform(-pi/12, pi/12)
+            else:
+                speed = rng.uniform(speeds[0], min(speeds[1], .6))
+            d = (cos(heading), sin(heading))
+            run = rng.uniform(*run_up)
+            meet = arc  # nominal ego arrival time at this arc length (1 model unit/s)
+            depart = max(0., meet-run/speed)
+            a = tuple(p-c*speed*(meet-depart) for p, c in zip(point, d))
+            b = tuple(p+c*run for p, c in zip(point, d))
+            if not sea.clear(a, b, radius) or dist(a, route[0]) <= 1.3 or dist(a, route[-1]) <= 1.3:
+                continue
+            if any(dist(a, s.start) <= 1.4 for s in ships):
+                continue
+            waypoints = ((0., a),) + (((depart, a),) if depart > 0 else ()) + \
+                        ((meet, point), (meet+run/speed, b))
+            ships.append(CourseChangeTraffic(waypoints, radius))
+            break
+        else:
+            raise ValueError('Scenario placement failed; preserve this failed seed')
+    return ships
+
+
 def traffic_to_dict(ship):
     """Serialise a Traffic-like object into a scenario JSON traffic entry.
     Plain fixed-voyage `Traffic` uses its dataclass fields as-is; scripted
```

`rebuild/src/shipnav/service.py` — planner resolution/clearance, route-scaled limit, placement, and `--profile`/`--placement` on the CLI (start and goal stay in metres):

```diff
--- a/rebuild/src/shipnav/service.py
+++ b/rebuild/src/shipnav/service.py
@@ -1,4 +1,5 @@
 from hashlib import sha256
+from math import dist
 from pathlib import Path
 import json
 from shipnav.maps import SeaMap, canonical_json
@@ -16,21 +17,29 @@
 def execute(map_data: dict, start: tuple, goal: tuple, model_dir: str = '',
             policy_name: str = 'sarl', global_goals: bool = True, seed: int = 0,
             count: int = 5, cancel=lambda: False, *, scenario=None, planner='astar_smooth',
-            filtered=False, uncertainty=True, observation=None, dynamics='holonomic') -> dict:
+            filtered=False, uncertainty=True, observation=None, dynamics='holonomic',
+            resolution=1., clearance=.7, limit=None, placement='uniform') -> dict:
     if scenario is not None:
         validate_scenario(scenario)
     validate_observation(observation or {})
     sea = SeaMap.from_dict(map_data)
-    scenario = make_scenario(sea,start,goal,count,seed) if scenario is None else scenario
-    if (canonical_json(SeaMap.from_dict(scenario['map']).to_dict()) != canonical_json(sea.to_dict())
-            or tuple(scenario['start'])!=tuple(start) or tuple(scenario['goal'])!=tuple(goal)):
-        raise ValueError('Scenario map/endpoints differ from run request')
+    if placement not in ('uniform', 'corridor'):
+        raise ValueError('Placement must be uniform or corridor')
     if planner == 'astar_smooth':
-        route = smooth(sea,astar(sea,tuple(start),tuple(goal)))
+        route = smooth(sea,astar(sea,tuple(start),tuple(goal),clearance,resolution),clearance)
     elif planner == 'theta':
-        route = theta_star(sea,tuple(start),tuple(goal))
+        route = theta_star(sea,tuple(start),tuple(goal),clearance,resolution)
     else:
         raise ValueError('Unknown planner')
+    if scenario is None:
+        # Corridor traffic is timed on the A*-smoothed reference route whichever planner
+        # runs, so planner comparisons share identical traffic.
+        reference = None if placement == 'uniform' else route if planner == 'astar_smooth' else \
+            smooth(sea,astar(sea,tuple(start),tuple(goal),clearance,resolution),clearance)
+        scenario = make_scenario(sea,start,goal,count,seed,corridor=reference)
+    if (canonical_json(SeaMap.from_dict(scenario['map']).to_dict()) != canonical_json(sea.to_dict())
+            or tuple(scenario['start'])!=tuple(start) or tuple(scenario['goal'])!=tuple(goal)):
+        raise ValueError('Scenario map/endpoints differ from run request')
     traffic = load_traffic(scenario)
     if scenario.get('traffic_mode')=='reactive':
         from shipnav.reactive import ReactiveTraffic
@@ -55,10 +64,14 @@
     else:
         raise ValueError('Policy must be sarl, orca, mpc or direct')
     observer = Observer(seed=scenario['seed'],**(observation or {}))
-    result = run_episode(sea,goals,traffic,policy,cancel=cancel,observer=observer,filtered=filtered,uncertainty=uncertainty,dynamics=dynamics)
+    # Default limit: 100 s, or three times the nominal route time on long real-map routes.
+    length = sum(dist(a, b) for a, b in zip(route, route[1:]))
+    limit = max(100., 3*length) if limit is None else limit
+    result = run_episode(sea,goals,traffic,policy,limit=limit,cancel=cancel,observer=observer,filtered=filtered,uncertainty=uncertainty,dynamics=dynamics)
     settings = {'seed': scenario['seed'], 'planner': planner, 'filtered': filtered, 'uncertainty': uncertainty, 'observation': observation or {}, 'dynamics': dynamics, 'requested_traffic': count,
                 'actual_traffic': len(traffic), 'policy': policy_name,
-                'global_goals': global_goals, 'dt': .25, 'limit': 100,
+                'global_goals': global_goals, 'dt': .25, 'limit': limit,
+                'resolution': resolution, 'clearance': clearance, 'placement': placement,
                 'radius': .5, 'speed': 1.0, 'query_env': False,
                 'traffic_model': scenario['traffic_mode']}
     result.update({'schema': 2, 'scenario': scenario, 'scenario_hash': scenario_hash(scenario), 'map': map_data, 'route': route, 'goals': goals,
@@ -85,17 +98,37 @@
     parser.add_argument('--planner', choices=['astar_smooth','theta'], default='astar_smooth')
     parser.add_argument('--filtered', action='store_true')
     parser.add_argument('--dynamics', choices=['holonomic','marine'], default='holonomic')
+    parser.add_argument('--profile', help='Vessel profile for a metric map (default: model for maps up to '
+                        '250,000 m², harbour_craft above). --start/--goal stay in metres')
+    parser.add_argument('--placement', choices=['uniform','corridor'],
+                        help='Traffic placement (default: uniform for model profile, corridor otherwise)')
     args = parser.parse_args()
     try:
         # json.JSONDecodeError is a ValueError; schema problems surface as
         # ValueError from validate_scenario rather than KeyError/TypeError.
+        from shipnav.scale import PROFILES, check_grid, default_profile, map_options, scaled
         scenario = json.loads(args.scenario.read_text()) if args.scenario else None
         if scenario is not None:
+            # A frozen scenario is already in model units and records its own planner settings.
             validate_scenario(scenario)
-            args.start,args.goal=scenario['start'],scenario['goal']
-        result = execute(scenario['map'] if scenario is not None else SeaMap.load(args.map).to_dict(), tuple(args.start), tuple(args.goal),
+            if args.profile:
+                raise ValueError('--profile applies to --map, not to a frozen --scenario')
+            map_data, start, goal = scenario['map'], tuple(scenario['start']), tuple(scenario['goal'])
+            options = map_options(map_data)
+        else:
+            real = SeaMap.load(args.map)
+            if args.profile is not None and args.profile not in PROFILES:
+                raise ValueError(f'Unknown profile; choose from {", ".join(PROFILES)}')
+            profile = PROFILES[args.profile] if args.profile else default_profile(real)
+            check_grid(real, profile)
+            map_data = scaled(real, profile).to_dict()
+            start, goal = (tuple(v/profile.length for v in p) for p in (args.start, args.goal))
+            options = profile.service_options()
+        placement = args.placement or ('uniform' if not map_data['metadata'].get('model_scale') else 'corridor')
+        result = execute(map_data, start, goal,
                          args.model, args.policy, not args.no_global_goals, args.seed, args.count,
-                         scenario=scenario,planner=args.planner,filtered=args.filtered,dynamics=args.dynamics)
+                         scenario=scenario,planner=args.planner,filtered=args.filtered,dynamics=args.dynamics,
+                         placement=placement, **options)
     except (ValueError, FileNotFoundError, RuntimeError) as error:
         parser.exit(2, str(error)+'\n')
     args.output.parent.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 5: Run `python -m pytest -v` and the parity capture again.** The whole suite must pass and every parity entry must be identical. Any difference means an optimisation changed behaviour; fix it rather than re-freezing.

- [ ] **Step 6: Try it from the command line.**

```bash
python -m shipnav.service --map maps/singapore-ubin.json --policy sarl --filtered \
  --start -4449.8 1383.9 --goal 4451.7 3038.1 --output results/ubin-sarl.json
```

`harbour_craft` and corridor traffic are selected automatically for this map. The start and goal are the `map_demo` Ubin endpoints in local metres.

- [ ] **Step 7: Review the diff and commit.**

```bash
git add rebuild/src/shipnav/scale.py rebuild/tests/test_scale.py rebuild/src/shipnav/maps.py \
  rebuild/src/shipnav/safety.py rebuild/src/shipnav/controllers/mpc.py rebuild/src/shipnav/scenarios.py \
  rebuild/src/shipnav/service.py
git commit -m "feat: run real maps through the model frame with vessel profiles"
```

### Task 7: Export and draw recorded navigation results

**Files:** `rebuild/src/shipnav/export.py`, `rebuild/tests/test_export.py`.

**Interfaces:** Consumes the schema-2 `execute()` result. Produces `draw(ax, result: dict, frame_index: int = -1) -> None`, `land_polygons(map_data)`, `csv_rows(result)`, `video_frames(result, speedup)`, `default_speedup(result)` and `export_run(result: dict, path: Path, speedup: float | None = None) -> None`.

- Results are in model units. `scale.units(result['map'])` gives metres per unit and seconds per model second; both are 1 for synthetic maps. Axis ticks, the title time and every CSV column are converted to metres, m/s and seconds.
- Vessels are drawn as true-size discs plus fixed-size markers, so they stay visible on kilometre maps.

- JSON is the complete reproducibility artifact, including nested observations and predictions.
- CSV has one row per recorded frame. Action columns (nominal/executed/actual velocity, intervention, no-feasible-action, decision latency, deadline miss, oldest observed target age, collision type, clearance) describe the step leaving that frame. The terminal frame has no action, so those columns are empty; nothing is invented.
- PNG is a final overview. MP4 replays in simulated time ÷ speed-up. The default speed-up is 1 for short runs (every frame at `1/dt` FPS, as before) and otherwise a whole number giving about two minutes of video. Frames are strided to stay at or under 30 FPS, and the terminal frame is always included.
- The overlay draws diagnostic `k` only on frame `k`: nominal and executed velocity arrows, predicted target tracks, and the filter's chosen rollout. `no_feasible_action` shows `n/a` when the filter is off.

- [ ] **Step 1: Write the behaviour tests.**

Create `rebuild/tests/test_export.py` with:

```python
import copy
import csv
import json
import shutil
import subprocess
import pytest
from shipnav.maps import SeaMap
from shipnav.service import execute
from shipnav.export import export_run


def test_exports_use_the_recorded_run(tmp_path):
    result = execute(SeaMap((0, 0, 10, 10)).to_dict(), (2, 2), (4, 2), policy_name='direct', count=0)
    export_run(result, tmp_path/'run.json')
    export_run(result, tmp_path/'run.csv')
    export_run(result, tmp_path/'run.png')
    loaded = json.loads((tmp_path/'run.json').read_text())
    assert loaded['frames'] == result['frames']
    with (tmp_path/'run.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result['frames'])
    assert float(rows[-1]['time_s']) == result['elapsed']
    assert (tmp_path/'run.png').read_bytes().startswith(b'\x89PNG')
    with pytest.raises(ValueError):
        export_run(result, tmp_path/'run.exe')


def test_polygon_land_and_diagnostics_come_from_the_stored_trace(tmp_path):
    sea = SeaMap.load('maps/harbour.json')
    result = execute(sea.to_dict(), (2, 2), (22, 22), policy_name='direct', count=3, seed=1, filtered=True)
    stored = json.loads(json.dumps(result))  # no policy or map object survives this
    export_run(stored, tmp_path/'run.png')
    export_run(stored, tmp_path/'run.csv')
    with (tmp_path/'run.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(stored['frames']) == len(stored['diagnostics'])+1
    first, d = rows[0], stored['diagnostics'][0]
    assert float(first['executed_vx']) == d['executed'][0]
    assert first['override'] == str(d['override'])
    assert rows[-1]['executed_vx'] == ''  # terminal frame: no invented action
    assert stored == json.loads(json.dumps(result))  # export did not mutate the trace


@pytest.mark.video
@pytest.mark.skipif(shutil.which('ffprobe') is None, reason='FFmpeg not installed')
def test_mp4_keeps_every_frame_at_one_over_dt(tmp_path):
    result = execute(SeaMap((0, 0, 10, 10)).to_dict(), (2, 2), (4, 2), policy_name='direct', count=0)
    export_run(result, tmp_path/'run.mp4')
    probe = json.loads(subprocess.run(
        ['ffprobe', '-v', 'error', '-count_frames', '-show_entries',
         'stream=nb_read_frames,r_frame_rate', '-of', 'json', str(tmp_path/'run.mp4')],
        check=True, capture_output=True, text=True).stdout)['streams'][0]
    assert probe['r_frame_rate'] == '4/1'
    assert int(probe['nb_read_frames']) == len(result['frames'])
```

- [ ] **Step 2: Run the tests before implementation.**

```bash
python -m pytest tests/test_export.py -v
```

Expected: failure because the module does not exist. Resolve missing environment dependencies (including FFmpeg for the `video` test) before treating a failure as evidence about application behaviour.

- [ ] **Step 3: Implement the following file.**

Create `rebuild/src/shipnav/export.py` with:

```python
"""Draw and export a recorded `execute()` result. Never re-runs a policy.

Results are stored in model units. Maps scaled by `shipnav.scale.to_model` carry
`model_scale` metadata; drawing, CSV and video convert to metres and seconds here.
"""
from math import ceil
from pathlib import Path
import csv
import json
from matplotlib.figure import Figure
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.animation import FFMpegWriter, writers
from matplotlib.ticker import FuncFormatter
from shapely.geometry import shape
from shipnav.scale import units

CSV_FIELDS = ['time_s', 'x_m', 'y_m', 'vx_mps', 'vy_mps', 'goal_index',
              'nominal_vx', 'nominal_vy', 'executed_vx', 'executed_vy', 'actual_vx', 'actual_vy',
              'override', 'no_feasible_action', 'decision_ms', 'deadline_miss',
              'max_target_age_s', 'ship_collision', 'land_collision', 'clearance_m']


def land_polygons(map_data: dict):
    """Yield every land Polygon of a schema-1 map, splitting MultiPolygons."""
    for feature in map_data['features']:
        geometry = shape(feature['geometry'])
        yield from getattr(geometry, 'geoms', [geometry])


def _patch(polygon, **style):
    # One compound path per polygon so interior rings (lakes/holes) stay open water.
    vertices, codes = [], []
    for ring in (polygon.exterior, *polygon.interiors):
        points = list(ring.coords)
        vertices += points
        codes += [MplPath.MOVETO] + [MplPath.LINETO]*(len(points)-2) + [MplPath.CLOSEPOLY]
    return PathPatch(MplPath(vertices, codes), **style)


def draw(ax, result: dict, frame_index: int = -1) -> None:
    ax.clear()
    x0, y0, x1, y1 = result['map']['bounds']
    L, T = units(result['map'])
    ax.set(xlim=(x0, x1), ylim=(y0, y1), xlabel='x (m)', ylabel='y (m)', aspect='equal')
    metres = FuncFormatter(lambda v, _: f'{v*L:g}')
    ax.xaxis.set_major_formatter(metres)
    ax.yaxis.set_major_formatter(metres)
    ax.set_facecolor('#dceef5')
    for polygon in land_polygons(result['map']):
        ax.add_patch(_patch(polygon, facecolor='#9b987f', edgecolor='none'))
    route = result['route']
    if route:
        ax.plot(*zip(*route), '--', color='#555555', label='Global route')
    goals = result['goals']
    if goals:
        ax.scatter(*zip(*goals), marker='*', color='#cf4b35', label='Goals', zorder=3)
    frames = result['frames']
    if frames:
        i = len(frames)-1 if frame_index < 0 else min(frame_index, len(frames)-1)
        frame = frames[i]
        ax.plot(*zip(*(f['position'] for f in frames[:i+1])), color='#17659c', label='Travelled')
        ax.add_patch(Circle(frame['position'], result['settings']['radius'], color='#17659c'))
        for position, ship in zip(frame['traffic'], result['traffic_definitions']):
            ax.add_patch(Circle(position, ship['radius'], color='#df9a36'))
        # True-size discs vanish on kilometre maps; fixed-size markers keep vessels visible.
        ax.scatter(*frame['position'], s=30, color='#17659c', edgecolors='white', zorder=4, label='Own ship')
        if frame['traffic']:
            ax.scatter(*zip(*frame['traffic']), s=30, color='#df9a36', edgecolors='white', zorder=4, label='Traffic')
        title = f"{result['status']} — {frame['t']*T:.1f} s"
        diagnostics = result.get('diagnostics', [])
        # Diagnostic k is the decision from frame k to k+1: the terminal frame has none,
        # and an earlier frame never shows a later tick's prediction.
        if i < len(diagnostics):
            d = diagnostics[i]
            for key, color in (('nominal', '#b05090'), ('executed', '#248060')):
                ax.arrow(*frame['position'], *d[key], color=color, width=.025, label=key.capitalize())
            for target in d['predictions']:
                ax.plot(*zip(*target['points']), ':', color='#e5a02e')
            if d['path']:
                ax.plot(*zip(*d['path']), '-', color='#248060', alpha=.5)
            infeasible = 'n/a' if d['no_feasible_action'] is None else d['no_feasible_action']
            title += f" | override={d['override']} | no feasible action={infeasible}"
        ax.set_title(title)
    ax.legend(loc='upper left')


def csv_rows(result: dict):
    """One row per recorded frame. Action columns describe the step leaving that
    frame; the terminal frame has no action, so those columns stay empty."""
    diagnostics = result.get('diagnostics', [])
    L, T = units(result['map'])
    V = L/T
    for i, f in enumerate(result['frames']):
        row = {'time_s': f['t']*T, 'x_m': f['position'][0]*L, 'y_m': f['position'][1]*L,
               'vx_mps': f['velocity'][0]*V, 'vy_mps': f['velocity'][1]*V, 'goal_index': f['goal_index']}
        if i < len(diagnostics):
            d = diagnostics[i]
            ages = [s['age'] for s in d['observed']]
            row.update(nominal_vx=d['nominal'][0]*V, nominal_vy=d['nominal'][1]*V,
                       executed_vx=d['executed'][0]*V, executed_vy=d['executed'][1]*V,
                       actual_vx=d['actual_velocity'][0]*V, actual_vy=d['actual_velocity'][1]*V,
                       override=d['override'], no_feasible_action=d['no_feasible_action'],
                       decision_ms=d['decision_ms'], deadline_miss=d['deadline_miss'],
                       max_target_age_s=max(ages)*T if ages else None,
                       ship_collision=d['ship_collision'], land_collision=d['land_collision'],
                       clearance_m=None if d['clearance'] is None else d['clearance']*L)
        yield row


def video_frames(result: dict, speedup: float = 1., max_fps: float = 30.):
    """Frame indices and FPS so the video lasts simulated time / speedup. Strides frames
    when needed to stay under `max_fps`; the terminal frame is always included."""
    if speedup <= 0:
        raise ValueError('Speed-up must be positive')
    n, (_, T) = len(result['frames']), units(result['map'])
    rate = speedup/(result['settings']['dt']*T)
    stride = max(1, ceil(rate/max_fps - 1e-9))
    indices = list(range(0, n, stride))
    if indices[-1] != n-1:
        indices.append(n-1)
    return indices, rate/stride


def default_speedup(result: dict, target_s: float = 120.) -> float:
    """1 for short runs; otherwise a whole-number speed-up giving about `target_s` of video."""
    _, T = units(result['map'])
    return float(max(1, ceil(result['elapsed']*T/target_s)))


def export_run(result: dict, path: Path, speedup: float | None = None) -> None:
    suffix = path.suffix.lower()
    if suffix not in ('.json', '.csv', '.png', '.mp4'):
        raise ValueError('Choose JSON, CSV, PNG or MP4')
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == '.json':
        path.write_text(json.dumps(result, indent=2, allow_nan=False))
        return
    if suffix == '.csv':
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(csv_rows(result))
        return
    figure = Figure(figsize=(7, 7))
    ax = figure.subplots()
    if suffix == '.png':
        draw(ax, result)
        figure.savefig(path, dpi=150)
        return
    if not writers.is_available('ffmpeg'):
        raise RuntimeError('FFmpeg is required to export MP4')
    frames = result['frames']
    if not frames:
        raise ValueError('No frames to export')
    indices, fps = video_frames(result, default_speedup(result) if speedup is None else speedup)
    writer = FFMpegWriter(fps=fps, codec='libx264')
    with writer.saving(figure, str(path), dpi=100):
        for i in indices:
            draw(ax, result, i)
            writer.grab_frame()
```

- [ ] **Step 4: Run the same test command and inspect the result.** Expected: PASS. Do not weaken assertions to conceal an algorithm or interface failure.

- [ ] **Step 5: Review the diff and commit.**

```bash
git add rebuild/src/shipnav/export.py rebuild/tests/test_export.py
git commit -m "feat: export and draw recorded navigation results"
```

### Task 8: Build the navigation GUI

**Files:** `rebuild/src/shipnav/gui.py`, `rebuild/tests/test_gui.py`.

**Interfaces:** Consumes `SeaMap`, `execute`, `draw`, and `export_run`. Produces `Window(sea: SeaMap, model_dir: str = "", runner=execute)`, `.apply_click(x, y)`, `.set_map(path)`, `.start_run()`, `.check_worker()` and module entry `python -m shipnav.gui`.

- Worker functions never touch Qt widgets. `start_run` snapshots every control value on the GUI thread and passes them positionally plus `planner=`, `dynamics=`, `filtered=`, `observation=` keywords. Runners must accept `**kwargs`. A Qt timer polls the future and moves results to the main thread.
- Two control rows: map/model/click mode/policy/ships/seed/run/stop/replay/export, then planner/dynamics/filter/perception/vessel profile/traffic placement/replay speed. All edit controls are disabled while a job runs. A Matplotlib navigation toolbar provides zoom and pan.
- The window keeps the metric map (`real`, including land edits in metres) and the model-unit map the service runs (`sea`). Loading a map picks `default_profile`: `model` up to 250,000 m², otherwise `harbour_craft` with traffic along the route. Clicks arrive in model units; land rectangles are converted to metres and the map is re-scaled.
- Changing the vessel profile keeps chosen endpoints at the same place in metres when they are still clear. A profile whose planner grid exceeds 250,000 cells is refused with a message, and the previous profile stays selected.
- Replay speed 1×/10×/50×/200× steps over frames whenever redraws would come faster than every 40 ms.

- [ ] **Step 1: Write the behaviour tests.**

Create `rebuild/tests/test_gui.py` with:

```python
from threading import Event
from time import sleep
import pytest
from shapely.geometry import box
from shipnav.maps import SeaMap
from shipnav.gui import Window
from shipnav.service import execute


def test_start_goal_and_land_editing(qtbot):
    window = Window(SeaMap((0, 0, 24, 24), metadata={'description': 'test'}))
    qtbot.addWidget(window)
    window.apply_click(3, 3)
    assert window.start == (3, 3)
    window.mode.setCurrentIndex(1)
    window.apply_click(21, 21)
    assert window.goal == (21, 21)
    window.mode.setCurrentIndex(2)
    window.apply_click(8, 8)
    window.apply_click(10, 10)
    assert len(window.sea.land) == 1 and window.sea.land[0].equals(box(8, 8, 10, 10))
    assert window.sea.to_dict()['metadata'] == {'description': 'test'}
    window.apply_click(2, 2)
    window.apply_click(4, 4)
    assert len(window.sea.land) == 1 and 'blocks start' in window.status.text()
    window.mode.setCurrentIndex(0)
    window.apply_click(9, 9)
    assert window.start == (3, 3) and 'clear of land' in window.status.text()
    window.close()


def test_real_map_loads_in_model_units_with_a_vessel_profile(qtbot):
    window = Window(SeaMap.load('maps/harbour.json'))
    qtbot.addWidget(window)
    window.set_map('maps/singapore-ubin.json')
    assert window.profile.name == 'harbour_craft' and window.vessel.currentText() == 'harbour_craft'
    assert window.placement.currentIndex() == 1
    assert window.sea.to_dict()['metadata']['model_scale']['length_m'] == 10
    assert window.sea.bounds == pytest.approx(tuple(v/10 for v in window.real.bounds))
    window.mode.setCurrentIndex(0)
    window.apply_click(-200, 50)                   # 2 km west, 270 m from shore
    assert window.start == (-200, 50)
    window.vessel.setCurrentText('coastal_ship')   # L = 50 m: same place in metres
    assert window.start == pytest.approx((-40, 10))
    window.vessel.setCurrentText('model')          # 1 m grid is far too large: refused
    assert window.profile.name == 'coastal_ship' and 'cells' in window.status.text()
    window.close()


def test_land_edit_on_a_scaled_map_is_stored_in_metres(qtbot):
    window = Window(SeaMap((0, 0, 1000, 1000)))
    qtbot.addWidget(window)
    window.vessel.setCurrentText('harbour_craft')
    window.mode.setCurrentIndex(2)
    window.apply_click(40, 40)
    window.apply_click(60, 50)
    assert window.real.land[0].equals(box(400, 400, 600, 500))
    assert window.sea.land[0].equals(box(40, 40, 60, 50))
    window.close()


def test_worker_leaves_gui_responsive_and_restores_controls(qtbot):
    entered, release = Event(), Event()
    def runner(*args, **kwargs):
        entered.set()
        release.wait(2)
        return execute(SeaMap((0, 0, 24, 24)).to_dict(), (2, 2), (4, 2), policy_name='direct', count=0)
    window = Window(SeaMap((0, 0, 24, 24)), runner=runner)
    qtbot.addWidget(window)
    window.start_run()
    qtbot.waitUntil(entered.is_set)
    assert not window.run_button.isEnabled()
    assert not window.planner.isEnabled()
    assert window.stop_button.isEnabled()
    window.apply_click(7, 7)
    assert window.start == (2, 2)
    release.set()
    qtbot.waitUntil(lambda: window.future is None)
    assert window.result['status'] == 'success'
    assert window.export_button.isEnabled()
    window.close()


def test_revised_controls_reach_the_runner_unchanged(qtbot):
    calls = []
    def capture(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError('controller failed')
    window = Window(SeaMap((0, 0, 24, 24)), runner=capture)
    qtbot.addWidget(window)
    window.planner.setCurrentText('theta')
    window.dynamics.setCurrentText('marine')
    window.filter_mode.setCurrentIndex(1)
    window.perception.setCurrentIndex(1)
    window.start_run()
    qtbot.waitUntil(lambda: window.future is None)
    assert calls == [{'planner': 'theta', 'dynamics': 'marine', 'filtered': True,
                      'observation': {'noise': .1, 'delay': .5, 'dropout': .1},
                      'placement': 'uniform', 'resolution': 1., 'clearance': .7}]
    assert 'controller failed' in window.status.text()
    assert window.run_button.isEnabled() and window.planner.isEnabled()
    window.close()


def test_stop_cancels_the_active_run_not_a_stale_event(qtbot):
    entered = Event()
    def runner(*args, **kwargs):
        cancel = args[8]
        entered.set()
        while not cancel():
            sleep(.01)
        return execute(SeaMap((0, 0, 24, 24)).to_dict(), (2, 2), (20, 2), policy_name='direct',
                       count=0, cancel=cancel)
    window = Window(SeaMap((0, 0, 24, 24)), runner=runner)
    qtbot.addWidget(window)
    window.start_run()  # replaces the Event created in __init__
    qtbot.waitUntil(entered.is_set)
    window.stop_button.click()
    qtbot.waitUntil(lambda: window.future is None, timeout=2000)
    assert window.result['status'] == 'cancelled'
    assert window.run_button.isEnabled()
    window.close()


def test_worker_error_is_visible(qtbot):
    def broken(*args, **kwargs):
        raise ValueError('Bad checkpoint')
    window = Window(SeaMap((0, 0, 24, 24)), runner=broken)
    qtbot.addWidget(window)
    window.start_run()
    qtbot.waitUntil(lambda: window.future is None)
    assert 'Bad checkpoint' in window.status.text()
    assert window.run_button.isEnabled()
    assert window.result is None
    window.close()
```

- [ ] **Step 2: Run the tests before implementation.**

```bash
QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui.py -v
```

Expected: failure because the module does not exist.

- [ ] **Step 3: Implement the following file.**

Create `rebuild/src/shipnav/gui.py` with:

```python
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from math import ceil
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QSpinBox)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from shipnav.maps import SeaMap
from shipnav.service import execute
from shipnav.export import draw, export_run
from shipnav.scale import PROFILES, check_grid, default_profile, scaled, units

DEGRADED = {'noise': .1, 'delay': .5, 'dropout': .1}
REPLAY_SPEEDS = (1, 10, 50, 200)
MODEL = Path(__file__).resolve().parents[3]/'CrowdNav-20250813-DIP/crowd_nav/data/output_trained'


def default_endpoints(sea: SeaMap, clearance: float):
    """Two model units inside opposite corners when clear; otherwise unset."""
    x0, y0, x1, y1 = sea.bounds
    start, goal = (x0+2, y0+2), (x1-2, y1-2)
    return (start if sea.clear(start, start, clearance) else None,
            goal if sea.clear(goal, goal, clearance) else None)


class Window(QWidget):
    def __init__(self, sea: SeaMap, model_dir: str = '', runner=execute):
        super().__init__()
        self.setWindowTitle('Ship navigation rebuild')
        self.resize(1100, 850)
        # `real` is the metric map (with any land edits); `sea` is what the service runs:
        # the same map in model units for the selected vessel profile.
        self.model_dir, self.runner = model_dir, runner
        self.real, self.profile = sea, default_profile(sea)
        check_grid(sea, self.profile)
        self.sea = scaled(sea, self.profile)
        self.start, self.goal = default_endpoints(self.sea, self.clearance)
        self.result, self.future, self.corner = None, None, None
        self.job_kind, self.replay_index, self.replay_step = '', 0, 1
        self.cancel_event = Event()
        self.pool = ThreadPoolExecutor(max_workers=1)
        layout, toolbar, options = QVBoxLayout(self), QHBoxLayout(), QHBoxLayout()
        layout.addLayout(toolbar)
        layout.addLayout(options)
        self.map_button = QPushButton('Load map')
        self.model_button = QPushButton('Load model')
        self.mode = QComboBox()
        self.mode.addItems(['Set start', 'Set goal', 'Add land (two corners)'])
        self.mode.currentIndexChanged.connect(self.clear_corner)
        self.policy = QComboBox()
        self.policy.addItems(['sarl', 'orca', 'mpc', 'direct'])
        self.count = QSpinBox()
        self.count.setRange(0, 15)
        self.count.setValue(5)
        self.count.setPrefix('Ships: ')
        self.seed = QSpinBox()
        self.seed.setRange(0, 9999)
        self.seed.setPrefix('Seed: ')
        self.run_button, self.stop_button = QPushButton('Run'), QPushButton('Stop')
        self.replay_button, self.export_button = QPushButton('Replay'), QPushButton('Export')
        self.planner = QComboBox()
        self.planner.addItems(['astar_smooth', 'theta'])
        self.dynamics = QComboBox()
        self.dynamics.addItems(['holonomic', 'marine'])
        self.filter_mode = QComboBox()
        self.filter_mode.addItems(['Unfiltered', 'Predictive filter'])
        # The supplied SARL never observed land or map edges; the demo defaults to the
        # filter. Choose Unfiltered explicitly for the ablation.
        self.filter_mode.setCurrentIndex(1)
        self.perception = QComboBox()
        self.perception.addItems(['Exact observations', 'Noisy delayed observations'])
        self.vessel = QComboBox()
        self.vessel.addItems(list(PROFILES))
        self.vessel.setCurrentText(self.profile.name)
        self.vessel.currentTextChanged.connect(self.set_profile)
        self.placement = QComboBox()
        self.placement.addItems(['Traffic anywhere', 'Traffic along route'])
        self.placement.setCurrentIndex(0 if self.profile.name == 'model' else 1)
        self.replay_speed = QComboBox()
        self.replay_speed.addItems([f'Replay {s}x' for s in REPLAY_SPEEDS])
        self.edit_controls = [self.map_button, self.model_button, self.mode,
                              self.policy, self.count, self.seed, self.run_button,
                              self.planner, self.dynamics, self.filter_mode, self.perception,
                              self.vessel, self.placement]
        for widget in self.edit_controls[:7] + [self.stop_button, self.replay_button, self.export_button]:
            toolbar.addWidget(widget)
        for widget in self.edit_controls[7:] + [self.replay_speed]:
            options.addWidget(widget)
        self.status = QLabel('Select start and goal, then run')
        layout.addWidget(self.status)
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.subplots()
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))  # zoom/pan for kilometre maps
        layout.addWidget(self.canvas)
        self.canvas.mpl_connect('button_press_event', self.clicked)
        self.map_button.clicked.connect(self.load_map)
        self.model_button.clicked.connect(self.load_model)
        self.run_button.clicked.connect(self.start_run)
        self.stop_button.clicked.connect(lambda: self.cancel_event.set())
        self.replay_button.clicked.connect(self.replay)
        self.export_button.clicked.connect(self.export)
        self.poll = QTimer(self)
        self.poll.setInterval(50)
        self.poll.timeout.connect(self.check_worker)
        self.poll.start()
        self.playback = QTimer(self)
        self.playback.timeout.connect(self.next_frame)
        self.set_busy(False)
        self.preview()

    @property
    def clearance(self):
        return self.profile.service_options()['clearance']

    def clear_corner(self):
        self.corner = None

    def set_profile(self, name):
        old, profile = self.profile, PROFILES[name]
        try:
            check_grid(self.real, profile)
        except ValueError as error:
            self.vessel.blockSignals(True)
            self.vessel.setCurrentText(old.name)
            self.vessel.blockSignals(False)
            self.status.setText(str(error))
            return
        ratio = old.length/profile.length
        self.profile, self.sea, self.corner = profile, scaled(self.real, profile), None
        # Keep the chosen endpoints at the same place in metres when still clear.
        self.start, self.goal = (None if p is None or not self.sea.clear(q := (p[0]*ratio, p[1]*ratio), q, self.clearance)
                                 else q for p in (self.start, self.goal))
        self.invalidate()
        self.status.setText(f'Vessel profile {name}: radius {profile.radius_m:g} m, '
                            f'{profile.speed_mps:g} m/s, grid {profile.resolution_m:g} m')

    def set_busy(self, busy):
        for widget in self.edit_controls:
            widget.setEnabled(not busy)
        self.stop_button.setEnabled(busy and self.job_kind == 'run')
        self.replay_button.setEnabled(not busy and self.result is not None)
        self.export_button.setEnabled(not busy and self.result is not None)

    def invalidate(self):
        self.playback.stop()
        self.result = None
        self.set_busy(False)
        self.preview()

    def preview(self):
        preview = {'map': self.sea.to_dict(), 'route': [],
                   'goals': [p for p in (self.start, self.goal) if p is not None],
                   'frames': [], 'settings': {'radius': .5}, 'traffic_definitions': []}
        draw(self.ax, preview)
        self.ax.set_title('Start and destination')
        self.canvas.draw_idle()

    def clicked(self, event):
        if event.inaxes is self.ax and event.xdata is not None and event.ydata is not None:
            self.apply_click(event.xdata, event.ydata)

    def apply_click(self, x, y):
        if self.future is not None:
            return
        point = (float(x), float(y))
        try:
            if self.mode.currentIndex() < 2:
                if not self.sea.clear(point, point, self.clearance):
                    raise ValueError('Choose a point clear of land and map edges')
                if self.mode.currentIndex() == 0:
                    self.start = point
                else:
                    self.goal = point
            elif self.corner is None:
                self.corner = point
                self.status.setText('Choose the opposite land corner')
                return
            else:
                a, b = self.corner, point
                self.corner = None
                L, _ = units(self.sea.to_dict())
                rect = (min(a[0], b[0])*L, min(a[1], b[1])*L, max(a[0], b[0])*L, max(a[1], b[1])*L)
                real = SeaMap(self.real.bounds, self.real.land + (rect,), self.real.to_dict()['metadata'])
                sea = scaled(real, self.profile)
                if not all(p is None or sea.clear(p, p, self.clearance) for p in (self.start, self.goal)):
                    raise ValueError('New land blocks start or goal')
                self.real, self.sea = real, sea
            self.invalidate()
            self.status.setText('Scene updated')
        except ValueError as error:
            self.status.setText(str(error))

    def load_map(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Load map', '', 'Map (*.json)')
        if filename:
            self.set_map(Path(filename))

    def set_map(self, path: Path):
        try:
            real = SeaMap.load(path)
            profile = default_profile(real)
            check_grid(real, profile)
            sea = scaled(real, profile)
        except (ValueError, TypeError, KeyError, OSError) as error:
            self.status.setText(f'Map not loaded: {error}')
            return
        self.real, self.profile, self.sea, self.corner = real, profile, sea, None
        self.vessel.blockSignals(True)
        self.vessel.setCurrentText(profile.name)
        self.vessel.blockSignals(False)
        self.placement.setCurrentIndex(0 if profile.name == 'model' else 1)
        self.start, self.goal = default_endpoints(self.sea, self.clearance)
        self.invalidate()
        self.status.setText(f'Map loaded with vessel profile {profile.name}; select valid start and goal')

    def load_model(self):
        directory = QFileDialog.getExistingDirectory(self, 'Folder with policy.config and rl_model.pth')
        if directory:
            self.model_dir = directory
            self.invalidate()
            self.status.setText('Model folder selected; weights are validated when Run starts')

    def start_run(self):
        if self.future is not None:
            return
        if self.start is None or self.goal is None:
            self.status.setText('Select a start and a goal clear of land first')
            return
        self.playback.stop()
        self.result = None
        self.cancel_event = Event()
        self.job_kind = 'run'
        self.status.setText('Running navigation…')
        self.set_busy(True)
        # Snapshot every widget value here; the worker never reads Qt objects.
        self.future = self.pool.submit(self.runner, self.sea.to_dict(), self.start, self.goal,
            self.model_dir, self.policy.currentText(), True, self.seed.value(), self.count.value(),
            self.cancel_event.is_set,
            planner=self.planner.currentText(),
            dynamics=self.dynamics.currentText(),
            filtered=self.filter_mode.currentIndex() == 1,
            observation={} if self.perception.currentIndex() == 0 else dict(DEGRADED),
            placement='uniform' if self.placement.currentIndex() == 0 else 'corridor',
            **self.profile.service_options())

    def check_worker(self):
        if self.future is None or not self.future.done():
            return
        try:
            value = self.future.result()
            if self.job_kind == 'run':
                self.result = value
                draw(self.ax, value)
                self.canvas.draw_idle()
                _, T = units(value['map'])
                self.status.setText(f"{value['status']}: {value['elapsed']*T:.1f} simulated seconds")
            else:
                self.status.setText('Export saved')
        except Exception as error:
            self.status.setText(f'Failed: {error}')
        finally:
            self.future = None
            self.set_busy(False)

    def replay(self):
        if self.result is not None:
            _, T = units(self.result['map'])
            interval = 1000*self.result['settings']['dt']*T/REPLAY_SPEEDS[self.replay_speed.currentIndex()]
            # Redraws cannot keep up below ~40 ms; skip frames instead of falling behind.
            self.replay_step = max(1, ceil(40/interval))
            self.replay_index = 0
            self.playback.start(round(interval*self.replay_step))
            self.next_frame()

    def next_frame(self):
        if self.result is None or self.replay_index >= len(self.result['frames']):
            self.playback.stop()
            return
        last = len(self.result['frames'])-1
        draw(self.ax, self.result, self.replay_index)
        self.canvas.draw_idle()
        self.replay_index = last if self.replay_index < last < self.replay_index+self.replay_step \
            else self.replay_index+self.replay_step

    def export(self):
        if self.result is None or self.future is not None:
            return
        filename, _ = QFileDialog.getSaveFileName(self, 'Export', 'run.json',
            'JSON (*.json);;CSV (*.csv);;PNG (*.png);;MP4 (*.mp4)')
        if filename:
            self.playback.stop()
            self.job_kind = 'export'
            self.set_busy(True)
            self.status.setText('Exporting…')
            self.future = self.pool.submit(export_run, self.result, Path(filename))

    def closeEvent(self, event):
        if self.future is not None:
            self.cancel_event.set()
            self.status.setText('Waiting for the active job to stop; close again when it finishes')
            event.ignore()
            return
        self.poll.stop()
        self.playback.stop()
        self.pool.shutdown(wait=False, cancel_futures=True)
        event.accept()


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = Window(SeaMap.load(Path('maps/harbour.json')), str(MODEL))
    window.show()
    sys.exit(app.exec())
```

- [ ] **Step 4: Run the same test command and inspect the result.** Expected: PASS. `test_stop_cancels_the_active_run_not_a_stale_event` must fail if Stop is connected directly to `self.cancel_event.set`; check that once.

- [ ] **Step 5: Review the diff and commit.**

```bash
git add rebuild/src/shipnav/gui.py rebuild/tests/test_gui.py
git commit -m "feat: build the navigation gui"
```

### Task 9: Run paired smoke experiments

**Files:** `rebuild/src/shipnav/evaluate.py`, `rebuild/tests/test_evaluate.py`.

**Interfaces:** Consumes `execute` and its trace schema. Produces `summarize(runs: list[dict]) -> dict` and `evaluate(map_data, model_dir, output: Path, seeds=range(10), counts=(5,8,12,15), runner=execute) -> list[dict]`. No-global-goals ablates waypoint guidance; direct ablates local collision avoidance; ORCA supplies a rule-based comparator.

This is a smoke utility for `maps/harbour.json` only: endpoints are fixed at `(2,2)` → `(22,22)` and all variants are unfiltered. Comparative results come from Task 11's frozen-scenario benchmark. The code below is unchanged from the first revision and passes against the current service.

- [ ] **Step 1: Write the behaviour tests.**

Create `rebuild/tests/test_evaluate.py` with:

```python
from shipnav.evaluate import summarize, evaluate
from shipnav.maps import SeaMap


def test_failures_do_not_improve_average_completion_time():
    rows = [{'status': 'success', 'elapsed': 10, 'distance': 9, 'inference_ms': [2]},
            {'status': 'collision', 'elapsed': 1, 'distance': 1, 'inference_ms': [4]},
            {'status': 'error', 'error': 'placement failed'}]
    summary = summarize(rows)
    assert summary['runs'] == 3 and summary['successes'] == 1
    assert summary['mean_success_time_s'] == 10
    assert summary['collisions'] == 1 and summary['errors'] == 1
    assert summary['mean_step_inference_ms'] == 3


def test_evaluation_pairs_seed_and_scenario_and_saves_every_run(tmp_path):
    calls = []
    def runner(map_data, start, goal, model, policy, global_goals, seed, count):
        calls.append((policy, global_goals, seed, count))
        return {'status': 'success', 'elapsed': 10, 'distance': 9,
                'inference_ms': [1], 'traffic_definitions': [{'seed': seed, 'count': count}]}
    rows = evaluate(SeaMap((0, 0, 24, 24)).to_dict(), '', tmp_path,
                    seeds=range(2), counts=(5,), runner=runner)
    assert len(calls) == 8 and len(rows) == 4
    assert len(list(tmp_path.glob('*.json'))) == 8
    assert (tmp_path/'summary.csv').exists()
    assert all(r['runs'] == 2 for r in rows)
```

- [ ] **Step 2: Run `python -m pytest tests/test_evaluate.py -v` before implementation.** Expect a missing module.

- [ ] **Step 3: Implement the following file.**

Create `rebuild/src/shipnav/evaluate.py` with:

```python
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
```

- [ ] **Step 4: Run the same test command.** Expected: PASS.

- [ ] **Step 5: Review the diff and commit.**

```bash
git add rebuild/src/shipnav/evaluate.py rebuild/tests/test_evaluate.py
git commit -m "feat: run paired navigation smoke experiments"
```

### Task 10: Native-window acceptance

The revised controls and diagnostics from the first revision's Task 10 are now in Tasks 7 and 8. What remains is manual and needs the real checkpoint and a display.

- [ ] Run the complete suite once: `QT_QPA_PLATFORM=offscreen python -m pytest -v`.
- [ ] Run `python -m shipnav.gui` in a native window. It opens `maps/harbour.json` with the supplied model folder, start `(2,2)` and destination `(22,22)`. Run SARL with 5 ships and the default predictive filter; record the status.
- [ ] Switch to *Unfiltered* and rerun the same seed. The expected outcome is a collision (observed: land collision at 4.5 s). Replay it and check the last diagnostic: confirm whether `land_collision` was the map edge or the island. Record the result; it is the motivating case for plan 05's constraint penalty, not a GUI defect.
- [ ] While running, move/resize the window. Confirm scene edits, the second control row and duplicate runs are disabled. Press Stop; wait for the current inference step and confirm `cancelled`, then rerun.
- [ ] Change the destination and add a land rectangle using two clicks. Run again and verify the global route changes. Try covering the start with land and selecting a point on land; both must be rejected with an understandable status message.
- [ ] Load `maps/singapore-ubin.json`. It must select `harbour_craft` and *Traffic along route*. Zoom in with the toolbar, set start `(-4449.8, 1383.9)` and goal `(4451.7, 3038.1)` m (the `map_demo` endpoints; tick labels are in metres), and run SARL with 5 ships and the filter. Confirm the window stays responsive for the tens of seconds the run takes, replay it at 50×, and export an MP4 (about two minutes long). Repeat with MPC + marine (expect over a minute).
- [ ] Switch the Ubin run to `coastal_ship`, then to `model`. The first re-scales and keeps valid endpoints; the second must be refused with the cell-count message.
- [ ] Load `maps/singapore-southern-islands.json` and plan a route between two islands to confirm the second geography also runs.
- [ ] Select an empty model folder and run SARL. Verify an error appears and controls are restored. Then select the correct model folder and verify recovery.
- [ ] Select Theta*, marine dynamics and noisy delayed observations; run and replay. Check the overlay arrows, predicted tracks and the title's override / no-feasible-action fields frame by frame.
- [ ] Replay a completed trace; export JSON, CSV, PNG and MP4 to different filenames. Verify exported coordinates/times match the display and that a 10-second simulated interval takes about 10 seconds in the MP4.
- [ ] Check the video with `ffprobe -v error -count_frames -show_entries stream=codec_name,nb_read_frames,r_frame_rate -of json results/demo.mp4`. Expected: 4 FPS at `dt=.25`, and `nb_read_frames` equal to the number of recorded frames, including the initial and terminal frame.
- [ ] Try an impossible map with a wall from one boundary to the opposite boundary; the GUI must show the planning failure (`No route at this grid resolution`), not an empty successful route.
- [ ] Run `python -m shipnav.evaluate --seeds 2 --counts 5`. Inspect all eight traces and `summary.csv`. Then run `python -m shipnav.evaluate` (160 episodes). This is a smoke batch only.
- [ ] If placement fails for dense traffic, retain the error rows. Reduce the count or enlarge the scenario in a documented subsequent experiment; do not drop difficult seeds from only one algorithm's results. Latencies include cold first inference and are descriptive, not real-time guarantees.
- [ ] Do not claim the application works solely because the offscreen GUI tests pass.

### Task 11: Metrics and paired evidence

**Files:** `rebuild/src/shipnav/metrics.py`, `rebuild/src/shipnav/benchmark.py`, `rebuild/tests/test_metrics.py`, `rebuild/tests/test_geometric_metrics.py`.

**Interfaces:** `metrics(run) -> dict`; `geometry_metrics(run) -> dict`; `quantile(values, q)`; `paired_interval(a: dict[id, float], b: dict[id, float]) -> dict`; `hierarchical_interval(a, b)` over training seed → scenario ID → value; `benchmark(scenarios, model_dir, output, variants=VARIANTS, runner=execute) -> list[dict]`.

- Failed runs keep their scenario hash and stay in the denominator. Planning failures are `planning_failure`; any other exception is `error`.
- `no_feasible_actions` is `None` when the filter was off (not evaluated), never 0.
- Cross-track error is measured against the run's own route. For planner comparisons, save one fixed reference route per scenario in the protocol and score against that; a planner's own route is only appropriate for its controller's tracking error.
- `domain_time_s` is dt-weighted exposure below 1 m ship clearance (an experimental domain definition). `sampled_land_clearance_m` is sampled at frames; collision status itself uses swept checks.
- Heading/speed violations use `abs(wrapped Δheading) > .35·dt` or `abs(Δspeed) > .2·dt` between successive marine diagnostics, starting from the first route leg's bearing at rest.

- [ ] **Step 1: Write the behaviour tests.**

Create `rebuild/tests/test_metrics.py` with:

```python
import json
from pathlib import Path
import pytest
from shipnav.metrics import paired_interval,quantile
from shipnav.benchmark import benchmark

def test_tail_and_paired_direction():
    assert quantile([1,2,3,100],.99)>90
    r=paired_interval({'a':1,'b':1},{'a':0,'b':0})
    assert r['difference']==r['low']==r['high']==1
    with pytest.raises(ValueError): paired_interval({'a':1},{'b':1})

def test_failed_runs_remain_in_denominator(tmp_path):
    scene={'map':{},'start':[0,0],'goal':[1,1],'seed':0}
    def broken(*args,**kwargs): raise RuntimeError('controller failed')
    rows=benchmark([scene],'',tmp_path,{'a':{},'b':{}},runner=broken)
    assert len(rows)==2 and all(r['status']=='error' for r in rows)
    assert len(list(tmp_path.glob('*-scenario.json')))==1

def test_frozen_scenarios_score_filtered_unfiltered_and_unreachable(tmp_path):
    scenes=[json.loads(Path('scenarios',n).read_text()) for n in ('head_on.json','unreachable.json')]
    rows=benchmark(scenes,'',tmp_path,{'direct':dict(policy_name='direct'),
                                       'direct_filtered':dict(policy_name='direct',filtered=True)})
    by={(r['scenario_hash'][:8],r['variant']):r for r in rows}
    assert len(rows)==4
    unfiltered=[r for r in rows if r['variant']=='direct' and r['status']!='planning_failure']
    assert unfiltered and all(r['no_feasible_actions'] is None for r in unfiltered)
    assert sum(r['status']=='planning_failure' for r in rows)==2
    assert all('cross_track_max_m' in r for r in rows if r['status']!='planning_failure')
    json.loads((tmp_path/'metrics.json').read_text())
```

Create `rebuild/tests/test_geometric_metrics.py` with:

```python
import pytest
from shipnav.maps import SeaMap
from shipnav.metrics import point_segment_distance,land_clearance,hierarchical_interval,geometry_metrics
from shipnav.service import execute

def test_geometric_metrics_have_known_values():
    assert point_segment_distance((3,4),(0,0),(10,0))==4
    assert land_clearance(SeaMap((0,0,10,10)),(2,5),.5)==1.5
    assert land_clearance(SeaMap((0,0,10,10),((4,4,6,6),)),(5,5),.5)==-1.5

def test_hierarchical_interval_uses_both_seed_levels():
    a={0:{'x':1,'y':1},1:{'x':1,'y':1}}; b={0:{'x':0,'y':0},1:{'x':0,'y':0}}
    assert hierarchical_interval(a,b)['low']==1
    with pytest.raises(ValueError): hierarchical_interval(a,{0:b[0]})

def test_marine_run_starting_off_east_has_no_false_dynamics_violation():
    run=execute(SeaMap((0,0,24,24)).to_dict(),(2,2),(2,20),policy_name='direct',count=0,dynamics='marine')
    m=geometry_metrics(run)
    assert run['status']=='success' and m['dynamics_violations']==0
    assert m['cross_track_max_m']<1e-6
    assert geometry_metrics({'status':'error','error':'x'})=={}
```

- [ ] **Step 2: Run `python -m pytest tests/test_metrics.py tests/test_geometric_metrics.py -v`.** Expect a missing module, not a missing third-party dependency.

- [ ] **Step 3: Implement the following files.**

Create `rebuild/src/shipnav/metrics.py` with:

```python
from math import dist
from statistics import mean
from random import Random

def quantile(values,q):
    if not values: return None
    x=sorted(values); p=(len(x)-1)*q; a=int(p); b=min(a+1,len(x)-1)
    return x[a]+(p-a)*(x[b]-x[a])

def metrics(run):
    ds=run.get('diagnostics',[]); ms=[d['decision_ms'] for d in ds]
    dt=run.get('settings',{}).get('dt',.25)
    route=run.get('route',[])
    length=sum(dist(a,b) for a,b in zip(route,route[1:]))
    cs=[d['clearance'] for d in ds if d.get('clearance') is not None]
    return {'status':run['status'], 'ship_collision':any(d['ship_collision'] for d in ds),
            'land_collision':any(d['land_collision'] for d in ds),
            'min_ship_clearance':min(cs) if cs else None,
            'domain_time_s':sum(c<1. for c in cs)*dt,
            'detour_ratio':run.get('distance',0)/length if length else None,
            'overrides':sum(d['override'] for d in ds),
            'override_rate':mean(d['override'] for d in ds) if ds else None,
            # None = filter off, so feasibility was never evaluated (not "zero infeasible").
            'no_feasible_actions':None if any(d['no_feasible_action'] is None for d in ds) else sum(d['no_feasible_action'] for d in ds),
            'solver_failures':sum(d.get('solver_failed',False) for d in ds),
            'decision_mean_ms':mean(ms) if ms else None,
            'decision_p95_ms':quantile(ms,.95),'decision_p99_ms':quantile(ms,.99),
            'deadline_misses':sum(v>1000*dt for v in ms)}

def paired_interval(a,b,seed=0,samples=2000):
    if not a or set(a)!=set(b):
        raise ValueError('Require complete matching scenario IDs')
    diffs=[a[k]-b[k] for k in sorted(a)]; rng=Random(seed)
    draws=[mean(rng.choices(diffs,k=len(diffs))) for _ in range(samples)]
    return {'difference':mean(diffs),'low':quantile(draws,.025),'high':quantile(draws,.975),'pairs':len(diffs)}
def point_segment_distance(p,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    f=max(0.,min(1.,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy))) if dx*dx+dy*dy else 0.
    return dist(p,(a[0]+f*dx,a[1]+f*dy))

def land_clearance(sea,p,radius):
    # Signed: negative inside land. SeaMap.minimum_clearance saturates at 0, so it cannot be used here.
    from shapely.geometry import Point
    x,y=p; x0,y0,x1,y1=sea.bounds
    values=[x-x0,y-y0,x1-x,y1-y]; point=Point(p)
    for polygon in sea.land:
        values.append(-polygon.boundary.distance(point) if polygon.contains(point) else polygon.distance(point))
    return min(values)-radius

def geometry_metrics(run):
    from shipnav.maps import SeaMap
    from math import pi, atan2
    frames=run.get('frames',[]); route=run.get('route',[])
    if not frames:
        return {}
    sea=SeaMap.from_dict(run['map'])
    errors=[min((point_segment_distance(f['position'],a,b) for a,b in zip(route,route[1:])),default=0.) for f in frames]
    land=[land_clearance(sea,f['position'],run['settings']['radius']) for f in frames]
    ds=run.get('diagnostics',[]); violations=0; exposure=0.
    # Marine runs start at rest on the first route leg's bearing (east if already arrived),
    # matching simulation.run_episode.
    heading=atan2(route[1][1]-route[0][1],route[1][0]-route[0][0]) if len(route)>1 else 0.; speed=0.
    for a,b,d in zip(frames,frames[1:],ds):
        dt=b['t']-a['t']
        if d.get('clearance') is not None and d['clearance']<1.: exposure+=dt
        if 'heading' in d:
            turn=(d['heading']-heading+pi)%(2*pi)-pi
            violations+=int(abs(turn)> .35*dt+1e-6 or abs(d['speed']-speed)>.2*dt+1e-6)
            heading,speed=d['heading'],d['speed']
    return {'cross_track_mean_m':mean(errors),
            'cross_track_max_m':max(errors),
            'sampled_land_clearance_m':min(land),
            'domain_time_s':exposure,'dynamics_violations':violations}

def hierarchical_interval(a,b,seed=0,samples=2000):
    # a and b: training seed -> scenario ID -> scalar. Pair both levels.
    if not a or set(a)!=set(b): raise ValueError('Training seeds differ')
    for key in a:
        if not a[key] or set(a[key])!=set(b[key]): raise ValueError('Scenario IDs differ')
    rng=Random(seed); keys=sorted(a); draws=[]
    for _ in range(samples):
        means=[]
        for key in rng.choices(keys,k=len(keys)):
            ids=sorted(a[key]); chosen=rng.choices(ids,k=len(ids))
            means.append(mean(a[key][i]-b[key][i] for i in chosen))
        draws.append(mean(means))
    return {'difference':mean(mean(a[k][i]-b[k][i] for i in a[k]) for k in keys),
            'low':quantile(draws,.025),'high':quantile(draws,.975),
            'training_seeds':len(keys)}
```

Create `rebuild/src/shipnav/benchmark.py` with:

```python
from pathlib import Path
import json
from shipnav.service import execute
from shipnav.scenarios import scenario_hash
from shipnav.planning import NoPath
from shipnav.metrics import metrics, geometry_metrics
from shipnav.scale import map_options

VARIANTS={
 'sarl_reference':dict(policy_name='sarl'),
 'sarl_theta':dict(policy_name='sarl',planner='theta'),
 'sarl_no_goals':dict(policy_name='sarl',global_goals=False),
 'sarl_filtered':dict(policy_name='sarl',filtered=True),
 'sarl_cv_filter':dict(policy_name='sarl',filtered=True,uncertainty=False),
 'sarl_degraded':dict(policy_name='sarl',filtered=True,observation={'noise':.1,'delay':.5,'dropout':.1}),
 'orca_reference':dict(policy_name='orca'),
 'orca_filtered':dict(policy_name='orca',filtered=True),
 'direct_reference':dict(policy_name='direct'),
 'marine_sarl':dict(policy_name='sarl',dynamics='marine',filtered=True),
 'marine_mpc':dict(policy_name='mpc',dynamics='marine',filtered=True),
 'marine_mpc_unfiltered':dict(policy_name='mpc',dynamics='marine',filtered=False)}

def benchmark(scenarios,model_dir,output,variants=VARIANTS,runner=execute):
    output=Path(output); output.mkdir(parents=True,exist_ok=True); rows=[]
    for scenario in scenarios:
        identity=scenario_hash(scenario)
        (output/f'{identity}-scenario.json').write_text(json.dumps(scenario,allow_nan=False))
        for name,config in variants.items():
            try:
                run=runner(scenario['map'],scenario['start'],scenario['goal'],model_dir,
                           scenario=scenario,**{**map_options(scenario['map']),**config})
            except NoPath as error:
                run={'status':'planning_failure','error':str(error)}
            except Exception as error:
                run={'status':'error','error':f'{type(error).__name__}: {error}'}
            run.update(scenario_hash=identity,variant=name,variant_config=config)
            (output/f'{identity}-{name}.json').write_text(json.dumps(run,allow_nan=False))
            rows.append({'scenario_hash':identity,'variant':name,**metrics(run),**geometry_metrics(run)})
    (output/'metrics.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    return rows
```

- [ ] **Step 4: Run the same test command.** Expected: PASS. Inspect failures before proceeding.

- [ ] **Step 5: Review the diff and commit.**

```bash
git add rebuild/src/shipnav/metrics.py rebuild/src/shipnav/benchmark.py rebuild/tests/test_metrics.py rebuild/tests/test_geometric_metrics.py
git commit -m "feat: score paired benchmark runs with geometry metrics"
```

### Task 12: Freeze the protocol and record pre-retraining baselines

Plan 05 compares new policies against these numbers, so complete this task before any training run.

- [ ] Freeze a small real-map family with `make_scenario(..., corridor=reference_route)` on `to_model(map, PROFILES['harbour_craft'])`: Ubin in dev/calibration, Southern Islands held out as the geographic generalization test. `benchmark()` reads each scenario's planner settings from its map (`scale.map_options`), so no per-variant configuration is needed. Report real-map metrics in metres and seconds using `scale.units`; pooled model-unit metrics are dimensionless and comparable across maps.
- [ ] Stratify inference-only and full decision timing by device/platform, cold versus warmed runs, traffic count and dynamics. For asynchronous CUDA timing, synchronize before/after measured work. Report p95/p99/deadline misses with sample counts.
- [ ] Freeze a `benchmark_protocol.json` before the held-out run: scenario IDs/hashes from `scenarios/splits.json`, split boundaries, seeds, primary metrics, variant pairs, the fixed reference route per scenario, runtime hardware and stopping rules. Set the success/collision targets that `2026-09-27-rebuild-decisions.md` §7 leaves open now, before seeing results. Ten seeds remain a smoke test. Begin with at least 100 distinct held-out scenarios across the named families (the current frozen test split has 11 canonical encounters; generate the rest with `tools/freeze_scenarios.py`), then choose additional sample size from pilot uncertainty/precision, not a desired significance result.
- [ ] Run `benchmark()` on the held-out split with every `VARIANTS` entry using the supplied SARL checkpoint. At minimum record `sarl_reference` (unfiltered) against `sarl_filtered`, `orca_reference` against `orca_filtered`, and `marine_mpc` against `marine_mpc_unfiltered`. Keep every trace and `metrics.json` under `results/` (ignored) and copy the protocol and summary into a tracked evidence file.
- [ ] Use scenario-paired bootstrap (`paired_interval`) for fixed policies. For learned policies (plan 05), keep independent training seeds (at least 3; preferably 5) and use `hierarchical_interval`; do not treat thousands of rollouts from one checkpoint as independent models. Publish seed-level results and intervals alongside pooled values. A failed/missing run is never deleted from only one arm.
- [ ] Isolate one main factor per comparison: A* vs Theta*, goals on/off, filter on/off, constant-velocity vs uncertainty-aware prediction, exact vs degraded sensing, SARL vs MPC **under the same dynamics/filter/observations**. The variant table provides defaults, not permission to compare mismatched controls.
- [ ] Update `rebuild/README.md`: CLI and GUI commands; vessel profiles and the similarity scaling (what dt, speeds and marine limits a profile implies, and that results/JSON are in model units with `model_scale` metadata); click workflow and both control rows; export formats and the CSV column meanings; experiment and benchmark commands and metric definitions; the no-neighbour direct fallback; constant-velocity traffic assumption; limits of encounter-rule coverage and the selected marine dynamics; model hash interpretation; results location; where the baseline benchmark evidence lives. This is the lightweight user manual, not an academic report.
- [ ] Commit the README and the tracked baseline evidence once the native-window checklist in Task 10 passes.
