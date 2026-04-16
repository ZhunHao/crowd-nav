# System Design — CrowdNav-DIP Maritime Navigation Demo

> Companion to the implementation plan at `~/.claude/plans/added-slides-md-which-is-logical-grove.md`.
> Source requirements: [slides.md](../slides.md) (professor's briefing, slides 13–16).

---

## 1. Requirements

### 1.1 Functional

| ID | Requirement | Slide |
|----|-------------|-------|
| F1 | Reproduce the provided CrowdNav DRL training/testing pipeline and export a simulation video | 13 |
| F2 | Allocate local goals (waypoints) without overlapping across DRL phases; spawn dynamic obstacles per-phase | 13 |
| F3 | Load a static land map and render it as background + treat it as static obstacles in simulation & collision | 13 |
| F4 | Integrate a global path planner (Theta* or equivalent) that emits waypoints consumed by the DRL policy | 13 |
| F5 | GUI with buttons: Load static map, Load DRL model, Set start, Set goal, Add static obstacles, Visualize, Export | 14 |
| F6 | Support the existing policies (ORCA, CADRL, LSTM-RL, SARL, OM-SARL) without retraining | 9–11 |

### 1.2 Non-Functional

| ID | Requirement | Target |
|----|-------------|--------|
| NF1 | **Latency** — GUI stays responsive during rollout | UI thread ≥30 FPS; rollout on worker thread |
| NF2 | **Throughput** — global planner returns waypoints in reasonable time for demo | < 2 s for a 200×200 grid (single voyage) |
| NF3 | **Determinism** — repeatable demos | Seedable RNG end-to-end |
| NF4 | **Portability** — runs on student laptops | Python 3.10, CPU-only path must work |
| NF5 | **Zero-backend** — no server | All in-process; file-based artifacts |
| NF6 | **Model compatibility** — reuse `data/output_trained/rl_model.pth` | Do not change SARL/CADRL/LSTM network signatures |

### 1.3 Constraints

- **Team size / timeline** — 2–3 students per group, ~1 semester. Favour small, understandable components over industrial robustness.
- **Existing stack** — Python 3.10, PyTorch, OpenAI Gym 0.15.7 (pinned), matplotlib, Python-RVO2 (vendored).
- **GUI framework** — PyQt5 (recommended by slide 16); matplotlib-Qt backend exists and is battle-tested → embed rather than re-render.
- **No cloud** — single-machine desktop demo; no network services.
- **Policy network is frozen** — we must supply state in the shape SARL expects (`self_state[:, :7]`, human observable states per row), so all changes live in env / planner / GUI.

---

## 2. High-Level Design

### 2.1 Component diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                         PyQt GUI process                              │
│                                                                        │
│  ┌─────────────────┐   signals   ┌────────────────────────────────┐  │
│  │  MainWindow     │◀───────────▶│  SimController (facade)         │  │
│  │  - QToolBar     │             │  - load_map / load_model        │  │
│  │  - FigureCanvas │             │  - set_start / set_goal         │  │
│  │    (matplotlib) │             │  - run / export                 │  │
│  └────────┬────────┘             └───────┬────────────────────────┘  │
│           │ mpl_connect                    │ owns                       │
│           │ click→world coords             │                            │
│                                            ▼                            │
│                            ┌──────────────────────────┐                │
│                            │  SimWorker (QThread)      │                │
│                            │  drives waypoint loop     │                │
│                            └───┬──────────┬───────────┘                │
│                                │          │                            │
│                                ▼          ▼                            │
│                ┌─────────────────────┐  ┌──────────────────────────┐  │
│                │  GlobalPlanner      │  │  Gym env (CrowdSim)       │  │
│                │  Theta* on grid     │  │  + Robot + Humans + map   │  │
│                │  returns [w1..wn]   │  │  reward / step / render   │  │
│                └──────────┬──────────┘  └─────────┬────────────────┘  │
│                           │                        │                    │
│                           ▼                        ▼                    │
│                ┌─────────────────────────────────────────────────┐    │
│                │  StaticMap (occupancy grid)                      │    │
│                │  shared source of truth for planner + env + GUI  │    │
│                └─────────────────────────────────────────────────┘    │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘
                      │ reads                    │ writes
                      ▼                          ▼
                ┌──────────────┐          ┌────────────────────────┐
                │ maps/*.png   │          │ exports/*.mp4, *.csv   │
                │ models/*.pth │          │                         │
                │ *.config     │          │                         │
                └──────────────┘          └────────────────────────┘
```

### 2.2 Data flow (one voyage)

```
1. User clicks "Load static map"   → StaticMap built from PNG/NPY
2. User clicks "Load DRL model"    → Policy loaded, weights into SARL
3. User clicks on canvas (start)   → env.robot_initx/y set
4. User clicks on canvas (goal)    → env.robot_goalx/y set
5. User clicks "Visualize"         → SimWorker.start()
     └─ GlobalPlanner.plan(start, goal, StaticMap) → [w1..wn]
     └─ for wi in waypoints:
          env.robot_goalx/y = wi
          env.light_reset(phase, test_case)
          until reached(wi) or timeout:
              action = policy.predict(JointState(robot_state, humans_obs))
              ob, r, done, info = env.step(action)
              yield frame  ──▶ FigureCanvas.update()
6. User clicks "Export"            → save MP4 / CSV of waypoints + telemetry
```

### 2.3 API contracts (internal, Python)

```python
# crowd_sim/envs/utils/static_map.py
class StaticMap:
    grid: np.ndarray          # uint8, 1 = blocked, 0 = free
    resolution: float         # metres per cell
    origin: tuple[float, float]
    @classmethod
    def from_file(cls, path: str) -> "StaticMap": ...
    def is_free(self, x: float, y: float) -> bool: ...
    def line_of_sight(self, a: XY, b: XY) -> bool: ...
    def to_obstacles(self) -> list[StaticObstacle]: ...
    def render_background(self, ax) -> None: ...

# crowd_nav/planner/theta_star.py
class ThetaStar:
    def __init__(self, static_map: StaticMap, inflation: float = 0.5): ...
    def plan(self, start: XY, goal: XY) -> list[XY]: ...

# gui/controllers/sim_controller.py
class SimController(QObject):
    frame_ready = pyqtSignal(np.ndarray)   # rendered RGB frame
    episode_done = pyqtSignal(dict)         # {"success": bool, "reward": float, ...}
    def load_map(self, path: str) -> None: ...
    def load_model(self, path: str, policy_name: str) -> None: ...
    def set_start(self, x: float, y: float) -> None: ...
    def set_goal(self, x: float, y: float) -> None: ...
    def add_obstacle(self, shape: dict) -> None: ...
    def run(self) -> None: ...        # non-blocking; emits signals
    def export(self, path: str) -> None: ...
```

### 2.4 Storage

Everything is file-based — no DB:

| Artifact | Path | Format |
|----------|------|--------|
| Land maps | `maps/*.png` or `maps/*.npy` | 8-bit grid, 0=water, 255=land |
| Trained models | `crowd_nav/data/output_trained/rl_model.pth` | PyTorch state dict |
| Configs | `crowd_nav/configs/*.config` | INI (existing) |
| Exports | `exports/*.mp4`, `exports/*.csv` | MP4 + waypoint CSV |
| Planner cache (optional) | `.cache/<map_hash>/` | `.npz` (numpy arrays of precomputed distance fields) |

---

## 3. Deep Dive

### 3.1 Data models

**Occupancy grid** (`StaticMap`):
- `grid[i, j] == 1` means cell blocked (land). `i` is row (y), `j` is column (x).
- `resolution` chosen so domain spans existing env bounds (`-15..15` per [crowd_sim.py:591](../crowd_sim/envs/crowd_sim.py)); e.g. `resolution=0.1m` → 300×300 grid.
- **Inflation**: during planning, grid is dilated by `robot_radius + safety_margin` so Theta* is clearance-aware. This is a well-known trick — prevents needing per-step clearance checks in the planner.

**Waypoint list**:
- Always terminates with the final goal.
- Empty list is invalid; the planner returns `[goal]` for trivial straight-line cases.
- Each waypoint is `(x, y)` in world coordinates, not grid cells.

**Joint state** (unchanged from existing SARL): `FullState(robot) + [ObservableState(h) for h in humans]`. Only the `gx, gy` fields vary between waypoints; no policy retraining needed.

### 3.2 Key algorithms

**Theta* — outline** (port from `zhm-real/PathPlanning`, MIT):
```
open_set = {start}
g[start] = 0; parent[start] = start
while open_set:
    s = argmin_{n in open_set} f[n]   # f = g + h (h = euclidean)
    if s == goal: return reconstruct(parent, goal)
    for neighbour s' of s:
        if line_of_sight(parent[s], s'):
            # Path-2: skip s, go directly
            cost = g[parent[s]] + dist(parent[s], s')
            if cost < g[s']: g[s'] = cost; parent[s'] = parent[s]
        else:
            # Path-1: classic A* relaxation
            cost = g[s] + dist(s, s')
            if cost < g[s']: g[s'] = cost; parent[s'] = s
```
Line-of-sight uses Bresenham through `StaticMap.grid`. Result: any-angle paths (not stuck on the 8-connected grid), fewer waypoints, smoother ship trajectories.

**Goal allocator** (WP-2):
```
def sample_unused_position(used_regions, bounds, min_dist):
    for _ in range(MAX_TRIES):
        p = uniform(bounds)
        if all(dist(p, u) >= min_dist for u in used_regions):
            if static_map.is_free(*p):
                return p
    raise RuntimeError("could not sample free position")
```
`used_regions` includes all prior waypoints + start + goal. Stops the "two humans walking the same corridor" artifact that breaks long voyages.

### 3.3 Threading model

| Thread | Responsibility |
|--------|---------------|
| Qt main (UI) | Button clicks, canvas events, matplotlib draw |
| `SimWorker` QThread | Waypoint loop: planner + rollout; emits `frame_ready` per step |
| None else | Single-machine, CPU-bound, no concurrency beyond this |

Rollout step is <50 ms on CPU for SARL with 5 humans — `frame_ready` at 20 Hz is plenty smooth. If we need more, downsample (emit every Nth frame).

### 3.4 Error handling

| Failure | Surface to user as | Recovery |
|---------|-------------------|----------|
| Invalid map file | QMessageBox — "Cannot load map: <path>" | User picks another file |
| Model/policy mismatch (e.g. input dim) | QMessageBox with diff | User picks matching config/model |
| Start/goal on land | QMessageBox, reject click | User clicks free cell |
| Planner returns no path | QMessageBox "No path found" | User edits obstacles |
| Episode timeout | Status bar "Timeout — replanning" | Replan from current pos (one retry) |
| Collision | Record in telemetry, continue | No auto-recovery; episode marked failed |

No silent failures. All paths either succeed or display a message — per coding-style "never silently swallow errors".

### 3.5 Configuration

Extend [env.config](../crowd_nav/configs/env.config) with maritime-specific keys:
```ini
[sim]
static_obs = true
land_map_path = maps/default.png
map_resolution = 0.1
map_origin_x = -15.0
map_origin_y = -15.0

[planner]
algorithm = theta_star          # theta_star | a_star
inflation_radius = 0.5
waypoint_simplify = true         # apply RDP reduction
goal_tolerance = 0.3

[goal_allocator]
min_inter_waypoint_dist = 1.0
```

All knobs have defaults in code so old configs keep working.

---

## 4. Scale and Reliability

Short version: **we don't need to scale**. This is a demo app. But some knobs matter:

| Concern | Current approach | When to revisit |
|---------|-----------------|-----------------|
| Planner speed on huge maps | Single-threaded Theta* + inflation cache | If maps exceed ~1000×1000 cells, switch to JPS or hierarchical planner |
| Policy inference speed | CPU path OK for SARL-with-5-humans | If humans ≥30 or frame rate drops, add `--gpu` path (already supported in existing test.py) |
| GUI responsiveness | `QThread` worker + signal-based frame push | Add frame drop / backpressure if canvas paint can't keep up |
| Memory | Grid + matplotlib fig; ~100 MB | Not an issue; revisit only if we add multi-voyage batch mode |

**Reliability for a demo**:
- **Determinism**: expose a seed in `env.config` (`random_seed`). Reuses Python `random.seed`, `np.random.seed`, `torch.manual_seed`. Without determinism, students can't debug.
- **Crash containment**: SimWorker wraps the rollout in try/except so a planner or policy exception shows a dialog instead of killing the app.
- **Telemetry**: CSV export includes per-step `(t, robot_xy, human_xys, action, reward, current_waypoint)`. Enough to reproduce any run offline.
- **No monitoring/alerting**: N/A for desktop demo.

---

## 5. Trade-offs

### 5.1 Planner choice — Theta* vs A* vs RRT*

| Dimension | A* (8-connected) | Theta* | RRT* |
|-----------|-----------------|--------|------|
| Path optimality | Suboptimal (angle-restricted) | Near-optimal any-angle | Asymptotically optimal |
| Implementation difficulty | Low | Medium (adds line-of-sight) | High |
| Output shape | Dense zig-zag waypoints | Sparse, smooth waypoints | Jittery, needs smoothing |
| Fit for DRL sub-goals | Too many goals, churn | Few clean waypoints — ideal | Overkill; needs post-processing |
| Slide recommendation | — | ✅ explicitly named | — |

**Pick Theta***. Professor named it; it produces the right *number* and *quality* of waypoints for the DRL local planner.

### 5.2 GUI — PyQt vs Tkinter vs web

| Dimension | PyQt5 | Tkinter | Web (Flask+JS) |
|-----------|-------|---------|----------------|
| Slide recommendation | ✅ | — | — |
| matplotlib embedding | First-class (`FigureCanvasQTAgg`) | Possible but clunky | Via WebSocket — heavy |
| Packaging | `pyinstaller` works well | Trivial | Two processes |
| Student familiarity | Medium | High | Variable |

**Pick PyQt5** — aligns with slide 16, best matplotlib story.

### 5.3 Static obstacle representation — inflated grid vs analytic shapes

| Dimension | Grid only | Analytic (rect/circle) only | Grid + analytic hybrid |
|-----------|-----------|----------------------------|------------------------|
| Planner cost | Cheap is-free / line-of-sight | Pairwise polygon checks | Grid for planner, shapes for render |
| Render cost | imshow bitmap | Draw patches | Both |
| Fidelity for "land" (irregular) | High | Poor | High |
| Fidelity for discrete "buoy" (simple) | Blocky | High | High |

**Hybrid**: store canonical occupancy as grid; also keep the existing `self.static_obstacles` list for user-added rect/circle shapes. `StaticMap.to_obstacles()` can flatten both into a uniform iterable for the env's collision check. Zero duplication because the existing `StaticObstacle` already renders as matplotlib patches.

### 5.4 Sub-goal injection — new API vs piggyback on `light_reset`

| Dimension | New mid-episode goal setter | Reuse `light_reset` (exists) |
|-----------|----------------------------|------------------------------|
| Code surface | +~40 lines, new path to test | 0 — already works |
| Agent state continuity | Can preserve velocity | Partially resets (fine for demo) |
| Risk to existing tests | Non-zero | Zero |

**Pick reuse**. Evidence: [test.py:96-145](../crowd_nav/test.py) already loops `env.robot_goalx = ...; env.light_reset()`. We're formalising an existing pattern, not inventing one. If the reset-on-waypoint seam becomes visible in videos, *then* add a no-reset goal setter.

### 5.5 What we'd revisit as the system grows

| Growth trigger | Revisit |
|---------------|---------|
| Real AIS data (8M records, slide 3) | Replace ORCA humans with AIS-replay humans; needs new `Human` subclass |
| Multi-agent training on maritime data | Retrain SARL with land-aware state; current plan freezes weights |
| Web demo / multi-user | Swap PyQt for a React + FastAPI split; move sim to server |
| Publishable experiments | Add `experiments/` framework with seeded sweeps and stats; current CSV is ad-hoc |
| Ship dynamics (turn-rate, acceleration limits) | Subclass `Robot`, add constrained action space; the professor's slide 2 calls this out — in scope for an advanced group |

---

## 6. Assumptions

1. Trained model in `data/output_trained/` was trained with 5 humans — GUI defaults match this.
2. Land map axes align with the env's existing world coords (`-15..15` in matplotlib).
3. Waypoints are reachable in straight-ish segments; we rely on Theta* + DRL local avoidance, not on reactive replanning except as a fallback.
4. PyQt5 is available on the school's machines (stated as recommended on slide 16).
5. The professor's "static obstacles" on slide 14 means user-drawn shapes (e.g. extra buoys) *on top of* the loaded land map — not a replacement.

---

## 7. Open Questions

1. **Map format** — PNG vs NPY vs ROS-style `.yaml + .pgm`? PNG + companion `.json` for resolution/origin is the simplest, but `.npy` skips image decode. *Proposal: support both; detect by extension.*
2. **Coordinate origin** — should the GUI let users pan/zoom, or fix the view to env bounds? *Proposal: fix for v1; pan/zoom is a polish task.*
3. **Replanning cadence** — replan every N seconds vs only on waypoint failure? *Proposal: only on failure; simpler and matches the "global + local" decomposition on slide 8.*
4. **Export format** — MP4 only, or MP4 + CSV + PNG trajectory overlay? *Proposal: all three — the CSV is cheap and makes debugging / reporting far easier.*

These are all cheap to defer; none block WP-1 or WP-3.
