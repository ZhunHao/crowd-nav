# WP-3 / R3 — Functional Static Obstacles (StaticMap + Collision + ORCA) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Plan destination:** copy this file to `docs/superpowers/plans/2026-04-17-wp3-static-map.md` before starting Task 1. It mirrors the WP-2 plan's structure so downstream WP-4 / WP-5 plans can link to it.

**Goal:** Turn the currently-decorative gray rectangles in `exports/baseline.mp4` into a load-bearing part of the simulation. Ship (a) a `StaticMap` class with an `is_free(x, y, margin)` query, (b) wire it into the WP-2 `GoalAllocator.allocate_waypoints` / `allocate_human_positions` seams so waypoints and human spawn points never land inside obstacles, (c) add robot-vs-obstacle collision detection in `CrowdSim.step()` (episode ends with `Collision`), and (d) inject obstacles into the RVO2 simulator so ORCA-driven humans route around walls. Ships the `[static_map]` config section. Preserves R7 (frozen SARL weights / observation shape).

**Architecture:** Pure geometry primitive (`crowd_sim/envs/utils/static_map.py`) + thin wiring. `StaticMap` is a frozen dataclass — no I/O, no gym, no matplotlib. WP-2's `GoalAllocator.sample_unused_position` already accepts an `is_free` callable, so Tier A is zero-change to the allocator module — only the call sites. Tier B adds one branch to `CrowdSim.step()` alongside the existing human-collision block. Tier C extends `ORCA.predict()` with a one-shot `sim.addObstacle() + sim.processObstacles()` on fresh-sim creation, driven by a new `ORCA.set_static_obstacles(polygons)` setter that `CrowdSim.reset()` / `light_reset()` calls after humans are placed. No change to SARL / CADRL / LSTM-RL observation tensors — policies still see `[human.get_observable_state() for human in self.humans]`, because the policy is frozen (R7).

**Tech Stack:** Python 3.10, `rvo2` (already a dependency, see WP-1 `setup_env.sh`), `configparser` (stdlib), pytest + the `unit` / `integration` / `slow` / `smoke` markers introduced in WP-1 `pyproject.toml`.

**Scope (what this plan delivers — and does NOT):**

| In scope (R3) | Out of scope (later WPs) |
|---|---|
| `StaticMap(obstacles, margin)` with `is_free(x, y, margin=None) -> bool` for rects + circles | Theta\* any-angle planner that routes waypoints *around* obstacles (WP-4 / R4) — we still use WP-2's straight-line `_straight_line_source` as the default waypoint source |
| `StaticMap.from_static_obstacles(list[dict], margin)` classmethod + `rect_vertices_ccw(obs)` helper for RVO2 input | Retraining SARL with a static-map channel in the observation (R7 forbids) |
| `[static_map]` config section (`enabled`, `margin`) + `PhaseConfig.static_map` accessor | GUI widgets for editing obstacles (WP-5 / R5) |
| Inject `is_free=static_map.is_free` into `GoalAllocator.allocate_waypoints` from `crowd_nav/test.py` | Changes to `generate_square_crossing_human`, `generate_static_human`, `mixed` rule |
| Route `generate_random_human_position('circle_crossing')` through `GoalAllocator.allocate_human_positions(..., is_free=...)` | New render paths for obstacles beyond the existing gray-rectangle draw |
| Tier B: robot collision vs obstacles in `CrowdSim.step()` → `Collision()` + `done=True` | Configurable obstacle layouts — this WP keeps the hardcoded 170-rect cordon from [crowd_sim.py:410-444](../../crowd_sim/envs/crowd_sim.py) and only makes it functional |
| Tier C: `sim.addObstacle(...)` + `sim.processObstacles()` in `ORCA.predict()` + `set_static_obstacles` setter | Dynamic / moving obstacles |
| Backward compat: `[static_map] enabled = false` falls back to today's "visual only" behaviour | Re-tuning `[goal_allocator] num_waypoints` for the new navigable area (WP-4 Theta\* will re-tune) |

**Non-goals:**
- No new info / reward class — Tier B reuses the existing `Collision` class from `crowd_sim/envs/utils/info.py` and the existing `self.collision_penalty`.
- No refactor of the `static_obstacles` hardcoded pattern in `reset()` — `StaticMap.from_static_obstacles(env.static_obstacles, ...)` wraps whatever is there.
- No change to `generate_square_crossing_human` / `generate_static_human` / `mixed` dispatcher branches — only `circle_crossing` is rerouted, consistent with WP-2.
- No rewiring of `ORCA` policy for the *robot* — in the R1 baseline the robot runs SARL (not ORCA), so ORCA-obstacle awareness only affects humans.
- No probabilistic sampling beyond WP-2's uniform-over-bounds + rejection loop.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `crowd_sim/envs/utils/static_map.py` | **create** | Frozen `Obstacle(kind, cx, cy, w, h, r)` and `StaticMap(obstacles, margin)` dataclasses. Exposes `is_free(x, y, margin=None)`, `from_static_obstacles(list[dict], margin)` classmethod, and `rect_vertices_ccw(obs)` helper. No I/O, no gym. |
| `crowd_sim/envs/utils/phase_config.py` | **modify** (append) | Add `StaticMapParams(enabled, margin)` frozen dataclass and `PhaseConfig.static_map` field with `_read_static_map(cp)` helper. Zero change to existing fields. Defaults: `enabled = True, margin = 0.5`. |
| `crowd_nav/configs/env.config` | **modify** (append) | Append `[static_map]\nenabled = true\nmargin = 0.5`. |
| `crowd_nav/data/output_trained/env.config` | **modify** (append) | Same `[static_map]` section so `run_baseline.sh` picks up the wiring. |
| `crowd_nav/test.py` | **modify** (~line 1108-1140) | After `env.reset`: build `static_map = StaticMap.from_static_obstacles(env.static_obstacles, phase_cfg.static_map.margin) if phase_cfg.static_map.enabled else None`; pass `is_free=static_map.is_free if static_map else None` into `allocator.allocate_waypoints`. |
| `crowd_sim/envs/crowd_sim.py` | **modify** in 4 places: (a) `reset()` + `light_reset()` — after `self.static_obstacles` is built (or unchanged) and after humans are placed, build `self.static_map` and call `human.policy.set_static_obstacles(...)` for every ORCA human. (b) `generate_random_human_position('circle_crossing')` — route through `GoalAllocator.allocate_human_positions(..., is_free=self.static_map.is_free)`. (c) `step()` — after computing `end_position`, check `self.static_map.is_free(*end_position, margin=self.robot.radius)`; on False set `info = Collision()` and `done = True`. | Wires StaticMap into the simulator without changing any policy-facing tensor shape. |
| `crowd_sim/envs/policy/orca.py` | **modify** | Add `self.static_obstacle_polygons: list[list[tuple[float,float]]] = []` init + `set_static_obstacles(polygons)` setter that invalidates `self.sim`. In `predict()`, on fresh-sim creation, loop `self.sim.addObstacle(poly)` + call `self.sim.processObstacles()` once. Delete the *"In this work, obstacles are not considered."* comment. |
| `tests/test_static_map.py` | **create** | Unit tests: inside/outside rect with + without margin; circle; `from_static_obstacles` roundtrip; `rect_vertices_ccw` CCW ordering; frozen-dataclass immutability; raises on unknown `type`. |
| `tests/test_allocator_respects_static_map.py` | **create** | Integration: `allocate_waypoints` + `allocate_human_positions` with a concrete `StaticMap` — every returned point passes `is_free(margin=robot_radius)`. |
| `tests/test_crowd_sim_static_collision.py` | **create** | Tier B regression: robot positioned at `(0, 4.9)` with `static_obstacle rect(cx=0,cy=5,w=20,h=1)` + velocity `(0, 0.3)` → `env.step(...)` returns `Collision()` + `done=True`. |
| `tests/test_orca_obstacles.py` | **create** | Tier C regression: ORCA human at `(-5, 0)`, goal `(5, 0)`, obstacle rect `(cx=0,cy=0,w=2,h=4)`; after 50 `predict() + step()` rounds the trajectory never enters the AABB. |
| `tests/test_phase_config.py` | **modify** (append) | Add cases for the new `[static_map]` parsing + real-file regression for the checked-in `env.config`. |
| `tests/test_waypoint_integration.py` | **modify** (extend) | After the baseline run, parse the `Allocated N waypoints (start=..., goal=...)` log line, rebuild the 10 straight-line waypoints, and assert `StaticMap(env.static_obstacles, margin=0.5).is_free(w)` for every waypoint. |

**Boundaries:** `static_map.py` knows nothing about `gym`, `configparser`, `rvo2`, or `CrowdSim`. `phase_config.py` knows only `configparser` and dataclasses. `orca.py` knows `rvo2` but not `StaticMap` — it accepts a pre-flattened polygon list. `crowd_sim.py` is the only module that imports both `StaticMap` and the allocator — it orchestrates.

---

## Prerequisites

Before starting, verify WP-1 and WP-2 are complete and the `navigate` conda env is active:

```bash
cd /Users/zhunhao/Documents/Projects/crowdnav-dip
git tag --list | grep -q '^r1-baseline$'         && echo "WP-1 ok" || echo "WP-1 missing"
git tag --list | grep -q '^wp2-goal-allocator$'  && echo "WP-2 ok" || echo "WP-2 missing"
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate navigate
python -c "from crowd_sim.envs.utils.goal_allocator import GoalAllocator; print('allocator importable')"
python -c "import rvo2; print('rvo2 importable')"
pytest -q -m "not slow" tests/                     # baseline green
ls crowd_nav/data/output_trained/rl_model.pth      # weights present (R7)
which ffmpeg                                        # required for integration test
```

Expected: both tags present, imports OK, 32+ tests green, `rl_model.pth` exists, `ffmpeg` on PATH.

---

## Task 1: `StaticMap` geometry primitives (failing test first)

**Files:**
- Create: `tests/test_static_map.py`
- Create: `crowd_sim/envs/utils/static_map.py`

**Why first:** every other Tier-A/B/C wiring reduces to calling `is_free(x, y)` or iterating `rect_vertices_ccw(obs)`. Building the module test-first pins the contract.

- [ ] **Step 1: Write the failing tests** — cover inside-rect, margin, outside, circle, `from_static_obstacles` roundtrip, `rect_vertices_ccw` CCW order, unknown `type` raises `ValueError`, frozen immutability. All tests marked `@pytest.mark.unit`.
- [ ] **Step 2:** `pytest tests/test_static_map.py -v` → `ModuleNotFoundError`.
- [ ] **Step 3: Implement** `crowd_sim/envs/utils/static_map.py`:
  - `Obstacle` frozen dataclass with `kind: Literal["rect", "circle"], cx, cy, w, h, r` (unused fields = 0.0).
  - `StaticMap(obstacles: tuple[Obstacle, ...], margin: float = 0.0)` frozen dataclass.
  - `is_free(self, x, y, margin=None)` — default to `self.margin`; returns `False` iff any obstacle contains `(x,y)` expanded by `margin`. Rect test: `abs(x-cx) <= w/2 + m and abs(y-cy) <= h/2 + m`. Circle test: `hypot(x-cx, y-cy) <= r + m`.
  - `@classmethod from_static_obstacles(cls, obs_list, margin=0.0)` — converts env's `{'type','cx','cy','w','h'}` / `{'type':'circle','cx','cy','r'}` dicts into `Obstacle` tuples. Raises `ValueError` on unknown `type`.
  - Module-level `rect_vertices_ccw(obs: Obstacle) -> list[tuple[float, float]]` returning `[(cx-w/2,cy-h/2),(cx+w/2,cy-h/2),(cx+w/2,cy+h/2),(cx-w/2,cy+h/2)]` — CCW order RVO2 requires.
- [ ] **Step 4:** `pytest tests/test_static_map.py -v` → PASS.
- [ ] **Step 5: Commit** — `feat(wp3): StaticMap primitives — is_free + rect_vertices_ccw`.

---

## Task 2: Extend `PhaseConfig` with `StaticMapParams`

**Files:**
- Modify: `crowd_sim/envs/utils/phase_config.py`
- Modify: `tests/test_phase_config.py`

- [ ] **Step 1: Write the failing tests** — append cases for `[static_map] enabled = false, margin = 0.8`, missing section falls back to defaults `(enabled=True, margin=0.5)`, and a real-file regression like WP-2's `test_phase_config_parses_checked_in_env_config`.
- [ ] **Step 2:** `pytest tests/test_phase_config.py -v` → failures on `AttributeError: ... static_map`.
- [ ] **Step 3: Implement** — append:

```python
@dataclass(frozen=True)
class StaticMapParams:
    enabled: bool = True
    margin: float = 0.5
```

Extend `PhaseConfig`:

```python
    static_map: StaticMapParams
```

And in `from_configparser`:

```python
        sm_enabled = cp.getboolean("static_map", "enabled", fallback=True)
        sm_margin = cp.getfloat("static_map", "margin", fallback=0.5)
        ...
        return cls(
            ...,
            static_map=StaticMapParams(enabled=sm_enabled, margin=sm_margin),
        )
```

- [ ] **Step 4:** `pytest tests/test_phase_config.py -v` → PASS.
- [ ] **Step 5: Commit** — `feat(wp3): PhaseConfig resolves [static_map] enabled + margin`.

---

## Task 3: Append `[static_map]` to both env.configs

**Files:**
- Modify: `crowd_nav/configs/env.config`
- Modify: `crowd_nav/data/output_trained/env.config`

- [ ] **Step 1:** Append to both files (after any `[goal_allocator]` block):

```ini

[static_map]
enabled = true
margin = 0.5
```

- [ ] **Step 2:** `pytest tests/test_phase_config.py::test_phase_config_parses_checked_in_env_config -v` → PASS with the new field.
- [ ] **Step 3: Commit** — `feat(wp3): add [static_map] section to env.configs`.

---

## Task 4: Wire `test.py` → allocator receives `is_free`

**Files:**
- Modify: `crowd_nav/test.py`
- Create: `tests/test_test_py_wires_static_map.py`

- [ ] **Step 1: Write the failing test** — AST-based guard (same pattern as `tests/test_test_py_uses_allocator.py`):
  - `test.py` imports `StaticMap` from `crowd_sim.envs.utils.static_map`.
  - `test.py` source contains `allocate_waypoints(` followed by `is_free=` in the same call (token search).
- [ ] **Step 2:** `pytest tests/test_test_py_wires_static_map.py -v` → FAIL.
- [ ] **Step 3: Implement** — in `crowd_nav/test.py`, add import:

```python
from crowd_sim.envs.utils.static_map import StaticMap
```

Replace the `is_free = None` line inside the WP-2 allocator block with:

```python
    static_map: StaticMap | None = (
        StaticMap.from_static_obstacles(env.static_obstacles, margin=phase_cfg.static_map.margin)
        if phase_cfg.static_map.enabled and env.static_obstacles
        else None
    )
    is_free = static_map.is_free if static_map is not None else None
```

- [ ] **Step 4:** `pytest tests/test_test_py_wires_static_map.py -v` → PASS. Re-run `pytest tests/test_waypoint_integration.py -m "integration and slow"` — still green (MP4 > 10 KB). Quickly spot-check `exports/baseline.mp4`: the red `Goal-4` star must no longer overlap the top gray bar.
- [ ] **Step 5: Commit** — `feat(wp3): test.py passes StaticMap.is_free into allocate_waypoints`.

---

## Task 5: Route `generate_random_human_position('circle_crossing')` through `allocate_human_positions`

**Files:**
- Modify: `crowd_sim/envs/crowd_sim.py`
- Modify: `tests/test_crowd_sim_human_generation.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_crowd_sim_human_generation.py`:
  - Stub `env.static_map` with a rect centered on the robot corridor midpoint.
  - Call `env.generate_random_human_position(human_num=5, rule='circle_crossing')`.
  - Assert every spawned human's `(px, py)` and `(gx, gy)` are `is_free`.
- [ ] **Step 2:** run — FAIL because current code skips `is_free`.
- [ ] **Step 3: Implement** — replace the WP-2 loop body:

```python
        elif rule == 'circle_crossing':
            self.humans = []
            is_free = self.static_map.is_free if self.static_map is not None else None
            allocator = GoalAllocator(max_tries=500)
            pairs = allocator.allocate_human_positions(
                robot_start=tuple(self.curr_post),
                robot_goal=tuple(self.local_goal),
                occupied=[(self.robot.px, self.robot.py)],
                human_num=human_num,
                min_dist=self.humans_radius * 2 + self.discomfort_dist,
                is_free=is_free,
            )
            for (sx, sy), (gx, gy) in pairs:
                human = Human(self.config, 'humans')
                human.set(sx, sy, gx, gy, 0, 0, 0)
                self.humans.append(human)
```

Note: `self.humans_radius` is read from `[humans] radius` — add `self.humans_radius = self.config.getfloat('humans', 'radius')` in `configure()` if not already cached. Keep `generate_circle_crossing_human_new` method intact for other callers but it is now dead code for the `circle_crossing` rule.

- [ ] **Step 4:** `pytest tests/test_crowd_sim_human_generation.py tests/test_waypoint_integration.py -v` → all PASS.
- [ ] **Step 5: Commit** — `feat(wp3): route circle_crossing humans through GoalAllocator with is_free`.

---

## Task 6: Tier B — robot-vs-static collision in `CrowdSim.step()`

**Files:**
- Modify: `crowd_sim/envs/crowd_sim.py`
- Create: `tests/test_crowd_sim_static_collision.py`

- [ ] **Step 1: Write the failing test** — robot at `(0, 4.5)`, env configured with a single rect `(cx=0, cy=5, w=20, h=1)` + action moving `+y` → `env.step` returns `done=True` and `isinstance(info, Collision)`.
- [ ] **Step 2:** run — FAIL (today returns `done=False, info=Nothing()`).
- [ ] **Step 3: Implement** — in `CrowdSim.step()`, immediately after `end_position = np.array(...)` and before the `if self.global_time >= ...` ladder:

```python
        static_collision = False
        if self.static_map is not None:
            if not self.static_map.is_free(
                float(end_position[0]), float(end_position[1]),
                margin=self.robot.radius,
            ):
                static_collision = True
```

Extend the collision branch:

```python
        elif collision or static_collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
```

Build `self.static_map` once in `reset()` / `light_reset()` after `self.static_obstacles` is populated:

```python
        self.static_map = None
        if self.static_obs_cfg and self.static_obstacles:
            self.static_map = StaticMap.from_static_obstacles(
                self.static_obstacles, margin=0.0
            )
```

(Robot radius is passed per-call via `is_free(..., margin=self.robot.radius)` so the `StaticMap` itself stays agnostic.)

- [ ] **Step 4:** run the full suite — `pytest -q -m "not slow"` → all green. Existing WP-2 integration test may now observe `Collision` instead of `Reaching goal`; that's allowed (it only asserts `returncode == 0`). If it regresses, investigate: the SARL baseline may genuinely drive through a wall — that's a *signal*, not a bug.
- [ ] **Step 5: Commit** — `feat(wp3): robot-vs-static-obstacle collision in CrowdSim.step`.

---

## Task 7: Tier C — ORCA obstacles

**Files:**
- Modify: `crowd_sim/envs/policy/orca.py`
- Modify: `crowd_sim/envs/crowd_sim.py`
- Create: `tests/test_orca_obstacles.py`

- [ ] **Step 1: Write the failing test** — single-human scenario, ORCA policy, obstacle rect between start and goal, 50 predict+step rounds, assert trajectory never enters AABB.
- [ ] **Step 2:** run — FAIL (human walks straight through).
- [ ] **Step 3: Implement `orca.py`**:
  - Add `self.static_obstacle_polygons: list[list[tuple[float,float]]] = []` in `__init__`.
  - Add setter:

    ```python
    def set_static_obstacles(self, polygons: list[list[tuple[float, float]]]) -> None:
        """Pass CCW vertex lists; triggers sim rebuild on next predict()."""
        self.static_obstacle_polygons = list(polygons)
        self.sim = None  # force rebuild so addObstacle takes effect
    ```

  - In `predict()`, inside the `if self.sim is None:` branch, after the `addAgent` loop:

    ```python
            for poly in self.static_obstacle_polygons:
                self.sim.addObstacle(poly)
            if self.static_obstacle_polygons:
                self.sim.processObstacles()
    ```

  - Delete the misleading *"In this work, obstacles are not considered."* line from the docstring.

- [ ] **Step 4: Implement `crowd_sim.py` wiring** — in `reset()` and `light_reset()`, after humans are placed and `self.static_map` is built:

    ```python
    if self.static_map is not None:
        polys = [
            rect_vertices_ccw(o)
            for o in self.static_map.obstacles
            if o.kind == "rect"
        ]
        for human in self.humans:
            if hasattr(human.policy, "set_static_obstacles"):
                human.policy.set_static_obstacles(polys)
    ```

    (Circle obstacles could be approximated as N-gons later; out of scope here — the hardcoded pattern uses rects only.)

- [ ] **Step 5:** `pytest tests/test_orca_obstacles.py -v` + the full fast suite → all green.
- [ ] **Step 6: Commit** — `feat(wp3): ORCA humans navigate around static obstacles via sim.addObstacle`.

---

## Task 8: Extend `test_waypoint_integration.py`

**Files:**
- Modify: `tests/test_waypoint_integration.py`

- [ ] **Step 1: Extend** — after the existing `returncode == 0` / mp4-size assertions, parse the `Allocated N waypoints (start=(sx, sy) goal=(gx, gy))` line from stdout/stderr, reconstruct the straight-line waypoints via the same `_straight_line_source` logic, then build an offline `StaticMap` from the hardcoded pattern and assert every waypoint is `is_free(margin=0.5)`.
- [ ] **Step 2:** `pytest tests/test_waypoint_integration.py -m "integration and slow"` → PASS.
- [ ] **Step 3: Commit** — `test(wp3): integration assertion — no waypoint inside an obstacle`.

---

## Task 9: Dry run + milestone tag

- [ ] **Step 1:** Clean stale artefacts:

  ```bash
  rm -rf exports/*.mp4 .pytest_cache
  ```

- [ ] **Step 2: Fast suite must be green.**

  ```bash
  pytest -q -m "not slow"
  ```

- [ ] **Step 3: WP-1 smoke still green.**

  ```bash
  pytest tests/test_baseline_smoke.py -v -m "smoke"
  ```

  Expected: PASS. `returncode == 0`, MP4 > 10 KB, same-seed runs match. Content may differ vs. r1-baseline because humans now avoid walls — that's expected and desired.

- [ ] **Step 4: WP-2 + WP-3 integration green.**

  ```bash
  pytest tests/test_waypoint_integration.py -v -m "integration and slow"
  ```

- [ ] **Step 5: Visual spot-check.**

  ```bash
  open exports/baseline.mp4     # macOS
  ```

  Confirm: (a) `Goal-4` is visibly off the y=5 bar, (b) humans walk around — not through — the gray bars.

- [ ] **Step 6: Cross-reference R3.**
  - [ ] R3 — `StaticMap.is_free` (`test_static_map.py` 6+ cases)
  - [ ] R3 — allocator respects static obstacles (`test_allocator_respects_static_map.py`)
  - [ ] R3 — robot collision vs obstacles (`test_crowd_sim_static_collision.py`)
  - [ ] R3 — ORCA obstacle awareness (`test_orca_obstacles.py`)
  - [ ] NF3 — determinism preserved (`test_baseline_is_deterministic_across_runs` still PASS)
  - [ ] R7 — weights untouched (no edits under `crowd_nav/data/output_trained/`)

- [ ] **Step 7: Tag the milestone.**

  ```bash
  git tag -a wp3-static-map -m "WP-3 complete: StaticMap.is_free wired through allocator + robot collision + ORCA obstacles"
  ```

---

## Self-review summary

Spec coverage ↔ tasks ↔ verifying tests:

| Requirement | Task(s) | Verified by |
|---|---|---|
| R3 Tier A — StaticMap geometry | Task 1 | `tests/test_static_map.py` (6+ cases) |
| R3 Tier A — allocator honors `is_free` | Tasks 4, 5 | `tests/test_allocator_respects_static_map.py`, integration assertion in Task 8 |
| R3 Tier B — robot collision | Task 6 | `tests/test_crowd_sim_static_collision.py` |
| R3 Tier C — ORCA obstacles | Task 7 | `tests/test_orca_obstacles.py` |
| Config — `[static_map]` section | Tasks 2, 3 | `tests/test_phase_config.py` (real-file regression) |
| Backward compat — `enabled = false` falls through | Task 4 | unit test in `test_phase_config.py` |
| NF3 — determinism under seed | Task 9 | `tests/test_baseline_smoke.py::test_baseline_is_deterministic_across_runs` |
| R7 — weights untouched | n/a | no edits under `crowd_nav/data/output_trained/` |
| WP-2 seams reused, not re-implemented | Tasks 4, 5 | `GoalAllocator.sample_unused_position / allocate_waypoints / allocate_human_positions` remain unchanged — only callers pass a real `is_free` |

**Placeholder scan:** none — every step cites exact file paths and code snippets.

**Type consistency:**
- `Point = tuple[float, float]` continues from WP-2; `StaticMap.is_free` accepts `float, float` separately (callers unpack).
- `Obstacle.kind: Literal["rect", "circle"]` so the dispatch in `is_free` and `rect_vertices_ccw` is type-checkable.
- `StaticMapParams(enabled: bool, margin: float)` mirrors WP-2's `AllocatorParams` naming.
- `ORCA.set_static_obstacles(polygons: list[list[tuple[float, float]]])` — same shape RVO2 expects.

**Frozen-dataclass invariant:** `Obstacle`, `StaticMap`, `StaticMapParams` are all `@dataclass(frozen=True)` per the user's Python immutability rule.

**Logging over print:** all new debug output uses `logging`. No new `print()` calls.

---

## Execution Handoff

**Two execution options:**

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task (1→9), review between tasks. Tasks 1-3 share no files (geometry / phase_config / env.configs). Tasks 4-5 both touch wiring but no file overlap. Tasks 6-7 both edit `crowd_sim.py` — run serially. Tasks 8-9 are verification only.

2. **Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, checkpoints at Tasks 3, 5, and 9.

### Critical Files for Implementation

- /Users/zhunhao/Documents/Projects/crowdnav-dip/crowd_sim/envs/utils/static_map.py (new — geometry heart of WP-3)
- /Users/zhunhao/Documents/Projects/crowdnav-dip/crowd_sim/envs/utils/phase_config.py (modify — append `[static_map]` parser)
- /Users/zhunhao/Documents/Projects/crowdnav-dip/crowd_nav/configs/env.config (append `[static_map]`)
- /Users/zhunhao/Documents/Projects/crowdnav-dip/crowd_nav/data/output_trained/env.config (append `[static_map]` — this is the config `run_baseline.sh` actually uses)
- /Users/zhunhao/Documents/Projects/crowdnav-dip/crowd_nav/test.py (modify — build `StaticMap` and pass `is_free` into `allocate_waypoints`)
- /Users/zhunhao/Documents/Projects/crowdnav-dip/crowd_sim/envs/crowd_sim.py (modify — build `self.static_map`, route `circle_crossing` humans, add Tier-B collision, hand polygons to ORCA)
- /Users/zhunhao/Documents/Projects/crowdnav-dip/crowd_sim/envs/policy/orca.py (modify — `set_static_obstacles` setter + `addObstacle`/`processObstacles` in `predict`)
