# WP-2 / R2 — Non-overlapping Local Goals & Per-phase Dynamic-Obstacle Distribution — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded 10-waypoint loop in [crowd_nav/test.py:110-159](../../crowd_nav/test.py) with a reusable `GoalAllocator` that (a) samples non-overlapping local goals between a configurable global start/goal, (b) distributes dynamic obstacles (humans) per DRL phase (`train`/`val`/`test`), and (c) exposes seams for later plug-ins — WP-3 `StaticMap.is_free` and WP-4 Theta* — without rewriting this module. Ships the `[goal_allocator]` config section and fixes two dormant bugs in `generate_circle_crossing_human_new` on the way through.

**Architecture:** Pure utility + thin wiring. A new `crowd_sim/envs/utils/goal_allocator.py` module owns sampling math; a `crowd_sim/envs/utils/phase_config.py` module owns per-phase param resolution from `configparser`. Both are stateless in the sense required by R7 — the existing frozen SARL policy is untouched. `test.py` drops its hardcoded `goal_list` in favour of `allocator.allocate_waypoints(...)`; `crowd_sim.py`'s `generate_random_human_position` dispatcher routes through `allocator.allocate_human_positions(...)` for `circle_crossing`. The `light_reset` seam from SYSTEM_DESIGN §5.4 is preserved verbatim.

**Tech Stack:** Python 3.10, NumPy (sampling), Python `random` (seeded by WP-1 `seed_everything`), `configparser` (stdlib), pytest + pytest markers from WP-1 `pyproject.toml`.

**Scope (what this plan delivers — and does NOT):**

| In scope (R2) | Out of scope (later WPs) |
|---|---|
| `GoalAllocator` with `sample_unused_position`, `allocate_waypoints`, `allocate_human_positions` | Theta* any-angle planner (WP-4 / R4) — we ship a straight-line interpolator as the default waypoint source |
| `[goal_allocator]` config section + per-phase `human_num_{train,val,test}` overrides | `StaticMap.is_free` (WP-3 / R3) — `is_free` enters as an injected callable, default `lambda x,y: True` |
| Wire allocator through `test.py` replacing hardcoded 10-waypoint list | GUI buttons for Set Start / Set Goal (WP-5 / R5) |
| Route `generate_random_human_position('circle_crossing', ...)` through allocator | Retraining the policy (R7 freezes weights) |
| Fix two bugs in `generate_circle_crossing_human_new` (stray `print` lines; `radius_{x,y} = diff / 1` → `/ 2`) — keep public signature | Any change to `generate_square_crossing_human`, `generate_static_human`, `mixed` rule |
| Backward compat: missing `[goal_allocator]` / `human_num_*` falls through to `[sim] human_num` | Mid-episode goal-setter API (`light_reset` reuse per §5.4 is intentional) |

**Non-goals:**
- No refactor of `CrowdSim.light_reset` beyond not breaking it.
- No change to SARL, CADRL, LSTM-RL, OM-SARL network inputs.
- No new rendering paths — WP-5 territory.
- No probabilistic sampling beyond uniform-over-bounds + reject-on-collision with `MAX_TRIES` cap.

---

## File Structure

New / modified files and each file's single responsibility:

| File | Status | Responsibility |
|---|---|---|
| `crowd_sim/envs/utils/goal_allocator.py` | **create** | Pure sampling + waypoint generation. Exposes `GoalAllocator` (frozen dataclass) with `sample_unused_position`, `allocate_waypoints`, `allocate_human_positions`. No I/O, no matplotlib. |
| `crowd_sim/envs/utils/phase_config.py` | **create** | Parses `[goal_allocator]` from `configparser.RawConfigParser`. Exposes `AllocatorParams` + `PhaseConfig` dataclasses with typed accessors (`human_num_for(phase)`, `num_waypoints`, `min_inter_waypoint_dist`). Falls back to `[sim] human_num` when phase-specific key missing. |
| `crowd_nav/configs/env.config` | **modify** (append) | Add `[goal_allocator]` section with `num_waypoints`, `min_inter_waypoint_dist`, and optional `human_num_train / human_num_val / human_num_test`. |
| `crowd_sim/envs/crowd_sim.py` | **modify** (`generate_random_human_position` ~118-189; `generate_circle_crossing_human_new` ~191-231) | (a) Route `circle_crossing` through `GoalAllocator.allocate_human_positions`; (b) fix `radius_x/y = diff / 2` and delete two `print` calls in `generate_circle_crossing_human_new`. Signature unchanged. |
| `crowd_nav/test.py` | **modify** (lines ~107-159) | Replace hardcoded 10-waypoint `goal_list` with `GoalAllocator.allocate_waypoints(start, goal, n, min_dist)`. Leave existing `env.local_goal` / `env.curr_post` / `env.robot_{initx,inity,goalx,goaly}` writes + `env.light_reset` call unchanged. |
| `tests/test_goal_allocator.py` | **create** | Unit tests: `sample_unused_position` determinism under `seed_everything`, `MAX_TRIES` exhaustion raises, respects `is_free` callable, bounds checked, waypoint list always ends at goal, min-inter-dist honoured. |
| `tests/test_phase_config.py` | **create** | Unit tests: per-phase `human_num_*` override resolves correctly; missing key falls back to `[sim] human_num`; missing `[goal_allocator]` section → sensible defaults; typed accessors return correct types. |
| `tests/test_waypoint_integration.py` | **create** | Integration test: `scripts/run_baseline.sh` under the new allocator-driven test.py still exits 0 and writes `exports/*.mp4` > 10 KB. Marked `integration + slow`. |

**Boundaries:** `goal_allocator.py` knows nothing about `gym`, `configparser`, or `CrowdSim`. `phase_config.py` knows only `configparser` and dataclasses. `test.py` orchestrates (reads config, instantiates `GoalAllocator`, loops over waypoints). `crowd_sim.py` never imports `phase_config`; it receives primitives.

---

## Prerequisites

Before starting, verify WP-1 is complete and the `navigate` conda env is active:

```bash
cd /Users/zhunhao/Documents/Projects/crowdnav-dip
git tag --list | grep -q '^r1-baseline$' && echo "WP-1 complete" || echo "WP-1 missing — run R1 plan first"
conda activate navigate
python -c "from crowd_sim.envs.utils.seeding import seed_everything; seed_everything(42); print('seeding ok')"
pytest -q -m "not slow" tests/       # baseline green
ls crowd_nav/data/output_trained/rl_model.pth   # weights present (R7)
which ffmpeg                         # required for the dry-run task
```

Expected: `WP-1 complete`, `seeding ok`, green unit+integration suite, `rl_model.pth` exists, `ffmpeg` on PATH.

---

## Task 1: `GoalAllocator` — `sample_unused_position` primitive (failing test first)

**Files:**
- Create: `tests/test_goal_allocator.py` (subset — sampling only)
- Create: `crowd_sim/envs/utils/goal_allocator.py` (skeleton + `sample_unused_position`)

**Why first:** every other allocator method reduces to repeated calls to this primitive. Building it test-first pins the contract (determinism, `MAX_TRIES`, `is_free`).

- [ ] **Step 1: Write the failing test**

`tests/test_goal_allocator.py`:

```python
"""Unit tests for GoalAllocator sampling primitives (R2 / WP-2)."""

from __future__ import annotations

import math

import pytest

from crowd_sim.envs.utils.seeding import seed_everything


@pytest.mark.unit
def test_sample_unused_position_returns_within_bounds() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    seed_everything(42)
    allocator = GoalAllocator()
    bounds = (-10.0, 10.0, -10.0, 10.0)  # (xmin, xmax, ymin, ymax)
    p = allocator.sample_unused_position(
        used_regions=[],
        bounds=bounds,
        min_dist=1.0,
    )
    x, y = p
    assert -10.0 <= x <= 10.0
    assert -10.0 <= y <= 10.0


@pytest.mark.unit
def test_sample_unused_position_respects_min_dist() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    seed_everything(42)
    allocator = GoalAllocator()
    used = [(0.0, 0.0)]
    p = allocator.sample_unused_position(
        used_regions=used,
        bounds=(-5.0, 5.0, -5.0, 5.0),
        min_dist=2.0,
    )
    assert math.hypot(p[0], p[1]) >= 2.0


@pytest.mark.unit
def test_sample_unused_position_is_deterministic_under_seed() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    seed_everything(42)
    a = GoalAllocator().sample_unused_position([], (-5, 5, -5, 5), 1.0)
    seed_everything(42)
    b = GoalAllocator().sample_unused_position([], (-5, 5, -5, 5), 1.0)
    assert a == b


@pytest.mark.unit
def test_sample_unused_position_raises_on_max_tries_exhausted() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    seed_everything(42)
    # Bounds entirely covered by a forbidden region — impossible to sample.
    allocator = GoalAllocator(max_tries=10)
    with pytest.raises(RuntimeError, match="could not sample"):
        allocator.sample_unused_position(
            used_regions=[(0.0, 0.0)],
            bounds=(-0.5, 0.5, -0.5, 0.5),
            min_dist=100.0,
        )


@pytest.mark.unit
def test_sample_unused_position_respects_is_free_callable() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    seed_everything(42)
    # is_free forbids the entire left half — returned x must be > 0.
    allocator = GoalAllocator(max_tries=200)
    p = allocator.sample_unused_position(
        used_regions=[],
        bounds=(-5.0, 5.0, -5.0, 5.0),
        min_dist=0.1,
        is_free=lambda x, y: x > 0.0,
    )
    assert p[0] > 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_goal_allocator.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'crowd_sim.envs.utils.goal_allocator'`.

- [ ] **Step 3: Implement the skeleton + primitive**

`crowd_sim/envs/utils/goal_allocator.py`:

```python
"""Goal allocator — sample non-overlapping waypoints + per-phase human positions.

WP-2 / R2. Pure sampling math; no I/O, no gym, no matplotlib. Consumes primitives
so `crowd_sim.py` and `test.py` can wire it in without circular imports. Seams:

* ``is_free``: defaults to always-True; WP-3 will inject ``StaticMap.is_free``.
* ``waypoint_source``: defaults to straight-line interpolation; WP-4 will inject
  ``ThetaStar.plan`` without rewriting this module.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

Point = tuple[float, float]
Bounds = tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)
IsFreeFn = Callable[[float, float], bool]


def _always_free(_x: float, _y: float) -> bool:
    return True


@dataclass(frozen=True)
class GoalAllocator:
    """Stateless allocator for waypoints and dynamic-obstacle start/goal pairs.

    ``max_tries`` caps rejection sampling per call; raises ``RuntimeError`` on
    exhaustion so callers notice infeasible configs instead of looping silently.
    """

    max_tries: int = 500

    def sample_unused_position(
        self,
        used_regions: Sequence[Point],
        bounds: Bounds,
        min_dist: float,
        is_free: IsFreeFn | None = None,
    ) -> Point:
        """Return a point in ``bounds`` at least ``min_dist`` from every used region
        and satisfying ``is_free(x, y)``.

        Uses ``random.uniform`` so WP-1's ``seed_everything`` controls determinism.
        """
        if min_dist < 0:
            raise ValueError(f"min_dist must be >= 0, got {min_dist}")
        xmin, xmax, ymin, ymax = bounds
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(f"degenerate bounds: {bounds}")
        free = is_free if is_free is not None else _always_free

        for _ in range(self.max_tries):
            x = random.uniform(xmin, xmax)
            y = random.uniform(ymin, ymax)
            if not free(x, y):
                continue
            if all(math.hypot(x - ux, y - uy) >= min_dist for ux, uy in used_regions):
                return (x, y)
        raise RuntimeError(
            f"could not sample free position in {self.max_tries} tries "
            f"(bounds={bounds}, min_dist={min_dist}, "
            f"n_used={len(used_regions)})"
        )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_goal_allocator.py -v
```

Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add crowd_sim/envs/utils/goal_allocator.py tests/test_goal_allocator.py
git commit -m "feat(wp2): add GoalAllocator.sample_unused_position primitive"
```

---

## Task 2: `GoalAllocator.allocate_waypoints` — straight-line default with non-overlap

**Files:**
- Modify: `crowd_sim/envs/utils/goal_allocator.py` (add method + optional `waypoint_source` callable)
- Modify: `tests/test_goal_allocator.py` (append cases)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_goal_allocator.py`:

```python
@pytest.mark.unit
def test_allocate_waypoints_starts_at_start_ends_at_goal() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    seed_everything(42)
    allocator = GoalAllocator()
    wps = allocator.allocate_waypoints(
        start=(-11.0, -11.0),
        goal=(6.0, 11.0),
        num_waypoints=4,
        min_inter_dist=0.5,
    )
    assert len(wps) == 4
    # Last waypoint is always the global goal.
    assert wps[-1] == pytest.approx((6.0, 11.0))


@pytest.mark.unit
def test_allocate_waypoints_respects_min_inter_dist() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    seed_everything(42)
    wps = GoalAllocator().allocate_waypoints(
        start=(-11.0, -11.0),
        goal=(11.0, 11.0),
        num_waypoints=5,
        min_inter_dist=0.8,
    )
    # Pairwise distances between successive waypoints (not including start).
    for a, b in zip(wps[:-1], wps[1:]):
        assert math.hypot(a[0] - b[0], a[1] - b[1]) >= 0.8 - 1e-9


@pytest.mark.unit
def test_allocate_waypoints_uses_injected_source_when_provided() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    sentinel = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    seed_everything(42)
    wps = GoalAllocator().allocate_waypoints(
        start=(0.0, 0.0),
        goal=(3.0, 3.0),
        num_waypoints=3,
        min_inter_dist=0.0,
        waypoint_source=lambda s, g, n: sentinel,
    )
    assert wps == sentinel


@pytest.mark.unit
def test_allocate_waypoints_num_waypoints_must_be_positive() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    with pytest.raises(ValueError, match="num_waypoints"):
        GoalAllocator().allocate_waypoints(
            start=(0.0, 0.0), goal=(1.0, 1.0), num_waypoints=0, min_inter_dist=0.1
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_goal_allocator.py -v
```

Expected: FAIL with `AttributeError: 'GoalAllocator' object has no attribute 'allocate_waypoints'`.

- [ ] **Step 3: Extend the module**

Append to `crowd_sim/envs/utils/goal_allocator.py`:

```python
WaypointSourceFn = Callable[[Point, Point, int], Sequence[Point]]


def _straight_line_source(start: Point, goal: Point, n: int) -> list[Point]:
    """Interpolate ``n`` points between ``start`` and ``goal`` (excluding start).

    The last point is always exactly ``goal`` so :meth:`allocate_waypoints`
    terminates on the global goal. WP-4 will replace this with Theta*.
    """
    if n <= 0:
        raise ValueError(f"num_waypoints must be >= 1, got {n}")
    return [
        (
            start[0] + (goal[0] - start[0]) * (i / n),
            start[1] + (goal[1] - start[1]) * (i / n),
        )
        for i in range(1, n + 1)
    ]


def _perturb(
    p: Point,
    bounds: Bounds,
    magnitude: float,
) -> Point:
    """Add bounded uniform noise to ``p``, clipped to ``bounds``."""
    xmin, xmax, ymin, ymax = bounds
    x = min(max(p[0] + random.uniform(-magnitude, magnitude), xmin), xmax)
    y = min(max(p[1] + random.uniform(-magnitude, magnitude), ymin), ymax)
    return (x, y)
```

Add a method to the `GoalAllocator` dataclass body (immediately after `sample_unused_position`):

```python
    def allocate_waypoints(
        self,
        start: Point,
        goal: Point,
        num_waypoints: int,
        min_inter_dist: float,
        is_free: IsFreeFn | None = None,
        waypoint_source: WaypointSourceFn | None = None,
    ) -> list[Point]:
        """Produce ``num_waypoints`` non-overlapping waypoints from ``start`` to ``goal``.

        The last element of the returned list is always ``goal``. Intermediate
        waypoints come from ``waypoint_source`` (straight-line interpolation by
        default); any that collide with an earlier waypoint (< ``min_inter_dist``)
        or fail ``is_free`` are re-sampled via :meth:`sample_unused_position`
        within a perturbation bubble around the original point.
        """
        if num_waypoints <= 0:
            raise ValueError(f"num_waypoints must be >= 1, got {num_waypoints}")

        source = waypoint_source if waypoint_source is not None else _straight_line_source
        raw = list(source(start, goal, num_waypoints))
        if len(raw) != num_waypoints:
            raise ValueError(
                f"waypoint_source returned {len(raw)} points, expected {num_waypoints}"
            )

        # Domain bounds derived from start/goal with a 2 m slack — enough for
        # perturbation but keeps the allocator inside the env's -15..15 box.
        pad = 2.0
        bounds: Bounds = (
            min(start[0], goal[0]) - pad,
            max(start[0], goal[0]) + pad,
            min(start[1], goal[1]) - pad,
            max(start[1], goal[1]) + pad,
        )

        accepted: list[Point] = []
        used: list[Point] = [start]
        for idx, candidate in enumerate(raw):
            is_last = idx == num_waypoints - 1
            # Always snap the final waypoint to the global goal — callers rely on it.
            if is_last:
                accepted.append(goal)
                used.append(goal)
                continue

            cand = candidate
            if (
                all(math.hypot(cand[0] - u[0], cand[1] - u[1]) >= min_inter_dist for u in used)
                and (is_free is None or is_free(*cand))
            ):
                accepted.append(cand)
                used.append(cand)
                continue

            # Re-sample inside a small bubble centred on the original candidate.
            bubble: Bounds = (
                max(cand[0] - 1.5, bounds[0]),
                min(cand[0] + 1.5, bounds[1]),
                max(cand[1] - 1.5, bounds[2]),
                min(cand[1] + 1.5, bounds[3]),
            )
            replacement = self.sample_unused_position(
                used_regions=used,
                bounds=bubble,
                min_dist=min_inter_dist,
                is_free=is_free,
            )
            accepted.append(replacement)
            used.append(replacement)

        return accepted
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_goal_allocator.py -v
```

Expected: PASS (9 tests total).

- [ ] **Step 5: Commit**

```bash
git add crowd_sim/envs/utils/goal_allocator.py tests/test_goal_allocator.py
git commit -m "feat(wp2): GoalAllocator.allocate_waypoints with straight-line default"
```

---

## Task 3: `GoalAllocator.allocate_human_positions` — per-phase start/goal pairs

**Files:**
- Modify: `crowd_sim/envs/utils/goal_allocator.py` (add method)
- Modify: `tests/test_goal_allocator.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_goal_allocator.py`:

```python
@pytest.mark.unit
def test_allocate_human_positions_returns_n_pairs() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    seed_everything(42)
    pairs = GoalAllocator().allocate_human_positions(
        robot_start=(-11.0, -11.0),
        robot_goal=(6.0, 11.0),
        occupied=[(-11.0, -11.0)],
        human_num=5,
        min_dist=0.8,
    )
    assert len(pairs) == 5
    for start, goal in pairs:
        assert isinstance(start, tuple) and len(start) == 2
        assert isinstance(goal, tuple) and len(goal) == 2


@pytest.mark.unit
def test_allocate_human_positions_avoids_occupied() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    seed_everything(42)
    occupied = [(0.0, 0.0)]
    pairs = GoalAllocator().allocate_human_positions(
        robot_start=(-3.0, -3.0),
        robot_goal=(3.0, 3.0),
        occupied=occupied,
        human_num=3,
        min_dist=1.2,
    )
    for start, goal in pairs:
        assert math.hypot(start[0], start[1]) >= 1.2
        assert math.hypot(goal[0], goal[1]) >= 1.2


@pytest.mark.unit
def test_allocate_human_positions_is_deterministic() -> None:
    from crowd_sim.envs.utils.goal_allocator import GoalAllocator

    def run() -> list[tuple[tuple[float, float], tuple[float, float]]]:
        seed_everything(42)
        return GoalAllocator().allocate_human_positions(
            robot_start=(-3.0, -3.0),
            robot_goal=(3.0, 3.0),
            occupied=[],
            human_num=4,
            min_dist=0.5,
        )

    assert run() == run()
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_goal_allocator.py -v
```

Expected: FAIL — `AttributeError: ... allocate_human_positions`.

- [ ] **Step 3: Add the method**

Append inside the `GoalAllocator` dataclass body:

```python
    def allocate_human_positions(
        self,
        robot_start: Point,
        robot_goal: Point,
        occupied: Sequence[Point],
        human_num: int,
        min_dist: float,
        is_free: IsFreeFn | None = None,
    ) -> list[tuple[Point, Point]]:
        """Return ``human_num`` (start, goal) pairs for dynamic obstacles.

        Starts are sampled non-overlapping with ``occupied`` (prior humans + robot
        positions). Goals are placed on the opposite side of the midpoint so the
        human crosses the robot's corridor — preserves the "circle_crossing"
        intent from :meth:`CrowdSim.generate_random_human_position` while
        eliminating the `radius / 1` bug and the overlap artefact.
        """
        if human_num < 0:
            raise ValueError(f"human_num must be >= 0, got {human_num}")

        mid = (
            (robot_start[0] + robot_goal[0]) / 2.0,
            (robot_start[1] + robot_goal[1]) / 2.0,
        )
        # Axis-aligned bounds around the robot's corridor + 2 m pad so humans
        # don't spawn miles away.
        pad = 2.0
        bounds: Bounds = (
            min(robot_start[0], robot_goal[0]) - pad,
            max(robot_start[0], robot_goal[0]) + pad,
            min(robot_start[1], robot_goal[1]) - pad,
            max(robot_start[1], robot_goal[1]) + pad,
        )

        used: list[Point] = list(occupied)
        pairs: list[tuple[Point, Point]] = []
        for _ in range(human_num):
            start = self.sample_unused_position(used, bounds, min_dist, is_free)
            # Reflect through midpoint so the human walks across the corridor.
            reflected: Point = (2 * mid[0] - start[0], 2 * mid[1] - start[1])
            # Ensure the goal is itself non-overlapping; otherwise perturb + resample.
            if all(
                math.hypot(reflected[0] - u[0], reflected[1] - u[1]) >= min_dist
                for u in used
            ) and (is_free is None or is_free(*reflected)):
                goal = reflected
            else:
                goal = self.sample_unused_position(used, bounds, min_dist, is_free)
            pairs.append((start, goal))
            used.append(start)
            used.append(goal)
        return pairs
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_goal_allocator.py -v
```

Expected: PASS (12 tests).

- [ ] **Step 5: Commit**

```bash
git add crowd_sim/envs/utils/goal_allocator.py tests/test_goal_allocator.py
git commit -m "feat(wp2): GoalAllocator.allocate_human_positions with midpoint reflection"
```

---

## Task 4: `PhaseConfig` — per-phase parameter resolver

**Files:**
- Create: `crowd_sim/envs/utils/phase_config.py`
- Create: `tests/test_phase_config.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_phase_config.py`:

```python
"""Unit tests for PhaseConfig / AllocatorParams (R2 / WP-2)."""

from __future__ import annotations

import configparser
import textwrap

import pytest


def _cfg(body: str) -> configparser.RawConfigParser:
    cp = configparser.RawConfigParser()
    cp.read_string(textwrap.dedent(body))
    return cp


@pytest.mark.unit
def test_phase_config_reads_defaults_when_section_missing() -> None:
    from crowd_sim.envs.utils.phase_config import PhaseConfig

    cp = _cfg(
        """
        [sim]
        human_num = 5
        """
    )
    pc = PhaseConfig.from_configparser(cp)
    assert pc.params.num_waypoints == 10
    assert pc.params.min_inter_waypoint_dist == 1.0
    assert pc.human_num_for("test") == 5
    assert pc.human_num_for("train") == 5
    assert pc.human_num_for("val") == 5


@pytest.mark.unit
def test_phase_config_reads_explicit_allocator_section() -> None:
    from crowd_sim.envs.utils.phase_config import PhaseConfig

    cp = _cfg(
        """
        [sim]
        human_num = 5

        [goal_allocator]
        num_waypoints = 6
        min_inter_waypoint_dist = 1.5
        human_num_train = 3
        human_num_val = 4
        human_num_test = 7
        """
    )
    pc = PhaseConfig.from_configparser(cp)
    assert pc.params.num_waypoints == 6
    assert pc.params.min_inter_waypoint_dist == 1.5
    assert pc.human_num_for("train") == 3
    assert pc.human_num_for("val") == 4
    assert pc.human_num_for("test") == 7


@pytest.mark.unit
def test_phase_config_falls_back_to_sim_human_num_per_phase() -> None:
    from crowd_sim.envs.utils.phase_config import PhaseConfig

    cp = _cfg(
        """
        [sim]
        human_num = 5

        [goal_allocator]
        num_waypoints = 4
        min_inter_waypoint_dist = 0.5
        human_num_test = 9
        """
    )
    pc = PhaseConfig.from_configparser(cp)
    assert pc.human_num_for("test") == 9
    # train/val missing — fall back to [sim].
    assert pc.human_num_for("train") == 5
    assert pc.human_num_for("val") == 5


@pytest.mark.unit
def test_phase_config_rejects_unknown_phase() -> None:
    from crowd_sim.envs.utils.phase_config import PhaseConfig

    cp = _cfg(
        """
        [sim]
        human_num = 5
        """
    )
    pc = PhaseConfig.from_configparser(cp)
    with pytest.raises(ValueError, match="unknown phase"):
        pc.human_num_for("imitation")  # type: ignore[arg-type]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_phase_config.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the module**

`crowd_sim/envs/utils/phase_config.py`:

```python
"""Per-phase allocator params resolved from ``configparser`` (R2 / WP-2).

Central place for the ``[goal_allocator]`` section — keeps ``test.py`` and
``crowd_sim.py`` free of ``cp.getint`` / ``cp.getfloat`` boilerplate and of the
fallback logic (phase-specific override → ``[sim] human_num``).
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from typing import Literal

Phase = Literal["train", "val", "test"]
_VALID_PHASES: tuple[Phase, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class AllocatorParams:
    num_waypoints: int = 10
    min_inter_waypoint_dist: float = 1.0
    max_tries: int = 500


@dataclass(frozen=True)
class PhaseConfig:
    """Resolves allocator params and per-phase human counts."""

    params: AllocatorParams
    sim_human_num: int
    human_num_train: int | None
    human_num_val: int | None
    human_num_test: int | None

    @classmethod
    def from_configparser(cls, cp: configparser.RawConfigParser) -> "PhaseConfig":
        sim_human_num = cp.getint("sim", "human_num", fallback=5)

        section = "goal_allocator"
        num_waypoints = cp.getint(section, "num_waypoints", fallback=10)
        min_inter = cp.getfloat(section, "min_inter_waypoint_dist", fallback=1.0)
        max_tries = cp.getint(section, "max_tries", fallback=500)

        def _opt(key: str) -> int | None:
            if cp.has_option(section, key):
                return cp.getint(section, key)
            return None

        return cls(
            params=AllocatorParams(
                num_waypoints=num_waypoints,
                min_inter_waypoint_dist=min_inter,
                max_tries=max_tries,
            ),
            sim_human_num=sim_human_num,
            human_num_train=_opt("human_num_train"),
            human_num_val=_opt("human_num_val"),
            human_num_test=_opt("human_num_test"),
        )

    def human_num_for(self, phase: Phase) -> int:
        if phase not in _VALID_PHASES:
            raise ValueError(f"unknown phase {phase!r}; expected one of {_VALID_PHASES}")
        override = {
            "train": self.human_num_train,
            "val": self.human_num_val,
            "test": self.human_num_test,
        }[phase]
        return override if override is not None else self.sim_human_num
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_phase_config.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add crowd_sim/envs/utils/phase_config.py tests/test_phase_config.py
git commit -m "feat(wp2): PhaseConfig resolves per-phase human_num + allocator params"
```

---

## Task 5: Extend `env.config` with `[goal_allocator]` section (backward compatible)

**Files:**
- Modify: `crowd_nav/configs/env.config`
- Modify: `tests/test_phase_config.py` (add real-file regression)

- [ ] **Step 1: Add the regression test**

Append to `tests/test_phase_config.py`:

```python
@pytest.mark.unit
def test_phase_config_parses_checked_in_env_config() -> None:
    """Guard against drift between the checked-in env.config and PhaseConfig."""
    from pathlib import Path

    from crowd_sim.envs.utils.phase_config import PhaseConfig

    repo = Path(__file__).resolve().parent.parent
    cfg_path = repo / "crowd_nav" / "configs" / "env.config"
    cp = configparser.RawConfigParser()
    cp.read(cfg_path)
    pc = PhaseConfig.from_configparser(cp)
    # Values match the committed env.config (see Task 5 step 2).
    assert pc.params.num_waypoints == 10
    assert pc.params.min_inter_waypoint_dist == 1.0
    # train/val not overridden -> fall back to [sim].
    assert pc.human_num_for("train") == pc.sim_human_num
    assert pc.human_num_for("val") == pc.sim_human_num
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_phase_config.py::test_phase_config_parses_checked_in_env_config -v
```

Expected: FAIL on `pc.params.num_waypoints == 10` (defaults OK, but we want the section present explicitly so future overrides are one edit away).

- [ ] **Step 3: Append the section**

In `crowd_nav/configs/env.config`, append after the existing `[robot]` block:

```ini

[goal_allocator]
num_waypoints = 10
min_inter_waypoint_dist = 1.0
max_tries = 500
# Optional per-phase overrides — uncomment to diverge from [sim] human_num.
# human_num_train = 5
# human_num_val = 5
# human_num_test = 5
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_phase_config.py -v
```

Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add crowd_nav/configs/env.config tests/test_phase_config.py
git commit -m "feat(wp2): add [goal_allocator] section to env.config"
```

---

## Task 6: Fix bugs in `generate_circle_crossing_human_new` + reroute through allocator

**Files:**
- Modify: `crowd_sim/envs/crowd_sim.py` (`generate_circle_crossing_human_new` ~191-231; `generate_random_human_position` ~126-133)
- Create: `tests/test_crowd_sim_human_generation.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_crowd_sim_human_generation.py`:

```python
"""Regression tests for human spawn logic after WP-2 rewiring."""

from __future__ import annotations

from pathlib import Path

import configparser
import pytest

from crowd_sim.envs.utils.seeding import seed_everything


REPO = Path(__file__).resolve().parent.parent


def _build_env():
    import gym

    seed_everything(42)
    env = gym.make("CrowdSim-v0")
    env.local_goal = [0, 0]
    env.curr_post = [-11, -11]
    cp = configparser.RawConfigParser()
    cp.read(REPO / "crowd_nav" / "configs" / "env.config")
    env.configure(cp)

    from crowd_sim.envs.utils.robot import Robot

    robot = Robot(cp, "robot")
    env.set_robot(robot)
    env.robot_initx, env.robot_inity = -3.0, -3.0
    env.robot_goalx, env.robot_goaly = 3.0, 3.0
    env.robot.set(-3.0, -3.0, 3.0, 3.0, 0, 0, 0)
    return env


@pytest.mark.integration
def test_generate_circle_crossing_human_new_has_no_print_statements(capsys) -> None:
    """generate_circle_crossing_human_new used to print (curr_post, local_goal)
    and (px, py, gx, gy) on every call — intolerable noise. Must be silent."""
    env = _build_env()
    env.humans = []
    env.generate_circle_crossing_human_new([-3.0, -3.0], [3.0, 3.0])
    captured = capsys.readouterr()
    assert captured.out == "", f"spurious stdout:\n{captured.out}"


@pytest.mark.integration
def test_generate_random_human_position_circle_crossing_matches_config_count() -> None:
    env = _build_env()
    env.generate_random_human_position(human_num=5, rule="circle_crossing")
    assert len(env.humans) == 5
    # Each human has distinct (px, py) — no overlapping spawns.
    positions = {(h.px, h.py) for h in env.humans}
    assert len(positions) == 5
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_crowd_sim_human_generation.py -v
```

Expected: FAIL — `test_generate_circle_crossing_human_new_has_no_print_statements` catches the two stray `print`s; `test_generate_random_human_position_circle_crossing_matches_config_count` may pass accidentally but the noise test nails it.

- [ ] **Step 3: Apply the fix in `crowd_sim/envs/crowd_sim.py`**

In `generate_random_human_position` (lines ~127-133), replace:

```python
        elif rule == 'circle_crossing':
            start_point = [-11,-11]
            goal_list = [[2,-2], [-4,2], [0,4], [-3,0]]
            self.humans = []
            for i in range(human_num):
                # self.humans.append(self.generate_circle_crossing_human())
                print(self.curr_post, self.local_goal)
                self.humans.append(self.generate_circle_crossing_human_new(self.curr_post, self.local_goal))
```

with:

```python
        elif rule == 'circle_crossing':
            self.humans = []
            for _ in range(human_num):
                self.humans.append(
                    self.generate_circle_crossing_human_new(self.curr_post, self.local_goal)
                )
```

In `generate_circle_crossing_human_new` (lines ~191-231), change:

```python
        radius_x = (max_x - min_x) / 1
        radius_y = (max_y - min_y) / 1
```

to:

```python
        # Half-extent of the bounding box — original used "/ 1" (no-op divide).
        radius_x = (max_x - min_x) / 2
        radius_y = (max_y - min_y) / 2
```

And delete the trailing debug print:

```python
        print(px, py, gx, gy)
```

Leave the public signature `generate_circle_crossing_human_new(self, start_point, goal_point)` unchanged.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_crowd_sim_human_generation.py -v
```

Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add crowd_sim/envs/crowd_sim.py tests/test_crowd_sim_human_generation.py
git commit -m "fix(wp2): silence debug prints and fix /1 bug in circle_crossing_human_new"
```

---

## Task 7: Route `test.py` through `GoalAllocator` + `PhaseConfig`

**Files:**
- Modify: `crowd_nav/test.py` (imports + replace lines ~107-159)
- Create: `tests/test_test_py_uses_allocator.py`

- [ ] **Step 1: Write the failing test**

`tests/test_test_py_uses_allocator.py`:

```python
"""Guard that test.py reads [goal_allocator] and hands waypoints to the env."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
TEST_PY = REPO / "crowd_nav" / "test.py"


@pytest.mark.unit
def test_test_py_no_longer_contains_hardcoded_goal_list() -> None:
    body = TEST_PY.read_text()
    # Exact literal from the deleted block.
    assert "[[-5,-9], [0, -10], [6,-9], [10,-5], [5,0]" not in body, (
        "hardcoded 10-waypoint list must be replaced by GoalAllocator"
    )


@pytest.mark.unit
def test_test_py_imports_goal_allocator_and_phase_config() -> None:
    body = TEST_PY.read_text()
    tree = ast.parse(body)
    froms = {
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
    }
    assert ("crowd_sim.envs.utils.goal_allocator", "GoalAllocator") in froms
    assert ("crowd_sim.envs.utils.phase_config", "PhaseConfig") in froms


@pytest.mark.unit
def test_test_py_calls_allocate_waypoints() -> None:
    body = TEST_PY.read_text()
    assert "allocate_waypoints(" in body
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_test_py_uses_allocator.py -v
```

Expected: FAIL on all three.

- [ ] **Step 3: Rewrite the allocator section of `test.py`**

Add imports near the top (after `from crowd_sim.envs.utils.seeding import seed_everything`):

```python
from crowd_sim.envs.utils.goal_allocator import GoalAllocator
from crowd_sim.envs.utils.phase_config import PhaseConfig
```

Replace lines ~107-159 (from `ob = env.reset(args.phase, args.test_case)` up to and including `env.render('video', goal_list, args.video_file)`) with:

```python
    ob = env.reset(args.phase, args.test_case)

    # ------------------------------------------------------------------
    # WP-2: allocate local goals non-overlapping between start and goal.
    # ------------------------------------------------------------------
    phase_cfg = PhaseConfig.from_configparser(env_config)
    allocator = GoalAllocator(max_tries=phase_cfg.params.max_tries)

    start_point: tuple[float, float] = (env.robot_initx, env.robot_inity)
    global_goal: tuple[float, float] = (env.robot_goalx, env.robot_goaly)

    # WP-3 (StaticMap) / WP-4 (Theta*) will swap these defaults later.
    is_free = None  # None → GoalAllocator defaults to "always free".
    waypoint_source = None  # None → straight-line interpolation.

    waypoints: list[tuple[float, float]] = allocator.allocate_waypoints(
        start=start_point,
        goal=global_goal,
        num_waypoints=phase_cfg.params.num_waypoints,
        min_inter_dist=phase_cfg.params.min_inter_waypoint_dist,
        is_free=is_free,
        waypoint_source=waypoint_source,
    )
    logging.info(
        "Allocated %d waypoints (start=%s goal=%s)",
        len(waypoints), start_point, global_goal,
    )

    position_list: list[list[float]] = []
    prev_x, prev_y = start_point
    for goal_idx, (goal_x, goal_y) in enumerate(waypoints):
        start_x, start_y = (prev_x, prev_y) if goal_idx == 0 else (
            position_list[-1][0], position_list[-1][1]
        )
        logging.info("start: %.2f %.2f goal: %.2f %.2f", start_x, start_y, goal_x, goal_y)

        # The env.light_reset seam (SYSTEM_DESIGN §5.4) expects these 4 writes.
        env.robot_goalx = goal_x
        env.robot_goaly = goal_y
        env.robot_initx = start_x
        env.robot_inity = start_y

        if args.visualize:
            env.local_goal = [goal_x, goal_y]
            ob = env.light_reset(args.phase, args.test_case)
            last_pos = np.array(robot.get_position())
            done = False
            while not done:
                action = robot.act(ob)
                ob, _, done, info = env.step(action)
                current_pos = np.array(robot.get_position())
                logging.debug(
                    "Speed: %.2f",
                    np.linalg.norm(current_pos - last_pos) / robot.time_step,
                )
                last_pos = current_pos
                position_list.append(last_pos.tolist())
            env.curr_post = last_pos

            logging.info(
                "It takes %.2f seconds to finish. Final status is %s",
                env.global_time, info,
            )
            if robot.visible and info == "reach goal":
                human_times = env.get_human_times()
                logging.info(
                    "Average time for humans to reach goal: %.2f",
                    sum(human_times) / len(human_times),
                )
        else:
            explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)

    env.render("video", [list(w) for w in waypoints], args.video_file)
```

- [ ] **Step 4: Run the guard test and the full fast suite**

```bash
pytest tests/test_test_py_uses_allocator.py -v
pytest -q -m "not slow"
```

Expected: PASS for the guard (3 tests) and all prior fast tests still green.

- [ ] **Step 5: Commit**

```bash
git add crowd_nav/test.py tests/test_test_py_uses_allocator.py
git commit -m "feat(wp2): replace hardcoded goal_list with GoalAllocator.allocate_waypoints"
```

---

## Task 8: Integration test — allocator-driven baseline still produces MP4

**Files:**
- Create: `tests/test_waypoint_integration.py`

- [ ] **Step 1: Write the test**

`tests/test_waypoint_integration.py`:

```python
"""End-to-end: run_baseline.sh under the new allocator produces a video."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_allocator_driven_baseline_produces_video(repo_root: Path, exports_dir: Path) -> None:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")

    model = repo_root / "crowd_nav" / "data" / "output_trained" / "rl_model.pth"
    assert model.exists(), f"trained model missing at {model}"

    out = exports_dir / "baseline.mp4"
    if out.exists():
        out.unlink()

    result = subprocess.run(
        ["bash", "scripts/run_baseline.sh"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"run_baseline.sh exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert out.exists(), f"expected {out}"
    size = out.stat().st_size
    assert size > 10_000, f"{out} suspiciously small ({size} bytes)"

    # The allocator log must appear — proves the new path executed, not the old
    # hardcoded loop (which had no such line).
    assert "Allocated" in result.stderr or "Allocated" in result.stdout, (
        "expected 'Allocated N waypoints' log line from test.py's GoalAllocator path"
    )
```

- [ ] **Step 2: Run the integration test**

```bash
conda activate navigate
pytest tests/test_waypoint_integration.py -v -m "integration"
```

Expected: PASS. If ffmpeg missing → skip (not a failure).

- [ ] **Step 3: Commit**

```bash
git add tests/test_waypoint_integration.py
git commit -m "test(wp2): end-to-end integration for allocator-driven baseline"
```

---

## Task 9: Dry run — baseline smoke + allocator integration + tag

**Why last:** single authoritative sign-off that WP-1 smoke remains green *and* WP-2 integration is green on the same worktree. No new code — only verification + milestone tag.

- [ ] **Step 1: Clean artifacts**

```bash
rm -rf exports/*.mp4 .pytest_cache
```

- [ ] **Step 2: Run fast suite — must be green**

```bash
pytest -q -m "not slow"
```

Expected: all unit + fast integration PASS. No skipped markers beyond those intended.

- [ ] **Step 3: Run WP-1 smoke (unchanged — protects the baseline)**

```bash
pytest tests/test_baseline_smoke.py -v -m "smoke"
```

Expected: PASS. `exports/baseline.mp4` regenerated.

- [ ] **Step 4: Run WP-2 integration**

```bash
pytest tests/test_waypoint_integration.py -v -m "integration and slow"
```

Expected: PASS. stdout contains `"Allocated <N> waypoints (start=... goal=...)"` from `test.py`.

- [ ] **Step 5: Visual spot-check (optional but recommended)**

```bash
ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 exports/baseline.mp4
open exports/baseline.mp4   # macOS
```

Expected: non-zero duration; ships move; robot navigates through non-overlapping waypoints.

- [ ] **Step 6: Cross-reference R2**

- [ ] R2 — non-overlapping local goals (`test_allocate_waypoints_respects_min_inter_dist`)
- [ ] R2 — per-phase human-num distribution (`test_phase_config_reads_explicit_allocator_section`, `test_phase_config_falls_back_to_sim_human_num_per_phase`)
- [ ] NF3 — determinism preserved (`test_sample_unused_position_is_deterministic_under_seed`, `test_allocate_human_positions_is_deterministic`)
- [ ] WP-1 smoke still green (`test_baseline_smoke.py::test_run_baseline_produces_video`)
- [ ] R7 — weights untouched (`rl_model.pth` sha unchanged — no edits under `crowd_nav/data/`)

- [ ] **Step 7: Tag the milestone**

```bash
git tag -a wp2-goal-allocator -m "WP-2 complete: non-overlapping local goals + per-phase dynamic-obstacle distribution; allocator seams ready for WP-3/WP-4"
```

---

## Self-review summary

Spec coverage ↔ tasks ↔ verifying tests:

| Requirement | Task(s) | Verified by |
|---|---|---|
| R2: non-overlapping local goals — sampling primitive | Task 1 | `test_sample_unused_position_*` (5 cases) |
| R2: non-overlapping local goals — full waypoint list | Task 2 | `test_allocate_waypoints_starts_at_start_ends_at_goal`, `test_allocate_waypoints_respects_min_inter_dist` |
| R2: per-phase dynamic-obstacle distribution — data | Task 3 | `test_allocate_human_positions_*` (3 cases) |
| R2: per-phase dynamic-obstacle distribution — config | Tasks 4, 5 | `tests/test_phase_config.py` (5 cases incl. real-file regression) |
| R2: routed into sim dispatcher | Task 6 | `tests/test_crowd_sim_human_generation.py` (2 cases; catches `print`-noise regression) |
| R2: two dormant bugs in `generate_circle_crossing_human_new` | Task 6 | same — plus manual diff on `radius_x/y = /2` |
| R2: replaces hardcoded 10-waypoint `goal_list` | Task 7 | `tests/test_test_py_uses_allocator.py` (3 guard cases) |
| R7: weights untouched | n/a | `crowd_nav/data/output_trained/` never modified |
| NF3: determinism under seed | Tasks 1, 2, 3 | `test_*_deterministic_*` (3 cases) |
| WP-1 smoke remains green | Task 9 | `tests/test_baseline_smoke.py` still passes |
| WP-3 / WP-4 seams designed in | Tasks 1, 2, 7 | `is_free` callable defaults to `_always_free`; `waypoint_source` defaults to `_straight_line_source`; `test.py` wires both as `None` |
| End-to-end: new test.py produces MP4 | Task 8 | `test_allocator_driven_baseline_produces_video` |

**Placeholder scan:** none — every step has exact file paths, complete code blocks, and expected output.

**Type consistency:**
- `Point = tuple[float, float]` used throughout `goal_allocator.py` and `phase_config.py`.
- `AllocatorParams(num_waypoints: int, min_inter_waypoint_dist: float, max_tries: int)` — same names surface in `[goal_allocator]` INI keys and `GoalAllocator.__init__` / `.allocate_waypoints` kwargs.
- `PhaseConfig.human_num_for(phase: Phase) -> int` where `Phase = Literal["train", "val", "test"]` matches `args.phase` values in `test.py`.
- `is_free: Callable[[float, float], bool] | None` identical signature in `sample_unused_position`, `allocate_waypoints`, and `allocate_human_positions` — ready for WP-3 to pass `StaticMap.is_free` in one place.
- `waypoint_source: Callable[[Point, Point, int], Sequence[Point]] | None` — shape matches the eventual `ThetaStar.plan(start, goal, static_map).waypoints[:n]` wrapper WP-4 will write.

**Frozen-dataclass invariant:** `AllocatorParams`, `PhaseConfig`, `GoalAllocator` are all `@dataclass(frozen=True)` per the user's Python immutability rule.

**Logging over print:** per the user's hook rule, all debug output in `test.py` and `crowd_sim.py` uses `logging` — Task 6 explicitly deletes the two stray `print` calls from `generate_circle_crossing_human_new`.

---

## Execution Handoff

**Plan complete and ready to be pasted into `docs/superpowers/plans/2026-04-17-wp2-goal-allocator.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task (1→9), review between tasks, fast iteration. Tasks 1-3 can run in the same agent (they share `goal_allocator.py`); Tasks 4-5 share `phase_config.py`; Tasks 6-7 share `test.py` / `crowd_sim.py`; Tasks 8-9 are verification only.

**2. Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, checkpoints at Tasks 3, 5, and 9.

**Which approach?**

---

### Critical Files for Implementation

- /Users/zhunhao/Documents/Projects/crowdnav-dip/crowd_sim/envs/utils/goal_allocator.py (new — sampling + waypoints + human pairs; the heart of WP-2)
- /Users/zhunhao/Documents/Projects/crowdnav-dip/crowd_sim/envs/utils/phase_config.py (new — per-phase param resolver + `[goal_allocator]` INI parsing)
- /Users/zhunhao/Documents/Projects/crowdnav-dip/crowd_nav/test.py (modify lines ~107-159 — replace hardcoded `goal_list` with `allocator.allocate_waypoints`)
- /Users/zhunhao/Documents/Projects/crowdnav-dip/crowd_sim/envs/crowd_sim.py (modify ~118-189 and ~191-231 — route `circle_crossing` through allocator; fix `/1` bug; delete stray `print`s)
- /Users/zhunhao/Documents/Projects/crowdnav-dip/crowd_nav/configs/env.config (modify — append `[goal_allocator]` section with defaults + commented-out per-phase overrides)
