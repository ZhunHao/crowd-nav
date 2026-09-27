# Supplied revision 2025-10-14 (`CrowdNav-20251014-DIP`)

The TA supplied a second package later in the project. It was delivered as
`CrowdNav-20250813-DIP-2` and has been renamed to `CrowdNav-20251014-DIP` after the
date of its last edits. The original package `CrowdNav-20250813-DIP` is unchanged
and remains the source the rebuild was ported and verified against
(`reference/manifest.json`, `src/shipnav/compat/PORT_NOTES.md`).

The complete raw diff is in `supplied-20250813-to-20251014.diff` (`.DS_Store`,
`__pycache__` and `*.pyc` excluded).

## Scope

Only three files changed and one backup file was added. Checkpoints,
`crowd_nav/configs/*`, `crowd_nav/policy/*`, `crowd_sim/envs/utils/*`,
`Python-RVO2-main/`, `setup.py` and `README.md` are byte-identical.

| File | SHA-256 (2025-10-14) |
|---|---|
| `crowd_nav/data/output_trained/env.config` | `e6a58c196befcee2af5c18af830ec3f43139ecbfc9e20ccdb21113b68351e27e` |
| `crowd_nav/test.py` | `1e17fed6e32730725eb924090111a01734e05ea73b61581d1cb024ea50d888a1` |
| `crowd_sim/envs/crowd_sim.py` | `e5fc6a67f1d1bd91eac55daa56ce1af6ef866a0378f5284b3a6d5993a970b883` |
| `crowd_sim/envs/crowd_sim.py.bak` | new; intermediate draft, see below |

## Changes

### 1. Trained-run config (`crowd_nav/data/output_trained/env.config`)

This is the config `test.py` loads alongside the checkpoint.

| Key (`[sim]`) | 2025-08-13 | 2025-10-14 |
|---|---|---|
| `human_num` | 4 | 3 |
| `static_obs` | false | true |

`static_obs_num = 15` and `static_obs_shapes = rect` are unchanged and are not
read by the new obstacle code.

### 2. Robot route (`crowd_nav/test.py:98`)

The start is still `[-11, -11]`. The chained local-goal list grows from 4 to 10
goals and snakes through the new walls:

- 2025-08-13: `[0,0] → [10,10] → [5,-5] → [-5,5]`
- 2025-10-14: `[-5,-9] → [0,-10] → [6,-9] → [10,-5] → [5,0] → [-2,-1] → [-8,2] → [-7,10] → [0,11] → [6,11]`

Each leg still starts from the previous leg's final robot position and calls
`env.light_reset()`. `env.curr_post = [-11,-11]` and `env.local_goal = [0,0]` are
still hard-coded before the loop, so the humans for the first `reset()` are
spawned around `[-11,-11] → [0,0]`, not around the first route leg.

### 3. Human spawn ellipse (`crowd_sim.py`, `generate_circle_crossing_human_new`)

Humans are spawned on an ellipse around the robot's current leg (current
position → local goal) and aim for the diametrically opposite point.

| Quantity | 2025-08-13 | 2025-10-14 |
|---|---|---|
| centre | `(min + max) / 3` (not the midpoint) | `(min + max) / 2` (midpoint) |
| radius per axis | `(max - min) / 3` | `(max - min) / 1` (3× larger) |
| spawn noise per axis | `±v_pref` | `±v_pref / 2` |

If a leg is axis-aligned (for example `[0,11] → [6,11]`), one radius is zero and
humans spawn on a line segment, spread only by the noise.

### 4. Static obstacles (`crowd_sim.py`, `light_reset` / `reset`)

- `self.static_obstacles = []` moved from `light_reset()` into `reset()`, so
  obstacles persist across the per-goal `light_reset()` calls.
- When `static_obs` is true, `reset()` builds a hard-coded maze of 1 m × 1 m
  squares (`{'type': 'rect', 'cx', 'cy', 'w': 1, 'h': 1}`):

| Wall | Cells |
|---|---|
| Left, `x = -15` | `y = -15 … 14` (30) |
| Right, `x = 15` | `y = -15 … 14` (30) |
| Bottom, `y = -15` | `x = -15 … 14` (30) |
| Top, `y = 15` | `x = -15 … 14` (30) |
| Inner, `y = -5` | `x = -15 … 4` (20) |
| Inner, `y = 5` | `x = 15 … -4` (20) |

The corner cell at `(15, 15)` is missing.

**The obstacles are rendered only.** `static_obstacles` is read in `render()`
(`crowd_sim.py:663`) and nowhere else. It is not used in collision checks,
rewards, observations, or the ORCA humans. The robot and humans can pass
through walls. The walls constrain behaviour only because the route goals are
placed around them.

### 5. `crowd_sim/envs/crowd_sim.py.bak`

This is an earlier draft (16:27, the final file is 16:50). It has all of change 3
and change 4, plus a different, unused `goal_list` in
`generate_random_human_position` (`[[0,-9],[6,-6],[5,0],[-8,2],[-5,10]]`). The
final `crowd_sim.py` keeps the original `[[2,-2],[-4,2],[0,4],[-3,0]]`, which is
dead code in both versions. Treat the `.bak` as non-authoritative.

## Impact on the rebuild

- **Inference, checkpoints and parity evidence:** unaffected. The checkpoint,
  policy and model code are identical, and the `shipnav` port has no dependency
  on the changed files.
- **Scenario and simulator work:** `shipnav` currently has no environment,
  route or human-spawn code. When that work is ported, choose which supplied
  revision is the target and record it. To reproduce the 2025-10-14 scenario,
  port changes 1 to 4 as written, including the render-only walls. Making walls
  physical (collision, observation or ORCA obstacles) would be a new behaviour,
  not a port. Record it as a deliberate change.
- **Existing evidence:** `reference/` and `migration/*legacy.json` were captured
  from 2025-08-13 and stay valid for that revision. Do not regenerate them from
  2025-10-14.
