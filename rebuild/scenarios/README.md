# Frozen scenarios

Generated deterministically by `tools/freeze_scenarios.py`; do not hand-edit
the JSON files here. Each `<name>.json` is a complete scenario in the
`shipnav.scenarios.make_scenario` schema (full map, `start`, `goal`, a
`traffic` list, `traffic_mode`, `family`, `split`, `seed`) -- a
self-contained initial condition, not a seed recipe that a planner or a
future version of the generator has to reproduce. Most `traffic` entries are
plain `Traffic` fields with `traffic_mode: "scripted"`; the `course_change`
family instead carries a `kind: "course_change"` entry with a `waypoints`
list (`CourseChangeTraffic`), and the `reactive` family carries plain
`Traffic` fields but `traffic_mode: "reactive"` (the fixed initial voyage the
service wraps in `ReactiveTraffic` at run time).
`splits.json` records, per split, each file's name, its SHA-256 (over the
canonical-JSON bytes) and its `scenario_hash` (over the parsed scenario
dict), plus `withheld_families` and any `failures`.

## Families

Eleven canonical encounter families, all in the `test` split, each on a 24x24
map unless noted, ego `(3,12)->(21,12)` unless noted, default `Traffic`
speed `.3` and radius `.6` unless noted. The nominal ego used to time the
encounters is a holonomic Direct ego on the `astar_smooth` route at 1 m/s
(so on the default route it is at `x = 3 + t`). Where the plan fixes a
family's endpoints the conflict is timed by target speed; otherwise by the
target's start position.

- `head_on` -- target `(21,12)->(3,12)`; meets the ego at `x ~ 16.8`.
- `crossing` -- target `(12,3)->(12,21)` at speed `1.0`, reaching the
  crossing point `(12,12)` at `t=9` together with the ego.
- `crossing_mirrored` -- target `(12,21)->(12,3)` at speed `1.0` (same timing).
- `overtake` -- target `(7,12)->(21,12)`, speed `.3` (plan-specified; the
  ego catches it at `x ~ 8.7`).
- `narrow_passage` -- land `(8,0,12,10)` and `(8,14,12,24)`, leaving a gap
  at `y` in `[10,14]` for `x` in `[8,12]`. Target `(12.1,12)->(3,12)` at
  `.3` starts just east of the gap and meets the ego inside it (`x=10`,
  `t=7`); the shared `y=12` centreline keeps 2m clearance to each block.
- `detour_harbour` -- `maps/harbour.json` (land `x` in `[9,14]`, `y` in
  `[5,18]`), ego `(2,12)->(22,12)`; the planned route detours north along
  `y=19.5`. Target `(7.7,20)->(22,20)` (2m clear of the block's north edge)
  is caught up by the ego alongside the harbour (`x ~ 11.5`); target
  `(22,4)->(2,4)` (1m clear of the south edge) is background traffic.
- `unreachable` -- land `(10,0,14,24)` spans the full map height, splitting
  it in two. No traffic: the point is that the goal is unreachable, not
  that traffic is dense. The service raises `NoPath`.
- `shore_goal` -- land `(23.7,0,24,24)` along the east edge, goal
  `(22.9,12)`, no traffic. The goal is .8m off the shore strip: above the
  planner's .7m clearance, so a genuine near-shore approach runs (marine
  runs report `terminal_speed`/`runout_min_clearance` for it).
- `multi_conflict` -- two nonoverlapping crossings, both on a collision
  course: `(8,3)->(8,21)` at `1.8` (meets the ego at `t=5`) and
  `(16,21)->(16,3)` at `9/13` (meets it at `t=13`).
- `course_change` -- scripted, deterministic `CourseChangeTraffic` (not
  reactive) with waypoints `(0,(12,10.2)) -> (6,(12,12)) -> (26,(18,12))`:
  north at `.3`, then an actual turn at the crossing point `(12,12)` into
  the ego's lane, heading east at `.3`. The ego passes `x=12` at `t=9`, after
  the turn, and closes on the target ahead of it. `traffic_mode: "scripted"`.
- `reactive` (frozen file `noncooperative_reactive.json`) -- a plain fixed
  `Traffic((14,8.7),(14,21))` voyage timed to reach the ego's route at `t=11`
  as the ego passes `x=14`, tagged `traffic_mode: "reactive"`. The frozen
  scenario stores only the initial voyage; wrapping it in `ReactiveTraffic`
  (a handcrafted, controlled-test collision-responsive heading change -- not
  COLREGs-compliant, not ORCA/reciprocal-velocity-obstacle) happens in the
  service at run time, not here. Because the realized path depends on which
  ego it faces, a paired comparison across ego policies is expected to
  produce different realized target paths for this family, not identical
  ones.

## Freeze-time validity checks

`tools/freeze_scenarios.py` checks every canonical family before writing it
and records a `failures` entry (`name`, `seed`, `split`, `error`) instead of
writing -- or adjusting -- a scenario that fails:

- every family except `unreachable` plans with `astar_smooth`;
  `unreachable` must raise `NoPath`;
- every encounter family's nominal straight-line CPA (nominal ego above vs.
  the frozen voyage) is below `ego radius .5 + target radius + 0.5 m`
  (centre distance), for every target in `multi_conflict`;
- the CPA lies at the crossing point (`crossing`, `crossing_mirrored`),
  inside the gap (`narrow_passage`), alongside the harbour
  (`detour_harbour`), or at/after the turn (`course_change`);
- the `reactive` target's reaction actually fires against a Direct ego.

## Split construction

- `test` -- the eleven canonical families above, one scenario each, `seed: 0`.
- `train` -- `make_scenario` draws on an open 24x24 map, seeds `1000..1009`
  (10 scenarios), 4 traffic ships each, ego `(2,2)->(22,22)`.
- `dev` -- `make_scenario` draws on a 24x24 map with land `(4,14,10,20)` and
  `(14,4,20,10)` (geometrically distinct from `calibration`'s harbour land),
  seeds `2000..2004` (5 scenarios), 3 ships each, ego `(2,2)->(22,22)`.
- `calibration` -- `make_scenario` draws on `maps/harbour.json`, seeds
  `3000..3004` (5 scenarios), 3 ships each, ego `(2,2)->(22,22)`.

Each seeded split uses its own map and a seed range disjoint from the other
two splits, so no scenario is shared or re-derivable across splits. A seed
whose placement raises (`make_scenario`'s retries exhausted) is recorded in
`splits.json`'s `failures` list (`seed`, `split`, `error`) and is never
resampled with a different seed -- resampling would let easy seeds silently
replace hard ones.

Every canonical and seeded scenario is validated the way `run_episode`
would validate its own inputs (waypoints clear of land, traffic clear of
land, no traffic overlapping the start or other traffic at `t=0`) before it
is written; a validation failure is recorded in `failures`, never silently
adjusted.

## Generalization note

`withheld_families` in `splits.json` lists the eleven canonical encounter
families. They appear only in `test` and never in `train`, `dev` or
`calibration` -- those splits only ever see `family: "random"` scenarios.
A policy or planner that only sees random encounters during training or
calibration and is then scored on these named families is being tested for
generalization to encounter geometries it has not seen, not for
memorization of them. See `scenarios/generalization.md` for the full
report: which splits contain the withheld families, what the held-out unit
for `test` is (the encounter family, not the map or the seed), and this
set's limits as a small smoke-scale check rather than statistical evidence.

## Noise keying and target identity

`Observer` noise is keyed on `(scenario seed, tick, target index)` (see
`shipnav.observations.Observer.observe`), so the same scenario replayed with
the same observer seed reproduces identical perceived noise deterministically.
This keying is by list index, not a stable per-target ID. If a later task
allows targets to be inserted or removed mid-episode, targets will need
stable IDs (not positional indices) so that noise draws and any per-target
history stay attached to the same target across such changes.
