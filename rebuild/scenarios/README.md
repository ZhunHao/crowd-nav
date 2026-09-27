# Frozen scenarios

Generated deterministically by `tools/freeze_scenarios.py`; do not hand-edit
the JSON files here. Each `<name>.json` is a complete scenario in the
`shipnav.scenarios.make_scenario` schema (full map, `start`, `goal`, a
`traffic` list of `Traffic` fields, `traffic_mode: "scripted"`, `family`,
`split`, `seed`) -- a self-contained initial condition, not a seed recipe
that a planner or a future version of the generator has to reproduce.
`splits.json` records, per split, each file's name, its SHA-256 (over the
canonical-JSON bytes) and its `scenario_hash` (over the parsed scenario
dict), plus `withheld_families` and any `failures`.

## Families

Nine canonical encounter families, all in the `test` split, each on a 24x24
map unless noted, ego `(3,12)->(21,12)` unless noted, default `Traffic`
speed `.3` and radius `.6` unless noted:

- `head_on` -- target `(21,12)->(3,12)`.
- `crossing` -- target `(12,3)->(12,21)`.
- `crossing_mirrored` -- target `(12,21)->(12,3)`.
- `overtake` -- target `(7,12)->(21,12)`, speed `.3` (matches the default;
  named explicitly since the family is about a slower target ahead of the
  ego on the same line).
- `narrow_passage` -- land `(8,0,12,10)` and `(8,14,12,24)`, leaving a gap
  at `y` in `[10,14]`. Target `(21,12)->(3,12)` transits the same gap in the
  opposite direction to the ego; the shared `y=12` centreline keeps 2m
  clearance to each land block.
- `detour_harbour` -- `maps/harbour.json` (land `x` in `[9,14]`, `y` in
  `[5,18]`), ego `(2,12)->(22,12)`. Two targets run east-west clear of the
  block: `(2,20)->(22,20)` (2m clearance to the block's north edge) and
  `(22,4)->(2,4)` (1m clearance to its south edge).
- `unreachable` -- land `(10,0,14,24)` spans the full map height, splitting
  it in two. No traffic: the point is that the goal is unreachable, not
  that traffic is dense.
- `shore_goal` -- land `(23.5,0,24,24)` along the east edge, goal
  `(22.9,12)`, no traffic. The goal is .6m off the shore strip: clear at
  the .5m ego radius used for validation (`.6 > .5`) but tight enough that a
  planner may legitimately report `NoPath`. That is the point of this
  family; endpoints are not tuned to make planning easy.
- `multi_conflict` -- two nonoverlapping crossings, `(8,3)->(8,21)` and
  `(16,21)->(16,3)`.

## Split construction

- `test` -- the nine canonical families above, one scenario each, `seed: 0`.
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

`withheld_families` in `splits.json` lists the nine canonical encounter
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
