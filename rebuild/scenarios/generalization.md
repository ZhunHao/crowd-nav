# Generalization report

This is the separate generalization report referenced by `splits.json`'s
`withheld_families` field. It states, in one place, what is held out of
training/calibration for benchmark purposes and why that is a weak but
non-zero generalization claim -- not statistical evidence.

## Withheld families

The eleven canonical encounter families are withheld from every seeded split:

- `head_on`
- `crossing`
- `crossing_mirrored`
- `overtake`
- `narrow_passage`
- `detour_harbour`
- `unreachable`
- `shore_goal`
- `multi_conflict`
- `course_change` -- a scripted, waypoint-interpolated target
  (`CourseChangeTraffic`) that makes a genuine turn at the crossing point,
  into the ego's lane, just before the ego arrives. Deterministic and
  immutable; not reactive.
- `reactive` -- a fixed initial voyage (`traffic_mode: 'reactive'`,
  frozen scenario file `noncooperative_reactive.json`) that the service
  wraps in `ReactiveTraffic` at run time, so the target reacts to
  whichever ego it actually faces. This handcrafted collision-responsive
  heading change is a controlled test rule only -- it is not COLREGs-compliant
  and not an ORCA/reciprocal-velocity-obstacle implementation. Because the
  realized path depends on the ego it meets, only the *initial* conditions
  are frozen here; a paired comparison across ego policies is expected to
  produce different realized target paths, not identical ones.

This list is exactly `scenarios/splits.json`'s `withheld_families` array; the
two are kept in sync by hand (there is a test that checks they name the same
families).

## Which splits contain them

| Split | Contains withheld families? |
|---|---|
| `test` | Yes -- exactly these eleven, one scenario each, and nothing else. |
| `train` | No -- only `family: "random"` scenarios (seeds 1000-1009, open map). |
| `dev` | No -- only `family: "random"` scenarios (seeds 2000-2004, land at (4,14,10,20) and (14,4,20,10)). |
| `calibration` | No -- only `family: "random"` scenarios (seeds 3000-3004, `maps/harbour.json`). |

## Held-out unit

The held-out unit for `test` is the **encounter family** (the named geometric
relationship between the ego route and the traffic's route -- head-on,
crossing, overtaking, a narrow passage, a harbour detour, an unreachable
goal, a goal hugging the shore, several simultaneous crossings, a scripted
course change, or a non-cooperative reactive target), not an individual
scenario, a map, or a seed. A policy or planner that only ever sees
`family: "random"` placements during training and calibration and is then
scored against these eleven named families is being asked whether it
generalizes to encounter *shapes* it was never shown, not whether it
memorized a specific start/goal/traffic tuple.

Note that some `test`-split families intentionally reuse a plain, open
24x24 layout that resembles `train`'s open map (`head_on`, `crossing`,
`crossing_mirrored`, `overtake`, `multi_conflict`, `course_change`, and
`reactive` all use it) -- the map itself is not the held-out variable. What
is withheld is the specific encounter geometry (the named family), which
never appears with that label, that traffic configuration, or that
combination of ego route and traffic route, in `train`, `dev`, or
`calibration`.

## Limits

- This is a **small, smoke-scale set**: eleven families, one scenario each, no
  repeated draws per family and no variation within a family (e.g. no
  `head_on` at a different speed or offset). It is a coverage checklist, not
  a statistically powered generalization benchmark -- a single pass/fail per
  family carries no confidence interval and is easily dominated by chance on
  any one run.
- `train`/`dev`/`calibration` seed counts (10/5/5) are themselves small; they
  bound what can be inferred about a policy's behavior on generic random
  traffic, independent of the withheld-family question above.
- Family withholding only says a *label* and its literal geometry did not
  appear in training; it says nothing about whether the underlying situation
  (e.g. "two vessels converging at a shallow angle") was implicitly present
  in some random draw under a different geometry. Treat a good `test` result
  as evidence of coverage of these eleven named shapes, not as proof of
  general collision-avoidance competence.
