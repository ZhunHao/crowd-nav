# Rebuild decisions

Checked against the briefing and current plans on 2026-09-27. These are design decisions, not completed implementation. The briefing does not prescribe most numerical choices below. The user's personal rebuild excludes literature work and academic deliverables. Legacy is temporary and is removed after modern acceptance; saved baseline evidence remains.

## 1. Global planner and waypoint selection

**Briefing:** combine global routing with DRL through intermediate goals (slide 8); integrate a global planner, giving Theta* as an example (slide 13).

**Decision:** eight-neighbour grid A* with Euclidean costs/heuristic and greedy line-of-sight smoothing is the default baseline. The smoothing algorithm repeatedly selects the furthest subsequent route point that can be connected safely. Basic Theta* is an additional selectable planner and paired comparison. It relaxes edges through a node's parent when there is line of sight; it is not simply the smoothing pass. Both use a default 1 m grid and 0.7 m land/boundary clearance. Both reject blocked endpoints and report an explicit no-route result at the chosen resolution. Theta* is not predeclared the winner.

## 2. Number and spacing of local goals

**Decision:** adaptive to the safe route, with no fixed count or equal spacing. The selected route includes start and final destination; execute the points after start in sequence. Intermediate goals are its internal vertices. A straight unobstructed route needs only the final goal; start equal to destination is immediately complete. In the holonomic reference, reaching within the configured radius (default 0.5 m) advances the goal index without resetting the episode.

The 1 m search grid is not a 1 m waypoint-spacing rule. No maximum segment length or minimum goal spacing has been selected. Marine-specific curvature/lookahead tuning remains an implementation/evaluation task; the bounded tracker and safety filter must handle the selected route, and limitations must be reported. The briefing asks that local goals not overlap in the same region, but provides no numerical separation or number.

## 3. Dynamic obstacles between phases

**Decision:** one continuous episode, with one clock and persistent traffic. Freeze and serialize traffic independently of the chosen planner before the run. Ships do not respawn or teleport when the ego reaches a waypoint. Initially they follow fixed straight voyages and stop at their destination. The marine phase adds reactive/course-changing traffic and explicitly noncooperative tests. Paired reactive experiments use the same initial conditions, rules and random inputs; realized trajectories may differ because ego behavior differs.

Observation noise, delay and dropout affect what a controller sees. Collision scoring still uses simulated truth. Retain the original traffic behavior only during baseline capture; do not reproduce `light_reset()` as the new waypoint-transition mechanism.

## 4. Land source, resolution and format

**Decision:** synthetic Cartesian maps first, in metres, with East–North `(x,y)` coordinates. The initial harbour covers `[0,0,24,24]` m and has rectangular land `[9,5,14,18]`. JSON stores `bounds` and a list of axis-aligned land rectangles. Geometry is continuous; 1 m is the planner's initial discretization, not the precision of a coastline raster.

CommonOcean scenario import and an NTNU simulator compatibility investigation follow in phase 03b. No real harbour, AIS dataset, chart provider, geographic projection or real-map resolution has been chosen for the first demo. The briefing's AIS discussion is background; it does not make real AIS ingestion an implementation requirement. Imported external data must preserve its provenance, units, timestep and coordinate conversion.

## 5. Land treatment in local control

**Decision:** land is a collision constraint, beyond the briefing's minimum of drawing visual static obstacles. The global planner inflates land by 0.7 m (default 0.5 m body radius plus 0.2 m planning margin). Local SARL provides a nominal velocity without being assumed to understand coastlines. The predictive filter checks candidate reachable motion against land/bounds and predicted moving vessels before execution. Truth integration separately detects swept collisions.

The initial filter checks 12 steps of 0.25 s (3 s), using the body radius for land checks; its settings and this difference from global clearance must be recorded. It logs nominal/executed commands, intervention, predicted clearance and no-feasible-action outcomes. It chooses the best scored candidate when none is feasible; no universal safety guarantee follows. Keep filter-off runs for ablations. The marine version rolls out bounded turning/acceleration instead of assuming instantaneous velocity changes.

## 6. Fresh model training

**Briefing:** reproduce/use a supplied DRL model; it does not explicitly demand a fresh full training run.

**Decision:** no retraining to reproduce the supplied SARL baseline or prove its modern inference port. The expanded rebuild includes a later fresh-training phase for the changed marine observation/action contract. Train a modern PPO baseline and a constraint-penalty ablation through Gymnasium after measuring pretrained/classical controllers. The existing weights remain a transfer baseline, not a policy with magically changed input dimensions.

Use AWS CLI for EC2 discovery, launch, SSM execution, checkpoint transfer and teardown. Choose account, region, instance, runtime and budget at execution after a CPU/GPU pilot. These cloud choices and exact training budgets are not yet fixed. Use at least three independent training seeds, preferably five if the budget supports them.

## 7. Experiments, ablations, metrics and thresholds

**Experiments:** baseline/parity; open-water tracking; head-on; crossing from both sides; overtaking; multiple conflicts; narrow passages; island/channel detours; shore-adjacent goals; unreachable routes; course changes; noncooperative and reactive traffic. Compare exact observations with noise, delay and dropout. Freeze train/development/calibration/test splits, including held-out maps/seeds and a separate withheld-family generalization test.

**Ablations:** A* smoothing versus Theta*; intermediate goals on/off; SARL/direct/ORCA/MPC with matched dynamics and observations; safety filter on/off; constant-velocity prediction versus uncertainty margins; exact versus degraded sensing; holonomic versus bounded marine dynamics as a separately labelled tier; pretrained versus newly trained policies and constraint penalty on/off. Avoid changing multiple controls in a comparison presented as one ablation.

**Scale:** original count sweep 5/8/12/15 with ten seeds is a smoke test, not a briefing-mandated experiment size. Start comparative testing with at least 100 distinct held-out scenarios and adjust sample size from pilot precision. Use paired scenario confidence intervals and hierarchical training-seed/scenario analysis for learned policies.

**Metrics:** success, ship collision, land collision, timeout, planning/controller errors and cancellation; clearance; time within a declared ship domain; completion time; distance/detour; tracking error; heading/acceleration violations; safety interventions/infeasibility; solver failures; inference-only and full-decision mean/p95/p99 latency; control-deadline misses. Keep failed runs in denominators and calculate completion time separately for successful runs.

**Functional thresholds selected:** final-goal arrival within default 0.5 m, default episode limit 100 s and step 0.25 s; no intermediate goal counts as final success. Every planned segment must pass clearance checks. Numerical migration comparisons use defined tolerances in plan 01b. Core behavior tests and modern smoke checks must pass before legacy retirement. The marine demo limits are 0.2 m/s² acceleration and 0.35 rad/s yaw rate, not measured full-size ship parameters. Record any decision exceeding its 250 ms control period at the default timestep.

**Still open:** an overall success-rate target, maximum acceptable collision rate, required improvement over SARL/ORCA, hard p95/p99 target, constraint-cost budget and statistically justified final sample size. Neither the slides nor the current plans establish values such as “95% success” or “zero collisions guaranteed.” Set performance targets in the experiment protocol before the held-out run; do not invent them after seeing results.

## 8. Export formats

**Decision:** JSON for full trace/settings/scenario/model and environment identity; CSV for tabular trajectories/diagnostics/metrics; PNG for static overviews; MP4 for replay. Replay comes from the recorded trace at `1/dt`, normally 4 FPS. Preserve the original renderer's 8 FPS only in the historical baseline and document its twice-speed playback. The slides request result export and simulation videos without specifying formats.

## 9. GUI, OS and packaging

**Decision:** PySide6 (Qt 6), Matplotlib QtAgg, background execution, cooperative cancellation and recorded replay. Provide map/model loading, start/destination selection, two-corner rectangular obstacle editing, path display and exports. Add planner/dynamics/filter/perception controls and diagnostic overlays. GUI and CLI call the same service.

Local target is macOS Apple Silicon; Linux x86_64 is the EC2 training target. The plans do not promise a Linux desktop GUI, Windows support or native cross-platform installers. Run from the source package in the locked modern environment. A `.app`, DMG, executable bundler, signing and release-distribution workflow have not been selected and are outside initial acceptance. PyQt appears only as an example in the briefing, not a mandate.

## 10. Report, presentation and deadlines

The original slides ask for a final report, competition presentation, presentation slides and demo user manual. They do not specify report length, presentation duration or submission dates. The 13-Aug-2025 date is the briefing date, not a submission deadline.

For this personal rebuild, literature review, final academic report and competition presentation are excluded. A concise README/user guide remains required. No self-imposed completion deadline has been set.
