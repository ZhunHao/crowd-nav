# ShipNav modern runtime

Real geographic maps and offline route planning are available through the `maps` extra. See [maps/README.md](maps/README.md) for the frozen Singapore sources, reproducible import, A*/Theta* comparison, preview commands, attribution and limits. The map backend uses wheel-backed Shapely, PyProj and Pyogrio; no NTNU or direct OSGeo/Fiona dependency is required.

The supported application stack is CPython 3.14.7 with the modern dependencies in `uv.lock`. Local acceptance covers macOS arm64 CPU, checkpoint inference and native ORCA, a visible Cocoa Qt window, PNG and FFmpeg export. Linux x86_64 wheel resolution requires glibc >=2.34; Linux execution, CUDA and MPS parity remain deferred.

From `rebuild/`, create the environment and install the separately hash-verified native wheel:

```sh
env -u PYTHONPATH VIRTUAL_ENV="$PWD/.venv-modern" uv sync --locked --all-extras --python 3.14.7 --active
.venv-modern/bin/python tools/install_rvo2.py
.venv-modern/bin/python tools/parity_fixture.py --output migration/inference.json
.venv-modern/bin/python tools/expanded_parity.py --output migration/expanded-modern.json
.venv-modern/bin/python tools/replay_migration.py --output migration/replay-modern.json
.venv-modern/bin/python -m pytest -q
.venv-modern/bin/python tools/smoke_modern.py
.venv-modern/bin/python tools/migration_summary.py
.venv-modern/bin/python tools/accept_modern.py
```

Use `.venv-modern/bin/python` for subsequent work. `uv sync` may remove the separately installed RVO2 wheel; rerun its installer afterward. Reproduction of the modern native build is documented in `migration/rvo2/README.md`. Application dependencies share one universal lock; `migration/linux-x86_64-requirements.txt` records the platform resolution. Future CUDA deployment needs its separately verified platform lock and the same application version and Python minor.

`migration/acceptance.json` records clean recreation and retirement evidence, versions, import isolation, retired path absence, and lock/source/checkpoint/wheel hashes. CPU action and value parity use frozen inputs and identical inference settings. All three 64-step standalone constant-velocity replay traces equal the frozen comparison traces and end in timeout; this records bounded failures and does not claim goal success or original simulator reproduction.

The temporary original runtime, environment, copied native build tree, setup recipe and capture entry points have been retired. `reference/` and frozen `migration/*legacy.json` and `migration/expanded-inputs.json` remain immutable verification evidence. The supplied source and checkpoints remain untouched input assets. Modern `shipnav.compat` modules preserve checkpoint inference behavior. Subsequent debugging uses this code and the recorded fixtures.

The TA later supplied a revised package, `CrowdNav-20251014-DIP`. It changes the simulator scenario (route, human spawning, render-only walls) but not the checkpoints or policy code. Read `migration/supplied-20251014-changes.md` before porting any environment or scenario code.

## Run and explore

Run these commands from `rebuild/` with the supplied model directory left unchanged:

```sh
.venv-modern/bin/python -m shipnav.gui
.venv-modern/bin/python -m shipnav.service --policy sarl --filtered \
  --map maps/harbour.json --start 2 12 --goal 22 12 --count 3 --seed 2000 \
  --model ../CrowdNav-20250813-DIP/crowd_nav/data/output_trained \
  --output results/run.json
```

The GUI's first row contains Load map, Load model, click mode, policy, ship count,
seed, Run, Stop, Replay and Export. Load a map, select **Set start**, click clear
water, then select **Set goal** and click the destination. **Add land (two
corners)** adds a rectangular obstacle with two clicks. Use the plot's pan/zoom
bar for kilometre maps; leave pan/zoom mode before placing endpoints. Changing
inputs invalidates the displayed result. Run computes in a worker; Stop requests
cooperative cancellation, retaining the partial trace. Replay uses recorded
frames and elapsed simulation time. Controls are disabled during a run/export;
wait for completion before closing.

The second row selects A* with smoothing or Theta*, holonomic or marine dynamics,
unfiltered or predictive filtering, exact or noisy/delayed observations, vessel
profile, uniform or route-corridor traffic, and replay speed. The GUI defaults to
the predictive filter; the CLI defaults to unfiltered unless `--filtered` is set.
The API also supports intermediate goals on/off, uncertainty margins on/off and
explicit observation parameters; CLI `--no-global-goals` targets the final goal
directly. One clock and persistent traffic span every intermediate goal.

## Vessel scale and units

Coordinates are Cartesian East–North. Metric maps and CLI map endpoints use
metres. Internally a profile maps vessel radius to 0.5, preferred speed to 1 and
timestep to 0.25. Let L = twice the physical radius and V = preferred speed:
lengths multiply by L, times by L/V, velocities by V. JSON/results retain these
model units and include `map.metadata.model_scale`; CSV, drawing and benchmark
physical metrics convert them back. Frozen scenarios already carry model units;
do not apply a second profile conversion.

| Profile | Radius / speed | Physical dt | Marine acceleration / yaw bounds | Grid / clearance |
| --- | --- | --- | --- | --- |
| model | 0.5 m / 1 m/s | 0.25 s | 0.2 m/s² / 0.35 rad/s | 1 m / 0.7 m |
| harbour_craft | 5 m / 5 m/s | 0.5 s | 0.5 m/s² / 0.175 rad/s | 50 m / 15 m |
| coastal_ship | 25 m / 7.5 m/s | 1.667 s | 0.225 m/s² / 0.0525 rad/s | 100 m / 50 m |

Traffic's nominal 0.3–1 model speed corresponds to 0.3V–V; scripted waypoints may
include waiting or other explicitly recorded speeds. Model-frame marine bounds
are 0.2 acceleration and 0.35 yaw rate. Synthetic episode limits default to 100
model seconds; scaled maps use at least three nominal planned-route traversal
times. Real-map results therefore need the recorded scale and limit for interpretation.

## Exports and metric meanings

Export the current result as JSON, CSV, PNG or MP4. JSON includes full frames,
truth/perception, nominal/executed actions, settings, scenario and model hashes,
source revision/diff and lock identity. PNG is a static overview. MP4 replays the
saved trace through FFmpeg; long runs automatically accelerate toward a two-minute
video, while the Python `export_run(..., speedup=1)` requests simulated real time.
No policy is rerun during export.

CSV has one row per frame; action fields describe the decision leaving that frame,
and are empty on the terminal row. `time_s`, `x_m`, `y_m`, `vx_mps`, `vy_mps` describe
physical time, position and incoming velocity; `goal_index` identifies the active
goal. `nominal_vx/vy`, `executed_vx/vy`, and `actual_vx/vy` are all physical m/s:
requested policy velocity, selected command, and integrated realized velocity.
`override` and `no_feasible_action` describe the filter; infeasibility is blank
when filtering is disabled. `decision_ms` is wall time. `max_target_age_s` is the
oldest perceived target age, `clearance_m` is truth ship clearance, and ship/land
collision flags come from swept collision checks. CSV `deadline_miss` and baseline timing tables compare recorded decision wall
time with the physical configured profile timestep. Raw JSON preserves the
original simulation diagnostic flag, which uses the model-time threshold.

Success means final-goal arrival within 0.5 model units. Report ship and land
collisions, timeout, planning/controller error and cancellation separately; a
planning failure remains in the complete denominator. Completion time is
success-conditioned. Distance and clearance are physical; own-route detour is
travelled distance divided by the run's own route length. Own-route cross-track
error measures controller tracking, not planner quality. The baseline additionally
uses one frozen reference per scenario for fair detour and tracking comparisons.
Land clearance is sampled at frames; collision detection is swept. Domain exposure
is time with truth ship clearance below the declared 1 physical metre threshold.
Overrides, infeasibility, solver failures and marine rate violations remain
visible even in successful runs.

## Experiments and baseline evidence

The historical ten-seed/count sweep is a smoke test:

```sh
.venv-modern/bin/python -m shipnav.evaluate --seeds 10 --counts 5 8 12 15 \
  --output results/evaluation
```

For frozen scenarios, `shipnav.benchmark.benchmark(scenarios, model_dir, output)`
runs every entry of `shipnav.benchmark.VARIANTS`, recording planner failures and
controller errors. Use the bounded-disk wrapper for the preregistered full matrix:

```sh
.venv-modern/bin/python tools/baseline_freeze.py
.venv-modern/bin/python tools/baseline_run.py pilot --output results/baseline-pilot
# Exact frozen commit/hash and held-out command are in evidence/baseline/README.md.
.venv-modern/bin/python tools/baseline_run.py audit --output results/baseline-heldout
```

[Baseline protocol and results](evidence/baseline/README.md) record the 110-scenario
corpus, all matched comparisons, CPU latency strata and scenario-paired bootstrap
intervals. Losslessly compressed traces and aggregate metrics live under ignored
`results/baseline-heldout/`; the protocol and summary are tracked. The extension
manifest preserves the historical scenarios and separates Ubin development from
Southern Islands geographic testing. Nearby synthetic perturbations provide
limited scenario diversity, not a sample of every maritime operating condition.
The supplied checkpoint is one fixed policy: scenario intervals do not constitute
independent training replication. Future training needs at least three independent
training seeds and hierarchical seed/scenario intervals.

## Controller limits

SARL and ORCA use a direct-to-goal fallback when there are no perceived neighbours;
this is recorded policy behavior, not learned empty-crowd inference. The supplied
SARL checkpoint was trained for pedestrian-style observations and does not observe
coastlines. Model hashes identify exact weight/config bytes, not training quality,
marine validation, or independent training seeds.

Predictions assume constant-velocity traffic, optionally inflated for uncertainty.
Scripted course changes and reactive targets deliberately violate that assumption.
Reactive paired runs share initial conditions and rules, but realized traffic may
respond differently to different ego trajectories. The filter searches predicted
motion and can report no feasible action; it does not prove collision avoidance.
Marine motion uses bounded acceleration and yaw with a circular footprint, not
calibrated hydrodynamics. Selected encounter rules and imported benchmark support
do not establish complete COLREGs compliance. Coastline maps lack depth, tides,
restricted waters and chart-grade safety semantics. CPU is validated; MPS, CUDA
and Linux runtime performance remain unvalidated.
