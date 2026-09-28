# Solo ship-navigation rebuild design

Status: revised plan approved in scope on 2026-09-27; application implementation and environment verification have not started. The [research review](../research/2026-09-27-navigation-sota-review.md) now informs every phase below. New controls are experimental comparators, not demonstrated state-of-the-art performance.

## Scope and source

The briefing's practical goals (slides 8 and 13–16) remain: reproduce supplied DRL navigation; generate videos; distribute local goals; sustain moving traffic across regions; show land/static obstacles; automatically plan global-to-local routes; load maps and models, select endpoints, edit obstacles and show/export navigation through a GUI; run experiments and ablations. Literature work is excluded. Academic reports, competition submissions and presentation decks are outside this personal rebuild. Lightweight operating instructions remain included.

The user additionally approved the research revisions, requested migration to the newest modern versions after baseline verification, and requested AWS CLI for EC2 if retraining is included. The plan therefore includes controlled marine dynamics, observation uncertainty, predictive filtering, external-benchmark adapters and a retraining phase. Retraining follows a measured classical/pretrained baseline; it is needed for the new marine action/observation space, not for reproducing the original checkpoint.

## Global constraints

- Never use `codex` in branch names or worktree directory names.
- Preserve `CrowdNav-20250813-DIP/` and the supplied briefing unchanged.
- Implement only inside `rebuild/`; keep planning documents inside `docs/superpowers/` and Git-ignored.
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

## Evidence from the starter

The supplied `crowd_nav/data/output_trained/` includes RL/IL weights and policy/environment/training configs. The trained environment config differs from the root config. `test.py` hardcodes local goals and calls `light_reset()`, which resets `global_time`; the rebuild must not repeat this. The original renderer uses 8 FPS for 0.25-second steps; retain this timing discrepancy only in the historical reference.

SARL expects nonempty neighbour tensors; provide an explicit empty-neighbour fallback. Its `query_env` setting changes lookahead: compare migration parity with identical settings before switching modern experiments to observation-only constant-velocity prediction. Imports through `crowd_sim.envs` eagerly pull legacy Gym and RVO2, so changing only the top-level Gym import is insufficient. Port the small inference dependency closure into a namespaced package, preserving checkpoint parameter names. Git is initialized on `main`; `/docs/`, root PDFs and briefing slides are ignored. No commits were made by these revisions.

## Phase boundaries

1. **01 Reference:** original simulator, CPU checkpoint, bounded episodes, video, hashes and action/input/value fixtures; isolated Python 3.10 compatibility environment.
2. **01b Migration:** refresh official stable releases, resolve Python/wheels on local macOS and Linux, port SARL inference with original weights, verify numerical/action parity, then freeze modern locks and retire the temporary legacy environment/support under gate M4. No legacy fallback remains.
3. **02 Geometry and routes:** keep conservative rectangular test maps, A* plus smoothing, add Basic Theta*, compare route length/clearance/time on identical maps.
4. **03 Scenarios and simulation:** fixed serialized voyages, canonical encounters, truth/perception separation, delay/noise/dropout, uncertainty prediction, predictive action filtering and truthful traces. Retain an unfiltered holonomic baseline.
5. **03b Marine control and interoperability:** bounded acceleration/yaw-rate model, velocity-to-control tracker, sampled receding-horizon MPC, reactive traffic, independent feasibility checks, CommonOcean and NTNU compatibility spikes before adding full marine infrastructure.
6. **04 Application and evidence:** one headless service for GUI and CLI; safety/prediction diagnostics; reproducible exports; held-out scenarios, matched ablations, tail latency and paired uncertainty intervals.
7. **05 Retraining on EC2:** Gymnasium API and observation/action contract, prediction-aware PPO baseline, constrained-reward ablation with an explicitly logged constraint budget, multiple seeds, resumable EC2 jobs via AWS CLI, held-out evaluation and model loading in the same GUI.

## Interfaces and ownership

All implementation paths are relative to `rebuild/`.

| Files | Responsibility |
|---|---|
| Temporary `legacy/pyproject.toml`, `locks/legacy.txt`, `src/shipnav/baseline.py`; permanent `reference/` evidence | Baseline capture and migration comparison; remove legacy setup/runtime files at M4, retain results/hashes/fixtures |
| `pyproject.toml`, `uv.lock`, `.python-version`, `tools/release_inventory.py`, `migration/` | Modern dependency targets, platform validation, parity evidence |
| `src/shipnav/compat/`, `policies.py` | Namespaced inference port, checkpoint-compatible SARL and optional native ORCA adapter |
| `maps.py`, `planning.py`, `theta.py` | Clearance geometry, A* and Theta* |
| `scenarios.py`, `scenarios/`, `observations.py`, `prediction.py` | Stable IDs/hashes, frozen splits, measured observations, uncertainty |
| `simulation.py`, `safety.py`, `service.py` | Truth integration, filtering, episode trace, shared execution |
| `dynamics.py`, `guidance.py`, `controllers/mpc.py`, `adapters/` | Marine tier, external frame/unit/model adapters |
| `export.py`, `gui.py`, `evaluate.py`, `metrics.py`, `benchmark.py` | Replay, diagnostics and paired experiments |
| `training/env.py`, `training/train.py`, `training/checkpoints.py`, `cloud/` | Gymnasium, training state, AWS CLI launch/run/cleanup |
| `tests/`, `README.md` | Behavioral verification and local/cloud operating instructions |

The existing `execute(map_data, start, goal, model_dir='', policy_name='sarl', global_goals=True, seed=0, count=5, cancel=...)` remains a convenience wrapper for the GUI. Its extended keyword arguments select planner, supplied scenario, observation model, filter and dynamics. Benchmarking supplies a frozen scenario explicitly; no algorithm regenerates traffic. Recorded schema 2 retains the original display/export fields and adds scenario/config/lock hashes, truth/perception, nominal/executed action, prediction, intervention and latency records.

## Acceptance and interpretation

- Baseline artifacts exist before changing Python or inference code. CPU port parity compares transformed tensors, values and selected actions on identical observations; full trace parity is a separate closed-loop test. Any `query_env` change is a separate ablation.
- Newest stable is an attempt and a verified lock, not an assertion that all independently newest packages interoperate. A release inventory and compatibility report expose every exception. The legacy environment exists only until the modern acceptance gate passes; then remove it and its setup/runtime support. Subsequent regressions compare the modern implementation against saved evidence without restarting legacy.
- Both planners preserve clearance and report graph-resolution failure distinctly. A Theta* route need not always beat smoothed A*.
- Fixed scenarios are identical across comparisons. Reactive ships share initial conditions/policy/randomness; their realized paths can differ in response to ego motion. No mid-episode respawning or clock reset.
- Filters evaluate reachable motions before execution, log overrides and an explicit no-feasible-action result. Emergency commands maximize predicted separation; stopping is not presumed safe. A finite candidate/horizon test is not a formal safety guarantee.
- Marine controllers all obey the same dynamics and observation limits. The supplied pedestrian checkpoint is a transfer baseline; it is not a validated marine controller. Conservative circular footprints precede exact hull checks.
- Original GUI functions remain required. New diagnostics use the recorded trace, not re-simulated trajectories. GUI edits invalidate any saved scenario and require regeneration before running.
- Evaluation includes head-on, both crossings, overtaking, multi-conflict, narrow channel, detour, shore-adjacent goal, unreachable map, course change and noncooperative traffic. Hold out complete maps/families/seeds as documented in a split manifest; calibrate uncertainty on development/calibration data only.
- Report all statuses/denominators; separate ship and land collisions; measure clearance, domain exposure, detour, tracking/dynamics error, intervention/no-feasible-action/solver failure, inference and end-to-end mean/p95/p99/deadline misses. Ten seeds are smoke testing, not a performance claim. Use paired scenario intervals and several independent training seeds.
- External scenario feasibility and a few encounter rules do not establish full COLREGs compliance. Actual AIS ingestion, full chart/polygon editing, certified safety and hydrodynamic fidelity beyond the selected model are not claimed.
- Retraining checkpoints record optimizer, RNG, normalization, environment/split/lock hashes and constraint state. EC2 artifacts must survive instance termination and reproduce locally. Cloud execution requires an actual budget and regional resource selection; this revision launches nothing.
