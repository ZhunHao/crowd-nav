# Solo ship-navigation rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the original practical project, then modernize and evaluate the approved research extensions.

**Architecture:** Capture baseline evidence, migrate to the newest stable stack and retire legacy once verified, then share one scenario/simulation contract across planning, marine control, GUI, evaluation and training.

**Tech Stack:** Temporary Python 3.10 baseline; newest verified stable Python/PyTorch/Gymnasium/Qt application; AWS CLI EC2 for retraining

**Spec:** [Revised design](../specs/2026-09-27-solo-rebuild-design.md)

## Global Constraints

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

---

All implementation paths below are relative to `rebuild/`. Run commands there. These are instructions and reference snippets, not installed software or benchmark results.

## Start here

For the selected choices and remaining open items, see [rebuild decisions](../2026-09-27-rebuild-decisions.md).

All requested revisions are incorporated into these documents. This is a detailed self-implementation plan, with file ownership, interface contracts, code/test examples and phase gates. It does not implement the application, install dependencies, run baseline verification, train a model or launch AWS resources. The original source remains intact. Git is initialized on `main`; documentation and supplied PDF/PPTX artifacts remain ignored.

Read the [revised design](../specs/2026-09-27-solo-rebuild-design.md), then follow the order below. Complete mandatory revision tasks within each phase; early reference snippets are scaffolding, not that phase's final architecture. The [research review](../research/2026-09-27-navigation-sota-review.md) explains source quality and why no universal “SOTA winner” is claimed. This research informs engineering; no academic literature deliverable is added.

| Order | Plan | Completion gate / environment |
|---|---|---|
| 1 | [01 Baseline](2026-09-27-01-baseline.md) | Original CPU episode/video, checkpoint/config/source hashes and fixtures; `.venv-legacy` only |
| 2 | [01b Modern migration](2026-09-27-01b-modern-migration.md) | Fresh stable-release inventory, modern parity, platform smoke tests and locks; M4 removes legacy; `.venv-modern` |
| 3 | [02 Maps and planners](2026-09-27-02-maps-and-planning.md) | Clearance geometry, A* and Basic Theta* compared on identical maps; modern |
| 4 | [03 Simulation and policies](2026-09-27-03-simulation-and-policies.md) | Complete 6b–6d: frozen scenarios, perception/prediction/filter, shared schema-2 service; modern |
| 5 | [03b Marine and benchmarks](2026-09-27-03b-marine-and-benchmarks.md) | Bounded tracker, sampled MPC, reactive traffic, external adapter compatibility/feasibility gate; modern plus isolated upstream spikes |
| 6 | [04 GUI, exports and evaluation](2026-09-27-04-gui-exports-and-evaluation.md) | Original GUI features plus safety diagnostics, held-out matched experiments and intervals; modern |
| 7 | [05 Training and EC2](2026-09-27-05-training-and-ec2.md) | Gymnasium parity, local training/resume smoke, CPU/GPU pilot, budgeted AWS CLI runs, independent seeds and local model replay; modern Linux CUDA lock |

Plan 03's final marine service depends on 03b modules; implement 03's holonomic safety tasks first, 03b next, then finish 03 Task 6d. Training refactors the already tested episode loop into a shared step engine; rerun phase 03/04 parity gates after that refactor.

## What changed from the initial plan

- Newest stable migration now happens immediately after reference verification. Old Python/NumPy/Gym pins are temporary baseline tools. Once modern acceptance passes, delete the legacy environment and runtime/setup support; retain only verification evidence and the supplied input assets. Current candidate versions and exact compatibility evidence are in 01b; versions are refreshed at execution rather than assumed compatible.
- A* remains a baseline; Theta* is an additional comparison. Traffic is serialized independently of the selected route/planner.
- Safety selection happens before execution, with nominal/executed actions and infeasibility recorded. Truth, observations and predictions have distinct interfaces and tests.
- A marine tier adds turning/acceleration limits, reachable-action filtering, reactive ships and an MPC comparator. CommonOcean/NTNU are installation/API/feasibility tasks with explicit coordinate and dependency boundaries.
- Evaluation includes canonical encounters, noise/delay/dropout, held-out maps/seeds, matched ablations, tail latency, paired intervals and independent training seeds.
- Retraining uses a Gymnasium interface and adds model/optimizer/RNG/normalization/manifest checkpoints. AWS CLI covers resource discovery, launch JSON/dry-run, SSM jobs, S3 recovery and EC2 cleanup. Account, region and spending limit are execution inputs, not guessed now.

## Original requirement coverage

| Practical briefing goal | Revised location |
|---|---|
| Install/run/debug supplied DRL and generate video | 01 reference and 01b parity |
| Distributed local goals and continuous navigation | 02 planners; 03 continuous episode tests |
| Moving traffic across navigation regions | 03 frozen scenario suite; 03b reactive traffic |
| Land map/static obstacles and automatic global planning | 02 maps, A*/Theta*; 04 editing |
| Rule-based and learned local navigation | 03 SARL/direct/ORCA; 03b MPC; 05 trained policy |
| Load maps/models; choose start/destination; add obstacles | 04 GUI and 05 new model adapter |
| Visualize/replay/export navigation | 04 JSON/CSV/PNG/MP4 plus prediction/action diagnostics |
| Experiments/ablations and visualizable results | 04 frozen protocol, every-run artifacts and uncertainty intervals |
| Lightweight user instructions | README updates in 01b, 04 and 05 |
| Academic literature/report/competition slides | Excluded from implementation critical path |

## Review and verification

Verification of this revision on 2026-09-27: **61 embedded Python blocks parsed**, **31 dependency-light behavioral checks passed**, and **2 checkpoint/native integration tests were deselected**. The checks used the host's existing Python 3.9.6 in a temporary directory, not the proposed modern environment. They covered geometry, both planners, scenario identity, perception/dropout, safety filtering, dynamics/MPC, service/export/metrics, and frame parity through the coroutine refactor. The AWS runner passed `bash -n`; relative documentation links resolved. No AWS command was executed. Existing host Matplotlib emitted deprecation warnings.

The plan's code samples are references to implement and test phase by phase. Dependency-light snippets are checked in a temporary directory; native legacy/RVO2, modern Python wheels, Qt/FFmpeg, upstream maritime APIs, Gymnasium training and AWS commands require their actual execution gates. A successful snippet test is not a verified dependency migration or trained policy.

After M4, use only the modern runtime; keep baseline results/hashes/fixtures for regression checks. Keep original source/checkpoints read-only, keep docs ignored, and commit only named future `rebuild/` files after behavioral tests pass. Do not commit large generated environments/results. Update locks only through a recorded migration, not during a comparison run.

For your own rebuild, start at 01. If you later want agent execution, choose inline execution with checkpoints or explicitly request subagent execution. Neither begins merely by approving this plan revision.
