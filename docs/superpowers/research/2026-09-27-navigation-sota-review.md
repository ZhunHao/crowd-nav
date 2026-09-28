# State-of-the-art review for the personal ship-navigation rebuild

Research date: **27 September 2026**. Scope: decisions informing the revised baseline, migration, planning, simulation, marine, application and training plans. This is engineering research requested by the user, not a reinstated academic literature-review deliverable.

## Recommendation

Keep the original SARL reproduction as a reference experiment. For a stronger rebuild, prioritize **predictive collision avoidance, realistic action constraints, independent encounter scenarios, and uncertainty evaluation** before adopting a larger learned model.

My recommended progression is:

1. Reproduce the original checkpoint and simulator unchanged.
2. Build a deterministic, independently specified scenario and evaluation layer.
3. Keep A* plus smoothing as the first global baseline; add Theta* as a comparison.
4. Compare SARL and classical local control under identical observations and traffic.
5. Add a predictive action filter, with explicit failure handling and intervention logs.
6. Add heading/speed dynamics and a model-predictive controller as a separate maritime tier.
7. Only then consider prediction-aware constrained RL or fresh training.

This is an engineering recommendation inferred from the sources below. I did not find evidence supporting one universally best method across pedestrian crowds, constrained waterways, different ship models, and open-water encounters. Reported paper results are not directly comparable without matching conditions. The initial minimal rebuild used SARL and holonomic motion; the accepted revisions add the boundaries and experiments below without claiming demonstrated maritime SOTA.

## What the relevant research establishes

### 1. The supplied SARL approach is a historical baseline

The original CrowdNav work models human–robot and human–human interactions using attentive aggregation. The author repository identifies it as ICRA 2019 work. It supports reproducing the supplied architecture, not a claim that its pedestrian-trained checkpoint solves ship dynamics, grounding avoidance or maritime rules. [Original paper](https://arxiv.org/abs/1809.08835), [author repository](https://github.com/vita-epfl/CrowdNav/).

**Plan implication — my assessment:** keep Task 1. Preserve the original configuration and record a reference trace. When the rebuilt simulator changes obstacle behaviour and sets `query_env=False`, identify that as a transfer experiment. Do not attribute the resulting performance difference solely to global planning.

### 2. Prediction and uncertainty are useful directions beyond SARL

CrowdNav++ combines recurrent graph attention with predicted multi-step pedestrian trajectories; its ICRA 2023 repository includes pretrained models and constant-velocity versus learned-prediction options. Its documented tested environments are older Ubuntu/Python configurations. It is a useful intermediate reference, not a verified macOS replacement. [Author implementation](https://github.com/Daniel-DHu/Hu_CrowdNav_Prediction_AttnGraph).

GenSafeNav, presented at CoRL 2025 according to the authors' project page, combines adaptive conformal uncertainty estimates with constrained RL. It evaluates changes in crowd velocity, policy and group behaviour and includes real-robot experiments. The authors provide training/model code; their quick-start uses a CUDA Docker environment. These results concern pedestrian navigation, so transferring the method to ships requires new modelling and evaluation. [Project and publication information](https://gen-safe-nav.github.io/), [paper](https://arxiv.org/abs/2508.05634), [official code](https://github.com/tasl-lab/GenSafeNav).

**Plan implication — my assessment:** introduce a prediction interface and delayed/noisy observations before importing another neural network. Start with constant-velocity predictions and a documented uncertainty margin. A hand-tuned margin is not conformal calibration and must not be described as such. Prediction-aware constrained learning is a later experiment requiring training and calibration data separate from the test set.

### 3. Explicit safety constraints matter more than an additional collision penalty

A September 2025 preprint combines robust MPC and control barrier functions for ASV navigation. Its reported simulations concern static obstacles, waterway boundaries and flow disturbances; it explicitly leaves stability and recursive-feasibility guarantees to parallel work. It therefore informs the design, but does not establish a general safety guarantee for our multi-ship problem. [Paper, including simulation scope and Remark 1](https://arxiv.org/html/2509.06687v1).

A newer maritime preprint, **Credibility-Aware Learning and Control for Safe USV Navigation under Perception Uncertainty**, combines uncertainty-weighted learning with a covariance- and recovery-aware CBF-QP filter. Version 5, dated 30 August 2026, includes braking/turning recovery and COLREGs-aware preferences. It is simulation evidence; peer-reviewed publication and a reproducible author code release were not verified in this review. Earlier indexed versions have a different title and describe a different shield, so use the versioned source. [August 2026 version](https://arxiv.org/html/2605.26974v5).

**Plan implication — my assessment:** add a separate `SafetyFilter` between the nominal policy and motion integration. Initially roll out a finite set of reachable candidate actions against predicted traffic and land over a short horizon. Reject unsafe candidates, preserve the nominal action when feasible, and record every override. If no feasible candidate exists, report the condition and select a documented least-risk emergency action; a stopped vessel can still be struck. This finite-horizon filter is a practical prototype, not a formal CBF guarantee.

### 4. New maritime RL papers exist, but do not justify choosing an algorithm by headline score

**PHG–PPO**, published in Ocean Engineering on 30 July 2026, combines expert guidance and curriculum learning for rules-aware collision avoidance; the institutional publication record describes radar/AIS-derived encounters and Imazu tests. The publisher full text was blocked in this session. I verified the institutional abstract and publication metadata, not its full experimental protocol or runnable code. [Author institution record](https://researchportal.ulisboa.pt/en/publications/rules-compliant-ship-collision-avoidance-based-on-progressive-hyb/).

**Multi-agent PPO–LSTM**, Journal of Network and Computer Applications, July 2026, uses recurrent cooperative learning and tests Imazu encounters and an ENC-derived waterway environment. Available publisher search text supports this description, but direct full-text access returned HTTP 403 and code availability was not verified. [Publisher record](https://www.sciencedirect.com/science/article/pii/S108480452600086X).

**Plan implication — my assessment:** treat both as recent research directions. A cooperative multi-agent training setup is a different problem from controlling only the ego vessel among uncooperative traffic. For this personal project, reproducible baselines and clear experiments are better selection criteria than paper-specific percentage improvements.

### 5. Global planning should match the motion model

Theta* is an established any-angle planning method. Its published comparisons distinguish it from A* with post-smoothing; Basic Theta* does not guarantee a true shortest continuous path. It is a useful upgrade/comparison for our geometric map, not a new 2026 algorithm. [Original conference paper, 2007](https://aaai.org/Papers/AAAI/2007/AAAI07-187.pdf), [extended paper](https://arxiv.org/abs/1401.3843).

A January 2025 maritime study compares RRT, RRT*, Informed RRT* and PQ-RRT* for nonholonomic planning. The reported improvements in route length involve runtime/tuning trade-offs. The article's data-availability statement limits access to its exact experimental software. Do not assume every later open repository reproduces those results. [Publisher paper](https://link.springer.com/article/10.1007/s10846-025-02222-7).

**Plan implication — my assessment:** retain A* plus smoothing for the first working baseline, then compare Theta* using path length, runtime, clearance and failures. Once bounded turning/acceleration becomes part of the vessel model, evaluate dynamically feasible planning/control. A geometrically short route with sharp corners is not sufficient evidence of a navigable ship trajectory.

### 6. ORCA remains useful, with a specific assumption

ORCA's reciprocal formulation assigns each agent part of the pairwise avoidance responsibility. The original project describes each agent taking half. That assumption is not satisfied by all constant-course or deliberately non-compliant traffic. [Authors' ORCA page](https://gamma-web.iacs.umd.edu/ORCA/).

**Plan implication — my assessment:** keep ORCA as a comparator, label whether surrounding vessels cooperate, and test both cooperative and non-cooperative traffic. Do not infer maritime rule compliance or universal safety from the use of ORCA. A velocity-obstacle controller and a model-predictive controller would provide useful additional classical comparisons.

### 7. Prefer established maritime infrastructure where it saves effort

**CommonOcean** provides composable vessel motion-planning benchmarks and scenario tooling. Its separate drivability checker covers collisions, water boundaries and model feasibility, including limited yaw rate. These checks are distinct from a full maritime-rules checker. [Benchmark project](https://commonocean.cps.cit.tum.de/), [drivability documentation](https://commonocean-documentation.readthedocs.io/en/latest/commonocean-dc/doc/docs/source/index.html).

**NTNU colav-simulator**, open-sourced in autumn 2025, provides scenario configuration, ship models, Gymnasium integration, chart-related functionality and evaluation support. Its documented North–East coordinate convention differs from an unqualified Cartesian plotting convention; conversion needs tests. Its chart ecosystem introduces native/geospatial dependencies, so local compatibility needs an installation spike. [Research-lab repository](https://github.com/ntnu-itk-autonomous-ship-lab/colav-simulator).

**NTNU rlmpc** supplies research implementations involving SAC and NMPC. Its README labels the cited SAC work unpublished, while the anti-grounding NMPC reference is an ACC 2024 paper. Treat those evidence levels separately. [Author code and citations](https://github.com/ntnu-itk-autonomous-ship-lab/rlmpc).

**Fossen's Python Vehicle Simulator** includes USV and ship models with guidance/control utilities, including Nomoto-based ship examples and an Otter USV model. It is a useful dynamics reference rather than a drop-in benchmark replacement. [Author implementation](https://github.com/cybergalactic/PythonVehicleSimulator).

**Plan implication — my assessment:** keep small synthetic maps for unit tests. Add a CommonOcean scenario adapter and consider an NTNU-simulator evaluation adapter before building a full marine simulator from scratch. Save the chosen scenario IDs, model parameters, coordinate transform and upstream revision. No upstream package was installed or executed for this review.

### 8. Modernize the new environment independently of the legacy reference

Gymnasium distinguishes natural termination from time-limit truncation and uses `reset(seed=...)`; the distinction affects RL bootstrapping. Its migration guide is not evidence that this particular Gym 0.15.7 code can be migrated without adaptation. [Official migration guide](https://gymnasium.farama.org/introduction/migration_guide/).

**Plan implication — my assessment:** use a separate temporary legacy environment for baseline reproduction, then remove it and its runtime/setup support when modern acceptance passes (user-directed lifecycle update). Retain measured results and frozen fixtures only as evidence. If fresh training is selected, expose the rebuilt environment through Gymnasium and validate its API. Do not freeze the entire future application to the legacy stack merely because the supplied checkpoint needs it. Resolve exact current dependency versions during the compatibility spike and record them afterward.

## Concrete changes to the existing plans

These recommendations were accepted and incorporated into the revised plans on 2026-09-27, including the new migration, marine-control and EC2-training phases. Application code has not been implemented.

| Priority | Existing plan | Recommendation | Proposed new files / test focus |
|---|---|---|---|
| First | 01 baseline | Separate legacy reproduction from modern experiments; add deterministic checkpoint/config trace comparison | `reference_manifest.json`; baseline action/trace regression tests |
| First | 02 planning | Keep geometric baseline; add Theta* comparison and route-clearance metrics | `planning/theta_star.py`; identical-map planner comparisons |
| First | 03 simulation | Decouple scenario generation from whichever global planner is under test; serialize traffic definitions once | `scenarios.py`; same scenario hash for every comparator |
| First | 03 simulation | Separate ground-truth state from perceived state; support delay/noise/dropout | `observations.py`, `prediction.py`; noisy observations must not change scoring truth |
| First | 03 policies | Add predictive filtering and log nominal/executed actions separately | `safety.py`; unsafe land/crossing actions, intervention records, no-feasible-action cases |
| First | 04 evaluation | Add canonical encounters and held-out maps instead of only crowd count × random seed | `scenarios/encounters/*.json`; frozen train/dev/test partitions |
| First | 04 evaluation | Expand metrics and matched ablations | `metrics.py`; ship/land collisions, minimum clearance, deadline misses, interventions |
| Next | 03 dynamics | Add a heading/speed model and tracker as a distinct simulation tier | `dynamics.py`, `guidance.py`; turning/acceleration constraints and tracking error |
| Next | 03 policies | Add an MPC comparator with the same dynamics and observation limits | `controllers/mpc.py`; solver failure/deadline/fallback tests |
| Next | 04 evaluation | Import external maritime scenarios and validate feasibility | `adapters/commonocean.py`; coordinate/units/body-footprint tests |
| Later | New training stage | Train prediction-aware or constrained RL on the selected marine dynamics | `training/`; held-out seeds, calibration partition, compute budget and checkpoints |
| Keep | 04 GUI/export | Retain shared headless service, replay, map editing and exports; expose new diagnostics | Nominal vs executed path, predicted traffic, clearance, filter interventions |

All proposed code paths are relative to the future `rebuild/src/shipnav/` unless they explicitly name scenario assets or a manifest. They map to tasks in the revised plan index; implementation files are not yet created. Read the revised plans for final filenames and APIs.

## Proposed evaluation design

This protocol is my recommendation, not a performance claim from a paper.

**Scenario families:** open-water tracking; head-on; crossing from both sides; overtaking; multiple simultaneous conflicts; narrow passage; island/channel detour; goal near shore; unreachable route; a target that changes course; a target that does not cooperate.

**Progressive observation conditions:** exact state; noisy position/velocity; delayed reports; intermittent target loss. Generate observation perturbations from scenario/time/target-indexed random streams so actions and early termination do not change the exogenous noise sequence across paired runs.

**Traffic conditions:** fixed trajectories first, then reactive traffic. For reactive traffic, match initial conditions, policy and exogenous randomness; resulting trajectories may legitimately differ in response to the ego vessel. Do not demand identical realized trajectories in that case.

**Metrics:** final-goal success; ship collision; land collision; timeout; planning failure; minimum body clearance; duration within the selected ship domain; travelled distance and detour; cross-track error; heading/acceleration feasibility; action-filter intervention rate; solver failures; mean/p95/p99 action latency and control-deadline misses. Measure end-to-end decision latency separately from neural inference. Once rule logic is implemented, report encounter-specific rule tests and applicability rather than claiming comprehensive legal compliance.

**Ablations:** A* smoothing versus Theta*; SARL versus a classical controller under the same motion model; global intermediate goals on/off; action filter on/off; constant-velocity versus uncertainty-aware prediction; exact versus degraded observations. Change one main factor at a time. Retain a fair unfiltered baseline instead of comparing a filtered new system only against an unfiltered old system.

**Statistics:** use the current ten-seed batch as a smoke test. For comparative claims, predefine a larger held-out scenario set, report denominators and uncertainty intervals, and use paired analysis. For learned methods, evaluate several independent training seeds, not merely many rollouts from one trained model. Zero observed collisions in a small batch does not demonstrate rare-event safety.

## Scope and evidence limits

- No universal leaderboard winner was established. A newer publication date alone is not evidence of better performance for this project.
- Source quality varies: original author code, publisher/institutional publication records, and versioned preprints are identified above. Abstract-only findings are labelled.
- Full-text limitations are material: the RMPC-CBF paper's scenarios are narrower than a general multi-vessel problem, and its theory is not a blanket guarantee.
- I did not reproduce the performance of any external algorithm or verify that its repository installs on this Mac.
- Pedestrian-trained checkpoints are not validated marine policies. Moving to different state/action dimensions can require a controller adapter or retraining; it is not just loading the same weights into another simulator.
- Canonical encounter tests do not prove complete COLREGs compliance. The project should remain a research/demo system.
- LLM-based maritime decision systems are also appearing, including CORALL with Imazu and hardware-in-the-loop evaluation, but they add a different set of dependencies and validation questions. They are not needed for the current goal. [Accepted manuscript record](https://uhra.herts.ac.uk/id/eprint/27245/).

## Accepted direction (2026-09-27)

For this personal rebuild, choose **a modernized version of the original project**: reproduce SARL, add robust scenario/evaluation boundaries and a predictive safety layer, then add realistic vessel control as a second tier. This keeps an achievable first demo while leaving useful experiments to explore. A full from-scratch constrained-RL maritime research system is now covered by a separate training/simulator plan with explicit experimental limits.
