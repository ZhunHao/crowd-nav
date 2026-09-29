# Pre-retraining CPU baseline

Completed all **1,320 / 1,320 arms**: 110 frozen scenarios × 12 variants, with no
infrastructure retry, missing arm or controller exception. There were 620
successes, 389 timeouts, 203 observed collisions and 108 deliberate planning
failures across the descriptive variant pool. Filtered ORCA and both marine MPC
variants meet the preregistered targets on this corpus. This does not establish
operational safety or compare unmatched dynamics tiers.

The [protocol](benchmark_protocol.json) was frozen before held-out execution at
commit `8611fa038f512b8725481212e790982fdd3380d1`, SHA256
`21cdd0d107785da0c64ba2b126599139662d9338e3290812e95b1931d94848da`.
The earlier protocol at `56dbf0c` was superseded **before any held-out run** to
replace planner-dependent default timeouts with one shared reference-route budget
per scenario. Source implementation is `54854cc`; all source/lock hashes and
checkpoint identities are listed in the frozen protocol.

The development [pilot summary](pilot-summary.json) contains 24 arms, from two
development scenarios. That tiny pilot cannot reliably estimate outcome variance:
the design uses the conservative Bernoulli variance ceiling 0.25 and nominal 95%
half-width 0.10, giving n=97, then applies the required minimum 100 and balanced
family/geographic allocation to choose exactly 110. No favorable-outcome stopping
or post-freeze additions are allowed. The controller targets, 80% success and at
most 5% observed collision, include the nine deliberate planning failures in the
complete denominator; they assess controller performance, not code acceptance.

## Reproduce and audit

From `rebuild/`, with the locked `.venv-modern` and supplied read-only checkpoint:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv-modern/bin/python tools/baseline_run.py heldout \
  --output results/baseline-heldout \
  --protocol-commit 8611fa038f512b8725481212e790982fdd3380d1 \
  --protocol-sha256 21cdd0d107785da0c64ba2b126599139662d9338e3290812e95b1931d94848da
.venv-modern/bin/python tools/baseline_run.py audit --output results/baseline-heldout
```

The runner verifies the committed protocol bytes, manifest hash and variant table,
then resumes complete journal rows without rerunning them. For a fresh independent
replication, select another output directory. Reproduce on the frozen source
revision; later revisions may intentionally change implementation. Before timing,
stop parallel test/acceptance jobs; the operating system and desktop still share
the CPU. The run pauses below 400 MiB free disk without deleting older evidence.

`results/baseline-heldout/` retains every trace, scenario and original per-unit
`metrics.json` losslessly as deterministic `.json.gz`; `rows.jsonl` is the complete
execution journal and `metrics.json` its aggregate. Each row contains stored-file
and original decompressed-byte hashes. `trace_hash` remains the original benchmark
hash; `trace_file` points to the compressed artifact. `load_verified(directory,
entry)` in `tools/baseline_evidence.py` checks both hashes before returning JSON.
For example, in Python with `tools` on `sys.path`:

```python
from pathlib import Path
import json
from baseline_evidence import load_verified
root = Path('results/baseline-heldout')
row = json.loads((root/'rows.jsonl').read_text().splitlines()[0])
run = load_verified(root, row['archive'])
```

Decompression recovers exact original bytes, including whitespace and number
formatting. Archive audit also checks recorded executed-run timeout against the
protocol's shared scenario limit; planning failures have no fabricated settings.
Original historical scenario files and their `scenarios/splits.json` hashes are
unchanged. The [extension](../../scenarios/baseline/splits.json) contains actual
endpoint/speed differences, plus 11 Southern Islands voyages; Ubin appears only
in development/calibration. Previously inspected canonical fixtures are held out
from training/tuning, not claimed unseen by all developers.

## Future runner reuse and dependency maintenance

The frozen baseline was independently audited at `8611fa0`; its protocol, summaries,
audit, test log and raw results remain unchanged. To reproduce or audit those
historical artifacts with their original tooling, check out
`8611fa038f512b8725481212e790982fdd3380d1` first and use the commands above. The
current runner intentionally rejects that protocol because its frozen source
hashes include the subsequently hardened tooling.

For a future study, freeze a new protocol on the intended source and locked
checkpoint before launch, and select a fresh output directory. The current
heldout, audit and summarize commands require its explicit `--protocol-commit`
and `--protocol-sha256`. They validate source, lock, checkpoint and manifests;
new directories record `run-identity.json`, and journal rows carry the protocol
identity. Resume accepts validated partial grids; final audit and summary require
the exact unique complete scenario × variant grid. Existing unbound directories
are rejected instead of relabelled. Pilot runs remain development probes without
a preregistered performance claim.

The frozen test log's 136 Qt/Matplotlib warnings concern Matplotlib's deprecated
`AA_UseHighDpiPixmaps` usage. Track this as a dependency maintenance follow-up:
verify an upstream fix in a separate locked runtime change and repeat the relevant
Qt/Matplotlib and native acceptance checks there. Do not suppress unrelated
warnings or upgrade the frozen runtime to polish historical evidence.

## Interpretation

All comparisons are scenario-paired A minus B, with 2,000 bootstrap resamples and
seed 9127. Success/collision include all 110 pairs. Completion time, fixed-reference
detour and tracking require both arms to succeed and both values to be finite;
excluded IDs and joint pair counts are reported. These conditional comparisons can
favor easier scenarios. Unadjusted intervals are exploratory, not confirmatory
superiority claims. A single SARL checkpoint supplies no training replication;
future trained policies need at least 3 independent training seeds (prefer 5) and
hierarchical seed/scenario analysis.

There are nine cases per canonical family, with small nearby perturbations, and
11 geographically held-out local voyages. The scenario bootstrap describes this
mixture; it does not establish geographic-population coverage or family-level
independence. The fixed geometric A*-smoothed reference is used for all planners;
null references explicitly mark unreachable cases. Physical distances/times use
recorded `scale.units`, while normalized model metrics and dimensionless detour
remain separate. Southern physical means are reported separately.

Inference timing covers the policy adapter call, including direct fallback when
no neighbours are perceived. Full-decision timing covers observation, inference,
prediction and filtering; it excludes planning, model loading, dynamics integration,
scoring, export and process startup. `cold_instance` is decision1 of a newly built
policy, decisions2–5 are transition, and decision6 onward is `warmed`. Caches may
already be warm; these labels do not claim process-cold timing or prove steady
state. Tables stratify CPU/platform, policy, dynamics, initial traffic count and
these timing categories, with mean/p95/p99, sample counts and physical-budget
misses. MPS, CUDA and Linux execution are unvalidated and unused.

## Scenario-screening transparency addendum

Before protocol freeze, `tools/freeze_scenarios.py:encounter_errors` called
`reaction_fires` for the original reactive fixture and its eight perturbations.
That validator runs a nominal Direct ego to check whether the target's reaction
rule activates; temporary regeneration tests repeated the same checks. The
validator inspects target velocity changes, not success/collision rates or policy
rankings. No case was selected or adjusted using a benchmark performance outcome.
These were scenario-definition screening probes under the existing generator's
encounter-validity contract. The corpus was therefore not wholly unexercised.
Preregistration applies to the fixed 1,320-arm performance matrix and its analysis;
it is an exploratory study, not an untouched confirmatory evaluation. Protocol
bytes remain immutable; this disclosure neither changes scenarios nor retunes
controllers after outcomes.

## Complete results

[Full summary and confidence intervals](summary.json), [per-scenario metrics and archive index](scenario-results.json.gz), [timing strata CSV](timing.csv), [integrity/runtime audit](audit.json), and [full test output](tests.txt) are tracked. The compressed per-scenario table publishes every frozen seed/ID, outcome, secondary metric, exclusion-relevant value and archive hash; exact timing arrays remain in each raw trace and the ignored full aggregate.

Every row below has n=110, including nine planning failures with unevaluated collision metrics. An observed collision rate of zero does not turn an unexecuted planning failure into a safe success. The rows describe distinct configurations; inferential comparisons are limited to the matched pairs below.

| Variant | Dynamics | Success | Ship / land collision | Timeout | Planning failure | Both targets met |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| sarl_reference | holonomic | 43 (39.1%) | 0 / 34 | 24 | 9 | no |
| sarl_theta | holonomic | 43 (39.1%) | 0 / 34 | 24 | 9 | no |
| sarl_no_goals | holonomic | 43 (39.1%) | 0 / 34 | 24 | 9 | no |
| sarl_filtered | holonomic | 35 (31.8%) | 0 / 0 | 66 | 9 | no |
| sarl_cv_filter | holonomic | 43 (39.1%) | 0 / 0 | 58 | 9 | no |
| sarl_degraded | holonomic | 9 (8.2%) | 0 / 0 | 92 | 9 | no |
| orca_reference | holonomic | 65 (59.1%) | 9 / 9 | 18 | 9 | no |
| orca_filtered | holonomic | 101 (91.8%) | 0 / 0 | 0 | 9 | yes |
| direct_reference | holonomic | 18 (16.4%) | 83 / 0 | 0 | 9 | no |
| marine_sarl | marine | 22 (20.0%) | 0 / 0 | 79 | 9 | no |
| marine_mpc | marine | 99 (90.0%) | 0 / 0 | 2 | 9 | yes |
| marine_mpc_unfiltered | marine | 99 (90.0%) | 0 / 0 | 2 | 9 | yes |

### Matched scenario comparisons

Values are **A minus B**, in percentage points, with exploratory 95% scenario-paired bootstrap intervals; every primary comparison uses all 110 pairs. Planner/goals equality in primary outcomes does not prove geometric or trajectory equivalence. The fixed-reference route is itself A*-derived, so its tracking score is adherence to a common nominated corridor, not proof of universally optimal routing.

| Factor (A − B) | Success difference [95% interval] | Collision difference [95% interval] | Joint successful completion-time pairs |
| --- | ---: | ---: | ---: |
| planner: sarl_reference − sarl_theta | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 43 |
| goals: sarl_reference − sarl_no_goals | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 43 |
| SARL_filter: sarl_reference − sarl_filtered | +7.3 [+1.8, +13.6] | +30.9 [+22.7, +40.0] | 34 |
| ORCA_filter: orca_reference − orca_filtered | -32.7 [-41.8, -23.6] | +16.4 [+10.0, +23.6] | 65 |
| MPC_filter: marine_mpc − marine_mpc_unfiltered | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 99 |
| prediction: sarl_filtered − sarl_cv_filter | -7.3 [-12.7, -2.7] | +0.0 [+0.0, +0.0] | 35 |
| sensing: sarl_filtered − sarl_degraded | +23.6 [+15.5, +31.8] | +0.0 [+0.0, +0.0] | 9 |
| matched_marine_controller: marine_sarl − marine_mpc | -70.0 [-78.2, -60.9] | +0.0 [+0.0, +0.0] | 22 |

Completion-time, reference-detour and reference-tracking intervals, joint eligibility counts and excluded scenario IDs are in `summary.json`. The matched marine SARL−MPC success difference is −70.0 percentage points [−78.2, −60.9] for this fixed mixture. These are single-checkpoint scenario comparisons, with no learned training-seed replication.

### Southern Islands physical results

This geographic subset contains 11 local voyages. Completion time includes successful runs only (its eligible count is shown); distance, clearance and fixed-reference tracking use all recorded traces, including timeouts/collisions. Units are converted with `scale.units`, not taken directly from model JSON.

| Variant | Success /11 | Completion n / mean (s) | Mean distance (m) | Mean minimum ship clearance (m) | Mean fixed-reference tracking (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| sarl_reference | 0 | 0 / — | 904.69 | 89.69 | 113.84 |
| sarl_theta | 0 | 0 / — | 904.69 | 89.69 | 113.84 |
| sarl_no_goals | 0 | 0 / — | 920.92 | 78.96 | 117.79 |
| sarl_filtered | 0 | 0 / — | 961.26 | 89.69 | 132.65 |
| sarl_cv_filter | 0 | 0 / — | 961.26 | 89.69 | 132.65 |
| sarl_degraded | 0 | 0 / — | 971.41 | 89.41 | 132.48 |
| orca_reference | 11 | 11 / 90.95 | 438.61 | 0.20 | 4.13 |
| orca_filtered | 11 | 11 / 91.86 | 443.00 | 1.83 | 5.14 |
| direct_reference | 0 | 0 / — | 161.36 | -1.45 | 0.05 |
| marine_sarl | 0 | 0 / — | 825.96 | 70.91 | 139.55 |
| marine_mpc | 11 | 11 / 92.86 | 437.74 | 4.35 | 4.03 |
| marine_mpc_unfiltered | 11 | 11 / 92.86 | 437.74 | 4.35 | 4.03 |

### CPU timing

Apple M2, macOS arm64, Python 3.14.7, one Torch intra/inter-op thread and bounded BLAS/OpenMP threads. Model loading/process startup are excluded. Cold-instance, transition and warmed strata for every traffic count are in `timing.csv` and `summary.json`; the table below shows pooled warmed decisions only. Counts are decision samples, not independent scenarios. Physical deadlines are 250 ms on synthetic maps and 500 ms for harbour_craft, evaluated per sample.

| Variant | Samples | Inference mean / p95 / p99 (ms) | Decision mean / p95 / p99 (ms) | Inference / decision deadline misses |
| --- | ---: | ---: | ---: | ---: |
| sarl_reference | 18635 | 15.993 / 20.573 / 43.759 | 16.026 / 20.631 / 43.819 | 0 / 0 |
| sarl_theta | 18635 | 15.593 / 20.452 / 24.945 | 15.625 / 20.495 / 25.017 | 0 / 0 |
| sarl_no_goals | 18617 | 15.960 / 20.205 / 26.504 | 15.997 / 20.253 / 26.750 | 1 / 1 |
| sarl_filtered | 30966 | 16.219 / 19.828 / 26.233 | 18.462 / 23.625 / 31.227 | 6 / 8 |
| sarl_cv_filter | 28542 | 16.084 / 20.119 / 28.191 | 18.254 / 24.180 / 30.956 | 1 / 1 |
| sarl_degraded | 38395 | 16.360 / 20.077 / 27.589 | 18.624 / 24.742 / 30.516 | 10 / 10 |
| orca_reference | 15289 | 0.008 / 0.011 / 0.017 | 0.038 / 0.053 / 0.074 | 0 / 0 |
| orca_filtered | 12128 | 0.011 / 0.018 / 0.033 | 1.521 / 5.198 / 5.988 | 0 / 0 |
| direct_reference | 3884 | 0.001 / 0.001 / 0.002 | 0.020 / 0.045 / 0.062 | 0 / 0 |
| marine_sarl | 34895 | 16.370 / 20.219 / 29.135 | 21.594 / 28.574 / 40.243 | 8 / 11 |
| marine_mpc | 10935 | 7.790 / 16.416 / 19.414 | 12.723 / 26.805 / 31.817 | 0 / 0 |
| marine_mpc_unfiltered | 10935 | 7.665 / 16.442 / 18.581 | 7.710 / 16.492 / 18.647 | 0 / 0 |

Inference samples include adapter/direct-fallback work when no neighbours are perceived. OS scheduling and desktop activity were not isolated, so these timings are observed local CPU performance, not a hard real-time guarantee. No acceptance tests or parallel policy jobs ran during the held-out matrix.

### Verification and storage

Full suite: **302 passed**, 136 dependency deprecation warnings (`AA_UseHighDpiPixmaps` in Matplotlib/Qt), 96.01 s. Archive audit verified all **2,750 unique archives**, raw and stored hashes, exact preregistered run order, complete scenario×variant grid, source/lock/model identities, matched controls and shared episode budgets. All 1,212 executed traces identify protocol commit 8611fa0 with an empty tracked diff; the 108 explicit planning failures have no fabricated execution provenance.

Unique archives retain 589,762,196 original bytes in 70,682,630 compressed bytes. Sum of per-unit service/metrics wall time was 3875.4 s; this is separate from the inference/decision timing distributions. Original frozen assets and hashes remain unchanged. No training, cloud operation, MPS/CUDA run or Linux runtime claim was made.
