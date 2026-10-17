# Preregistered PPO retraining and held-out evaluation

All **10 / 10 training runs** completed: two variants (`ppo`, `ppo_lagrangian`) ×
five seeds, 100 logical chunks × 20,480 steps = 2,048,000 steps each, with no
partial checkpoint and no rerun. Every run had one checkpoint selected on the
generated selection split. The held-out evaluation then completed **2,530 / 2,530
episodes** (110 frozen scenarios × 23 arms). The audit verified every row and every
raw/stored archive hash.

With the predictive safety filter, all ten learned arms meet the preregistered
targets (success ≥ 80%, observed collision ≤ 5%). Without the filter, none does.
Retrained PPO is slightly below marine MPC on success and far above the supplied
SARL checkpoint. The Lagrangian constraint penalty did **not** lower held-out
domain-time violations. These are exploratory, unadjusted results on one scenario
corpus, not a claim of operational safety.

## Frozen inputs

| Input | Commit | SHA256 |
| --- | --- | --- |
| [Training configuration](training_config.json) | `982c844eff0c0d5369494b4c4253086fd0f1dff3` | `c415818debe8ab470db87080f2a932f016f1a55c05f3f3864e267685de603537` |
| [Held-out protocol](protocol.json) | `fc85f1ce888ab99d60c8d9fb06199acea73f32a2` | `36deb410c91921c3a65d6bbe513d312e4923f6b0c48c2d8c874b3c20a00bf3d6` |

The training configuration was committed alone before any full training run. Its
recorded training source is `7115da70092c388b7e3f03b18a926685326c73cd`; every
checkpoint manifest carries that commit. The configuration fixes:

- eight vector environments on CPU;
- the safety filter on during training, with exact sensing;
- a training corpus with SHA256 `94a71bf7…a747`;
- a selection corpus with SHA256 `eba4d2c7…227b`.

The Lagrangian budget is **6.688 domain-seconds per episode**. It is marine MPC's
mean `domain_time_s` over all 200 generated selection scenarios (163 successes,
8 collisions and 29 timeouts, all retained). Plain PPO receives no budget.

The protocol was committed alone after all ten selections and before any held-out
command. It pins the 110 test scenarios, the shared episode limits from the
[baseline](../baseline/README.md), the SARL checkpoint hashes, every selected
policy file hash, the source/lock hashes and the comparison list. The freezer
refused to run unless the tree was clean, `src/` was unchanged since the training
source, and every selection was a complete chunk of its own run.

## Training

The [EC2 pilot](pilot/README.md) measured throughput and verified stop/resume. On
9–16 October 2026, all ten full runs instead ran sequentially on a local Apple M2
(8 cores, macOS 26.6.2, CPU), one at a time. This was the user's choice. They used
the committed trainer, the frozen parameters and no AWS resources. Total learning
time was 15.4 hours.

| Run | Runtime (min) | Steps/s | Final training success | Final training collision | Final mean cost | Final multiplier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ppo/seed0` | 107.3 | 318 | 0.891 | 0.000 | 12.33 | — |
| `ppo/seed1` | 95.7 | 357 | 0.890 | 0.007 | 8.77 | — |
| `ppo/seed2` | 91.9 | 371 | 0.826 | 0.042 | 13.88 | — |
| `ppo/seed3` | 78.5 | 435 | 0.871 | 0.014 | 12.09 | — |
| `ppo/seed4` | 100.6 | 339 | 0.913 | 0.007 | 9.42 | — |
| `ppo_lagrangian/seed0` | 92.2 | 370 | 0.880 | 0.015 | 7.90 | 19.06 |
| `ppo_lagrangian/seed1` | 94.2 | 362 | 0.328 | 0.016 | 6.69 | 9.47 |
| `ppo_lagrangian/seed2` | 89.5 | 381 | 0.841 | 0.022 | 11.17 | 19.28 |
| `ppo_lagrangian/seed3` | 87.5 | 390 | 0.865 | 0.016 | 11.43 | 23.11 |
| `ppo_lagrangian/seed4` | 88.2 | 387 | 0.854 | 0.023 | 9.98 | 22.53 |

The "final" columns are the last training chunk's own episodes, not an
evaluation. `ppo_lagrangian/seed1` collapsed late in training: it met the budget,
but its success fell to 0.33. It was kept as trained, not rerun. Each run's
[training curve](curves/) is copied unchanged. All 1,000 checkpoint manifests were
hash-verified against the frozen source, variant, seed, environment count, chunk
size and budget.

## Checkpoint selection

`tools/select_checkpoints.py` evaluated 20 candidates per run on all 200 selection
scenarios: every fifth chunk, including the last. The rule picks the highest
success among candidates with collision ≤ 5%. Ties go to lower mean domain time,
then the earlier chunk. Every run had at least one such candidate, so the fallback
rule was never used. Selection used one thread per library and took about 70–95
minutes per run.

| Run | Selected chunk | Selection success | Selection collision | Mean domain s |
| --- | ---: | ---: | ---: | ---: |
| `ppo/seed0` | 54 | 0.865 | 0.025 | 10.65 |
| `ppo/seed1` | 84 | 0.855 | 0.030 | 10.43 |
| `ppo/seed2` | 59 | 0.870 | 0.020 | 9.63 |
| `ppo/seed3` | 24 | 0.865 | 0.035 | 10.19 |
| `ppo/seed4` | 34 | 0.865 | 0.025 | 9.42 |
| `ppo_lagrangian/seed0` | 49 | 0.865 | 0.025 | 9.95 |
| `ppo_lagrangian/seed1` | 44 | 0.850 | 0.025 | 11.13 |
| `ppo_lagrangian/seed2` | 24 | 0.870 | 0.030 | 10.75 |
| `ppo_lagrangian/seed3` | 24 | 0.855 | 0.020 | 11.07 |
| `ppo_lagrangian/seed4` | 9 | 0.860 | 0.035 | 9.33 |

Every [selection](selections/) lists all candidates and the chosen file hashes.
All ten selected checkpoints exceed the 6.688 s budget on the selection split.

## Held-out results

The run used single-threaded CPU and took 31.3 minutes of summed per-episode wall
time. The denominator is all 110 scenarios. It includes the nine deliberately
`unreachable` scenarios, which every arm records as `planning_failure`, so 101/110
(0.918) is the maximum achievable success. "Violation" means `domain_time_s` >
6.688 s, computed on the 99 synthetic-map scenarios only.

| Arm | Success | Collision | Outcomes | Violation rate | Southern Islands success |
| --- | ---: | ---: | --- | ---: | ---: |
| `marine_mpc` | 0.900 | 0.000 | 99 success, 2 timeout, 9 planning failure | 0.089 | 11/11 |
| `marine_mpc_unfiltered` | 0.900 | 0.000 | 99 success, 2 timeout, 9 planning failure | 0.089 | 11/11 |
| `marine_sarl` | 0.200 | 0.000 | 22 success, 79 timeout, 9 planning failure | 0.211 | 0/11 |
| `ppo_s0_filtered` | 0.836 | 0.000 | 92 success, 9 timeout, 9 planning failure | 0.100 | 11/11 |
| `ppo_s1_filtered` | 0.882 | 0.000 | 97 success, 4 timeout, 9 planning failure | 0.178 | 11/11 |
| `ppo_s2_filtered` | 0.836 | 0.000 | 92 success, 9 timeout, 9 planning failure | 0.100 | 11/11 |
| `ppo_s3_filtered` | 0.918 | 0.000 | 101 success, 9 planning failure | 0.133 | 11/11 |
| `ppo_s4_filtered` | 0.836 | 0.000 | 92 success, 9 timeout, 9 planning failure | 0.100 | 11/11 |
| `ppo_lagrangian_s0_filtered` | 0.882 | 0.000 | 97 success, 4 timeout, 9 planning failure | 0.133 | 11/11 |
| `ppo_lagrangian_s1_filtered` | 0.918 | 0.000 | 101 success, 9 planning failure | 0.100 | 11/11 |
| `ppo_lagrangian_s2_filtered` | 0.836 | 0.000 | 92 success, 9 timeout, 9 planning failure | 0.056 | 11/11 |
| `ppo_lagrangian_s3_filtered` | 0.918 | 0.000 | 101 success, 9 planning failure | 0.100 | 11/11 |
| `ppo_lagrangian_s4_filtered` | 0.836 | 0.000 | 92 success, 9 timeout, 9 planning failure | 0.200 | 11/11 |
| `ppo_s0_unfiltered` | 0.627 | 0.291 | 69 success, 32 collision, 9 planning failure | 0.000 | 6/11 |
| `ppo_s1_unfiltered` | 0.382 | 0.536 | 42 success, 59 collision, 9 planning failure | 0.000 | 6/11 |
| `ppo_s2_unfiltered` | 0.382 | 0.536 | 42 success, 59 collision, 9 planning failure | 0.000 | 6/11 |
| `ppo_s3_unfiltered` | 0.664 | 0.255 | 73 success, 28 collision, 9 planning failure | 0.000 | 7/11 |
| `ppo_s4_unfiltered` | 0.636 | 0.282 | 70 success, 31 collision, 9 planning failure | 0.000 | 7/11 |
| `ppo_lagrangian_s0_unfiltered` | 0.391 | 0.527 | 43 success, 58 collision, 9 planning failure | 0.000 | 7/11 |
| `ppo_lagrangian_s1_unfiltered` | 0.636 | 0.282 | 70 success, 31 collision, 9 planning failure | 0.000 | 7/11 |
| `ppo_lagrangian_s2_unfiltered` | 0.545 | 0.373 | 60 success, 41 collision, 9 planning failure | 0.000 | 6/11 |
| `ppo_lagrangian_s3_unfiltered` | 0.382 | 0.536 | 42 success, 59 collision, 9 planning failure | 0.000 | 6/11 |
| `ppo_lagrangian_s4_unfiltered` | 0.382 | 0.536 | 42 success, 59 collision, 9 planning failure | 0.000 | 6/11 |

Across seeds, filtered PPO averages 0.862 success (range 0.836–0.918). Filtered
PPO-Lagrangian averages 0.878 (0.836–0.918). Both have zero observed collisions.
Unfiltered, the means are 0.538 and 0.467 success, with 0.380 and 0.451 collision.
All 10 filtered learned arms and both MPC arms meet the targets. SARL and all 10
unfiltered learned arms do not.

Outcomes cluster by scenario family: each family is nine close variations. The
filtered arms' timeouts fall entirely in `narrow_passage` (9) or `overtake` (4),
which explains the repeated counts. Every arm's trace archive is distinct; no run
was duplicated. Unfiltered arms' violation rate of 0.000 is not a constraint
success: their collisions end episodes early. For example, `ppo_s1_unfiltered`'s
longest domain exposure was 4.25 s.

### Seed-and-scenario intervals

These are hierarchical bootstrap intervals: training seeds first, then scenarios.
Fixed comparators are replicated under each seed key. The two learned variants are
paired by seed index, which is an arbitrary pairing. Intervals are exploratory and
unadjusted for multiplicity.

| Comparison (a − b) | Success difference | Any-collision difference |
| --- | --- | --- |
| filtered PPO − marine MPC | −3.8 pp [−7.5, −0.2] | 0.0 [0.0, 0.0] |
| filtered PPO-Lagrangian − filtered PPO | +1.6 pp [0.0, +4.4] | 0.0 [0.0, 0.0] |
| filtered PPO − unfiltered PPO | +32.4 pp [+20.9, +45.1] | −38.0 pp [−50.5, −26.4] |
| filtered PPO − marine SARL | +66.2 pp [+61.3, +71.1] | 0.0 [0.0, 0.0] |
| marine MPC − unfiltered marine MPC | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

The filter is the dominant safety contribution for the learned policies. On this
corpus it changed no marine-MPC outcome.

### Constraint budget

Mean synthetic-map violation rates over seeds are 0.122 for filtered PPO (per
seed 0.100, 0.178, 0.100, 0.133, 0.100) and 0.118 for filtered PPO-Lagrangian
(0.133, 0.100, 0.056, 0.100, 0.200). Marine MPC's rate is 0.089 and SARL's is
0.211. The Lagrangian penalty raised training multipliers to 9–23, but it did not
measurably reduce held-out violations relative to plain PPO. Selection does not
use domain time except as a tie-break.

### Southern Islands (zero-shot transfer)

The 11 `southern_corridor` scenarios use the real Southern Islands map, which was
never in the training corpus. Every filtered learned arm and both MPC arms reached
all 11 goals. Unfiltered learned arms reached 6–7, and SARL reached none (all
timeouts). The domain budget is in physical seconds calibrated on 24–48 m
synthetic maps, so violation rates on this map are reported as not comparable.
Per-arm physical metrics are in `summary.json` under `southern_physical_metrics`.
For example, `ppo_s0_filtered` kept a 20.4 m mean minimum ship clearance against
MPC's 4.4 m. It also had a larger reference cross-track error (29.1 m against
4.0 m).

## Limitations

- One corpus of 110 scenarios in 12 families. The 99 canonical-family cases were
  inspected during development, and the same corpus earlier evaluated fixed
  policies.
- Five training seeds per variant. Seed pairing between variants is arbitrary.
- Training-time evaluation used generated scenarios. The selection split shares
  its generator with training but not its scenarios.
- A single CPU host. Timing is in `summary.json` and is not a real-time
  certification.
- Observed collisions of zero on 110 scenarios do not bound the true collision
  rate tightly, and none of this is a claim of operational safety.

## Reproduce and audit

From `rebuild/`, on the protocol commit:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv-modern/bin/python tools/retrain_eval.py heldout --output results/retraining-heldout \
  --protocol-commit fc85f1ce888ab99d60c8d9fb06199acea73f32a2 \
  --protocol-sha256 36deb410c91921c3a65d6bbe513d312e4923f6b0c48c2d8c874b3c20a00bf3d6
.venv-modern/bin/python tools/retrain_eval.py audit --output results/retraining-heldout \
  --protocol-commit fc85f1ce888ab99d60c8d9fb06199acea73f32a2 \
  --protocol-sha256 36deb410c91921c3a65d6bbe513d312e4923f6b0c48c2d8c874b3c20a00bf3d6
.venv-modern/bin/python tools/retrain_eval.py summarize --output results/retraining-heldout \
  --protocol-commit fc85f1ce888ab99d60c8d9fb06199acea73f32a2 \
  --protocol-sha256 36deb410c91921c3a65d6bbe513d312e4923f6b0c48c2d8c874b3c20a00bf3d6
```

The runner needs the selected checkpoints under `results/train/<variant>/seed<k>/`.
It verifies the committed protocol bytes and every pinned file hash before running,
and it resumes complete journal rows. Checkpoints, traces and logs live under
ignored `results/`. This folder tracks the configuration, protocol,
[summary](summary.json), [curves](curves/) and [selections](selections/).
