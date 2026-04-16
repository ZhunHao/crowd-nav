#!/usr/bin/env bash
# Reproduce the R1 baseline: load the trained SARL policy, run test_case 0,
# write exports/baseline.mp4. Headless (no matplotlib window) so it runs in CI.
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
cd "$here"

mkdir -p exports

# MPLBACKEND=Agg prevents any window pop-up on CI / ssh sessions.
export MPLBACKEND="${MPLBACKEND:-Agg}"

cd crowd_nav
# --visualize runs a single rollout per goal (vs. 500 test-size episodes per
# goal without it) and is what populates the position list that render() draws
# into the mp4. With MPLBACKEND=Agg set above, no matplotlib window is opened,
# so this remains CI-safe.
python test.py \
  --policy sarl \
  --model_dir data/output_trained \
  --phase test \
  --test_case 0 \
  --seed 42 \
  --visualize \
  --video_file ../exports/baseline.mp4
