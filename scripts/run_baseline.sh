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
python test.py \
  --policy sarl \
  --model_dir data/output_trained \
  --phase test \
  --test_case 0 \
  --seed 42 \
  --video_file ../exports/baseline.mp4
