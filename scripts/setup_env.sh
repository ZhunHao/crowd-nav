#!/usr/bin/env bash
# Idempotent installer for the CrowdNav-DIP baseline (R1 / WP-1).
# Safe to re-run — skips work already done.
set -euo pipefail

ENV_NAME="${ENV_NAME:-navigate}"
PY_VERSION="3.10"

here="$(cd "$(dirname "$0")/.." && pwd)"
cd "$here"

log() { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }

# 1. Conda env ----------------------------------------------------------------
if ! command -v conda >/dev/null; then
  echo "conda not found — install Miniconda from https://www.anaconda.com/download/success" >&2
  exit 1
fi

# Make `conda activate` usable inside this script.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "conda env '$ENV_NAME' already exists — reusing"
else
  log "creating conda env '$ENV_NAME' (python $PY_VERSION) from conda-forge"
  # --override-channels -c conda-forge avoids the Anaconda ToS prompt on fresh miniconda installs.
  conda create -y -n "$ENV_NAME" --override-channels -c conda-forge "python=$PY_VERSION"
fi
conda activate "$ENV_NAME"

# 2. Python-RVO2 (vendored) ---------------------------------------------------
log "installing cython (required by Python-RVO2 build)"
pip install --quiet "cython<3"

if python -c "import rvo2" 2>/dev/null; then
  log "rvo2 already installed — skipping vendored build"
else
  log "building vendored Python-RVO2"
  # CMake >= 4 dropped support for cmake_minimum_required < 3.5. The vendored
  # RVO2 C++ library declares an older minimum, so we set the policy version
  # explicitly to stay compatible with modern cmake.
  export CMAKE_POLICY_VERSION_MINIMUM="${CMAKE_POLICY_VERSION_MINIMUM:-3.5}"
  pushd Python-RVO2-main >/dev/null
  rm -rf build
  python setup.py build
  python setup.py install
  popd >/dev/null
fi

# 3. crowd_sim + crowd_nav ----------------------------------------------------
log "pip install -e . (crowdnav package, editable)"
pip install --quiet -e .

log "pinning gym==0.15.7 (newer versions break this env)"
pip install --quiet "gym==0.15.7"

# 4. Dev / test deps ----------------------------------------------------------
log "installing pytest for smoke tests"
pip install --quiet pytest

# 5. ffmpeg sanity ------------------------------------------------------------
if ! command -v ffmpeg >/dev/null; then
  echo "WARNING: ffmpeg not found — video export will fail." >&2
  echo "  Linux : sudo apt install ffmpeg" >&2
  echo "  macOS : brew install ffmpeg" >&2
fi

log "running preflight checks"
if ! crowdnav-preflight; then
  echo "preflight failed — see hints above" >&2
  exit 1
fi

log "done. Activate with:  conda activate $ENV_NAME"
