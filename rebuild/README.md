# Supplied CPU baseline

This temporary baseline directly configures the supplied CrowdSim and SARL checkpoint. Original simulation, reward, policy and 8 fps rendering behavior are preserved. The macOS wrapper resolves FFmpeg from PATH only during rendering and restores Matplotlib settings afterward.

Verified on Python 3.10.20, macOS 26.6.2 arm64, Apple clang 21.0.0 (clang-2100.1.1.101), CMake 3.31.10, FFmpeg 9.0.1, torch 2.2.2, Gym 0.15.7, NumPy 1.26.4 and Matplotlib 3.8.4. Full platform/package evidence is in `reference/manifest.json` and `reference/acceptance.json`.

From workspace root:

```sh
uv venv --python 3.10 --seed rebuild/.venv-legacy
cd rebuild
export PYTHONDONTWRITEBYTECODE=1
export PATH="$PWD/.venv-legacy/bin:$PATH"
python -m pip install 'setuptools<70' wheel 'Cython<3' 'cmake<4'
mkdir -p .venv-legacy/vendor
cp -R ../CrowdNav-20250813-DIP/Python-RVO2-main .venv-legacy/vendor/Python-RVO2
(cd .venv-legacy/vendor/Python-RVO2 && python -m pip install --no-build-isolation .)
python -m pip install -r locks/legacy.txt
python -m pip install --no-deps -e .
python - <<'PY'
from pathlib import Path
import site
Path(site.getsitepackages()[0], 'crowdnav-reference.pth').write_text(str(Path('../CrowdNav-20250813-DIP').resolve()) + '\n')
PY
python -m pytest -v
MPLBACKEND=Agg python -m shipnav.baseline ../CrowdNav-20250813-DIP/crowd_nav/data/output_trained
ffprobe -v error -show_entries stream=codec_name,nb_frames,r_frame_rate,duration -of json results/baseline/baseline.mp4
MPLBACKEND=Agg python tools/capture_reference.py
python tools/parity_fixture.py --output reference/inference.json
python tools/reference_manifest.py
```

The native RVO2 dependency is built separately from its disposable copy; it and the editable application are deliberately absent from the reusable lockfile's machine-specific direct paths. `legacy/pyproject.toml` freezes this baseline project definition. This environment is reference-only, pending later modern acceptance/removal.

Seed 0 printed result: `ReachGoal`, simulation_time `14.0`, frames `56`, time_step `0.25`, policy `SARL`, query_env `true`. Terminal robot position is `(4.7931619758612705, 4.732563262494248)` with goal `(5, 5)`. Seeds 1 and 2 reached the goal at 12.25 and 12.5 seconds respectively. Paired reruns of all three seeds passed full trace and terminal-result comparison at `rtol=1e-6, atol=1e-7`; comparison excludes wall time and video bytes.

The H.264 baseline video contains 56 frames at 8 fps, lasting 7 seconds, so it plays at twice simulation speed. Original `env.states` and video frames are recorded before each integration step: the last video frame is at 13.75 simulation seconds, while `baseline.json` records the actual terminal state after integration at 14 seconds. Controller inspected the start/middle/end contact sheet and confirmed ego and four moving neighbours are visible and goal approach agrees with ReachGoal; full playback inspection was not claimed.

`reference/` includes all source hashes before and after execution, input config/checkpoint hashes, full versioned episode traces/results, six inference fixtures with every candidate value forward call, tolerances and output hashes. All 78 supplied files have identical before/after SHA-256 hashes. Generated videos and runtime logs are ignored under `results/`.

Tests pass 4/4. The legacy Matplotlib/pyparsing combination emits 12 dependency deprecation warnings; these are recorded rather than hidden.
