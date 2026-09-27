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
