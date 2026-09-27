# Optional modern native ORCA

Built locally on macOS arm64 for CPython 3.14.7, from the bundled Python-RVO2
source in a disposable copy. The original tree is never built or modified.
Cython 3.3.0 and CMake 4.4.3 were current stable releases checked 2026-09-27:
https://pypi.org/project/Cython/ and https://pypi.org/project/cmake/.

`modern.patch` lifts the obsolete CMake minimum from 2.8 to 3.5 (CMake 4
removed compatibility below 3.5), and makes the original implicit 0.0.0
package version explicit. Native algorithm and Cython sources are unchanged.
Bundled Apache 2.0 LICENSE is retained. Compiler output is in build.log;
wheel SHA-256 is in wheels.json. The wheel is platform-specific, not a
portable Linux validation. Linux must run this build and tests on its target.

Reproduce from `rebuild/` after syncing the core environment:

```
VIRTUAL_ENV="$PWD/.venv-modern" uv sync --locked --all-extras --python 3.14.7 --active
.venv-modern/bin/python tools/build_rvo2.py
.venv-modern/bin/python tools/install_rvo2.py
.venv-modern/bin/python -m pytest tests/test_migration.py::test_native_orca_runs -v
```

For the sync command set `VIRTUAL_ENV` to the absolute `.venv-modern` path,
or use the project's existing modern environment recipe. Build dependencies
are pinned with hashes in build-requirements.txt (including transitive packaging) and installed only into a disposable build
environment; only the resulting wheel enters modern. Optional native support
is intentionally absent from the core dependency graph and uv.lock. A fresh
core sync may remove this wheel: install it as the explicit subsequent step.
The build requires a local C/C++ compiler and uv; it changes no global tools.
