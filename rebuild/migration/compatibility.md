# Modern environment compatibility

Inventory refreshed from PyPI JSON on 2026-09-27; all selected project versions match current stable releases and non-yanked artifacts. Official https://www.python.org/downloads/ confirms CPython 3.14.7 as newest stable; 3.15 is prerelease. Standard GIL runtime is required (`Py_GIL_DISABLED=0`). Exact wheel names, hashes and Requires-Python are saved in releases.json and wheel-inspection.md.

`uv python install 3.14.7` with host uv 0.11.25 failed with `error: No download found for request: cpython-3.14.7-macos-aarch64-none`. An existing installed CPython 3.14.7 was used, verified by runtime sys.version (rather than its directory name). Interpreter provenance is in macos-arm64/smoke.json. No Python downgrade was made.

Universal lock resolution and macOS ARM64 sync pass. An explicit Linux x86_64 binary-only resolution at manylinux_2_28 failed: PySide6 6.11.2 has no matching wheel; available x86_64 Linux wheels require manylinux_2_34. Explicit manylinux_2_34 resolution passes and is saved in linux-x86_64-requirements.txt and linux-resolution-output.txt. Linux deployment must use glibc >=2.34. Linux CPU execution and CUDA execution remain pending a separately authorized host; no Linux or GPU runtime acceptance is claimed.

macOS CPU tensor forward/backward, native Cocoa Qt window, PNG and FFmpeg MP4 exports are exercised by tools/smoke_modern.py. MPS availability is recorded only; MPS parity is not yet validated. RVO2, CasADi, SciPy/Cython build experiments and geospatial packages remain outside core resolution. The legacy interpreter remains until M4 acceptance.

Future commands must use `UV_PROJECT_ENVIRONMENT=.venv-modern uv run --locked ...` or `.venv-modern/bin/python`. Any upgrade requires parity again. CUDA must use its own verified project/index/lock; PyPI is unchanged globally.
