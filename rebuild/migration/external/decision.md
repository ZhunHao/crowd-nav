# D4 decision: NTNU and CommonOcean external maritime tooling

Date: 2026-09-27. Scope: task D4 (verify maritime adapter conventions and
installation). Full COLREGs legality is outside the meaning of either
feasibility checker discussed below; neither is claimed to cover it.

## NTNU colav-simulator

**Not reinstalled for this task.** A bounded compatibility investigation was
already run by the user and its evidence lives, untracked, in the main
checkout at
`rebuild/migration/ntnu-investigation/`
(see `README.md` there; not copied or committed here per controller ruling).

Verdict recorded there: the unmodified NTNU stack is not ready to adopt in
ShipNav's modern environment (Python 3.14.7). Reproduced blockers: both
`colav-simulator` and its pinned `seacharts` dependency declare
`>=3.10,<3.12`; a Hatchling wheel-build conflict; native GDAL/Fiona have no
CPython 3.14 wheels and the host lacks `gdal-config`; and chart assets are
Git-LFS pointers only (no LFS filter). The models are also import-coupled to
the geospatial stack (`models -> config_parsing -> file_utils ->
map_functions -> osgeo/SeaCharts`), so even a map-free model import currently
fails.

**Decision: NTNU remains an external-evaluator candidate, not adopted.** No
`adapters/ntnu.py` is added (per controller ruling, since the platform is
unsupported on the app's current Python 3.14 stack). The synthetic marine
backend (`shipnav/dynamics.py`, discrete heading/speed model) is retained as
the only marine backend for now. This should be revisited only after a
dedicated Python-version/native-toolchain port of NTNU's stack, as outlined
in that investigation's "Recommended next experiment".

## CommonOcean

Disposable environment: `migration/external/commonocean/.venv-commonocean`
(git-ignored), CPython 3.10.20 via `uv venv --python 3.10.20` (both
`commonocean-io` and `commonocean-drivability-checker` declare
`Requires-Python`/classifiers only up to 3.10; the app's Python 3.14.7 is
incompatible with the upstream package metadata, so an older interpreter was
used for this env only, per the controller ruling). Evidence files:
`install.log`, `install2.log`, `install-io.log`, `freeze.txt` (all in this
directory).

### What was read before installing

- https://commonocean.cps.cit.tum.de/ (tool overview) and the linked
  Drivability Checker / Input-Output pages.
- https://pypi.org/project/commonocean-io/ (2025.1; deps incl.
  `commonocean-vessel-models==1.0.0`, `commonroad-io==2023.1`; **license
  GPLv3**; classifiers Python 3.8-3.10).
- https://pypi.org/project/commonocean-drivability-checker/ (2025.1; depends
  on `commonroad-drivability-checker`; **license BSD**; classifiers Python
  3.8-3.10; install command documented as requiring Python 3.8/3.9/3.10).
- https://pypi.org/project/commonocean-vessel-models/ (1.0.0; **license
  BSD**).
- The official CommonOcean interface tutorial
  (https://commonocean-documentation.readthedocs.io/en/latest/commonocean-dc/doc/docs/source/02_commonocean_interface.html),
  confirming the `CommonOceanFileReader` / `create_collision_checker` /
  `create_collision_object` / `GeneralState` / `TrajectoryPrediction` API
  used by the task brief's `check_collision` starting point.

### Install attempt and native build failure

`uv pip install --python .venv-commonocean/bin/python commonocean-io==2025.1
commonocean-drivability-checker==2025.1` (see `install.log`):

- `commonocean-io==2025.1` and its pure-Python dependency chain
  (`commonocean-vessel-models==1.0.0`, `commonroad-io==2023.1`,
  `commonroad-vehicle-models==3.0.2`, `matplotlib`, `numpy`, `shapely`,
  `lxml`, `iso3166`, ...) built and installed cleanly — this package needs no
  native compilation.
- `commonocean-drivability-checker==2025.1` pulls in
  `commonroad-drivability-checker==2023.1`, which builds a native C++
  extension (`pycrcc`, bundling the third-party `ccd`/`libccd` and `s11n`
  sources) via CMake. The build failed on this host (macOS arm64, CMake
  4.4.3): `commonroad-drivability-checker`'s bundled `ccd` CMakeLists.txt
  calls `cmake_minimum_required` below 3.5, which CMake >=4 refuses to
  configure at all (`install.log`).
  - One bounded, non-invasive follow-up attempt was made: setting the
    documented CMake environment variable `CMAKE_POLICY_VERSION_MINIMUM=3.5`
    (a CMake-provided compatibility escape hatch for exactly this class of
    ecosystem breakage; no vendor source was patched). This got past the
    hard error and progressed further into the build (`install2.log`), but
    then failed on a genuine missing **native library**: CMake could not
    find Boost (`Could NOT find Boost (missing: Boost_INCLUDE_DIR)`,
    `cpp/collision/CMakeLists.txt:3`). Boost is not installed on this host
    (`brew list boost` reports not found) and is not obtainable from PyPI or
    from either project's own repository, so per the effort-bound in the
    controller ruling this was recorded as the stopping point rather than
    installing a system package to work around it.

**Result: `commonocean-drivability-checker` (and therefore `commonocean_dc`,
the collision-oracle package) could not be installed in this environment.**
This mirrors the NTNU investigation's pattern of a missing native geospatial
dependency (GDAL/Fiona there, Boost/CMake here) — a genuine compatibility
blocker, not a fixable oversight in this task's scope.

### What was verified with the part that did install

`commonocean-io` (pure Python, installed and importable) was used to load
two real upstream scenarios from the official scenario repository
(https://gitlab.lrz.de/tum-cps/commonocean-scenarios.git, revision
`2ab3a4b3964c9df70bc4ce93f7b6d59c80f9de39`, BSD-3-Clause license,
Copyright 2019 TUM Professorship of Cyber-Physical Systems — see that repo's
`LICENSE.txt`, cloned read-only under the git-ignored `upstream/` directory):

- `scenarios/HandcraftedTwoVesselEncounters_01_24/ZAM_AAA-1_20240121_T-786.xml`
  (one dynamic obstacle, 170-state trajectory, dt=10s, no waters/static
  obstacles) — exported into the canonical schema at
  `scenarios/external/commonocean_ZAM_AAA-1_20240121_T-786.json` with full
  provenance (repository, revision, license) recorded under
  `external_metadata`.
- `scenarios/NewYorkManhattan/USA_NYM-1_20190613_T-1.xml` (waters, 6 static
  obstacles with exact polygons, 24 dynamic obstacles, dt=1s) — used only to
  read the installed `Waters`/`StaticObstacle`/`Location` API surface
  (not committed: 1.8 MB, kept in the git-ignored upstream checkout).

Findings recorded honestly in `adapters/commonocean.py` /
`import_scenario()`:

- CommonOcean scenario coordinates are already planar Cartesian (metres).
  `adapters.coordinates.north_east_to_xy` (the NTNU-style North-East
  transform) is deliberately **not** applied to CommonOcean data, per the
  task brief.
- Neither sample scenario's `location.geo_transformation` declares an
  EPSG/CRS; only an approximate WGS84 reference point (`gpsLatitude`,
  `gpsLongitude`) is present. This is recorded as unverified rather than
  guessed. `import_scenario` raises `NotImplementedError` if it ever
  encounters a scenario that *does* declare a `geo_transformation`, since
  that case has not been verified.
- Per-actor control limits are **not** embedded in the CommonOcean scenario
  XML schema (no obstacle-to-vessel-model mapping in either sample file).
  The separately-installed `commonocean-vessel-models==1.0.0` package
  exposes generic (non-scenario-specific) parameter sets
  (`vesselmodels.parameters_vessel_1`: `v_max=16.8` m/s, `a_max=0.24` m/s²,
  `w_max=0.03` rad/s, verified by import); these are recorded for reference
  only and not attributed to any actor.
- All obstacle predictions observed in both sample files are
  `TrajectoryPrediction` (fixed, time-indexed). `import_scenario` raises
  `NotImplementedError` for any `SetBasedPrediction` (time-varying occupancy
  set) rather than silently dropping or approximating it, since none of the
  sampled scenarios exercise that path and it has not been verified.
- Land/static-obstacle polygons are kept exact under
  `external_metadata.original_static_obstacles[].exact_polygon`; a
  conservative axis-aligned bounding-box approximation is additionally
  provided and explicitly labelled `conservative_rectangle_approximation`,
  never presented as the exact shape.

### `check_collision` / `check_trajectory`

`adapters/commonocean.py` implements `check_collision` exactly per the task
brief's verified starting point (matches the official tutorial's
`CommonOceanFileReader` / `GeneralState` / `TrajectoryPrediction` /
`create_collision_checker` / `create_collision_object` API) and
`check_trajectory`, which names the collision, water-boundary, and
model-feasibility components separately. Because `commonocean_dc` is not
installed in any environment available here, `check_collision` raises
`ImportError` when called (verified by
`tests/test_commonocean.py::test_check_collision_requires_commonocean_dc`,
an `xfail(raises=ImportError, strict=True)` test — it would fail loudly, not
silently pass, if `commonocean_dc` ever became importable without this code
being revisited). `check_trajectory` catches that `ImportError` and reports
`collision: {checked: False, reason: "commonocean_dc unavailable: ..."}`
rather than pretending the check passed. Water-boundary and
model-feasibility are always reported `checked: False` — they are not
implemented at all in this adapter, and are never represented as passed.

No known-colliding/known-clear/mismatched-dt tests against the real upstream
collision oracle could be run (that requires `commonocean_dc`, which is not
installed anywhere in this investigation). This is recorded as a genuine gap,
not worked around.

### Decision

**CommonOcean's collision oracle (`commonocean_dc`) is not usable in this
environment without a native Boost/CMake toolchain fix that is out of scope
for this task's effort bound.** `commonocean-io` import/export is usable and
is implemented, tested against a real upstream sample, and exported with
full provenance. `adapters/commonocean.py` is committed with both parts: the
verified `import_scenario`, and `check_collision`/`check_trajectory` written
honestly against the documented API but currently non-functional in any
available environment (tests marked `external`, skipped when `commonocean`
is not importable, and one `xfail(strict=True)` guarding the
`commonocean_dc` ImportError so a future successful install is caught rather
than silently accepted as still-unchecked).

## Addendum 2026-09-28 — retry with native dependencies installed

At the user's request the missing native libraries were installed with Homebrew and the build was retried in the same disposable environment (CPython 3.10.20, `CMAKE_POLICY_VERSION_MINIMUM=3.5`):

| Attempt | Host change | Result | Log |
|---|---|---|---|
| 3 | `brew install boost` (1.92.0) | Boost found; configure fails: Eigen3 not found | `commonocean/install3.log` |
| 4 | `brew install eigen` (5.0.1) | Eigen 5 rejected: the build requests `Eigen3 3.0.5` via CONFIG and Eigen 5's version file refuses major 3 | `commonocean/install4.log` |
| 5 | `brew install eigen@3` (3.4.1, keg-only; `CMAKE_PREFIX_PATH`/`Eigen3_DIR` pointed at it) | Configure passes; compile fails: `clang++: error: unsupported option '-msse'/'-msse2'/'-msse3'/'-mssse3'` for arm64 | `commonocean/install5.log` |

PyPI publishes no macOS arm64 wheel for any `commonroad-drivability-checker` release (only x86_64 wheels up to 2022.2.1), and `commonocean-drivability-checker==2025.1` hard-pins `commonroad-drivability-checker==2023.1`, `numpy~=1.24.0` and `scipy<=1.7.2`. Going further requires patching vendored C++ build flags, which this spike does not do.

**Decision (unchanged, now with stronger evidence): do not adopt CommonOcean as a dependency or backend.** Keep `commonocean-io` (pure Python, works) as an optional external scenario importer in this disposable environment. If an independent collision oracle is still wanted, run the upstream checker on Linux x86_64 (e.g. an EC2 instance or an x86_64 container) in its own legacy environment; its discrete-time check is weaker than ShipNav's continuous swept checks and serves only as a cross-check. Homebrew packages installed on the host for this attempt: boost 1.92.0, eigen 5.0.1, eigen@3 3.4.1 (removable with `brew uninstall boost eigen eigen@3`).
