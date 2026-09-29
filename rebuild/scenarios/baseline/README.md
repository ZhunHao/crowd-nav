# Frozen baseline extension

`splits.json` extends the unchanged historical
`../splits.json`: 110 test, 7 development and 7 calibration scenarios. Original
canonical cases are previously inspected regression fixtures, held out from
training/tuning, not claimed unseen by developers. Every stored scenario is exact
JSON compressed losslessly with deterministic gzip; the manifest records both
stored and decompressed SHA256, physical-content SHA256 and scenario identity.

Regenerate with `.venv-modern/bin/python tools/baseline_freeze.py`. The extension
uses all 11 named canonical families, each original plus eight endpoint/speed
perturbations validated for nominal encounters (99 cases). Eleven 350–550 m
Southern Islands corridor voyages supply geographic holdout, with 1–3 ships.
Four Ubin voyages belong only to development/calibration. Fixed real-map routes
are built on `to_model(..., PROFILES['harbour_craft'])`; `make_scenario` receives
the route as `corridor`. Original map provenance/licensing stays embedded.

The fixed route is A* plus smoothing at the scenario's recorded planner scale.
It is computed before controller outcomes. Deliberately unreachable cases have a
null reference, with no meaningful detour/tracking score. Eight perturbations are
nearby cases, not broad independent operational coverage: bootstrap precision is
conditional on this scenario mixture and does not establish geographic-population
or encounter-family independence.

See `../../evidence/baseline/benchmark_protocol.json` for preregistration and
`../../evidence/baseline/README.md` for results and archive audit commands.
