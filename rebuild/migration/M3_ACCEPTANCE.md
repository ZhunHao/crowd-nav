# M3 CPU acceptance

Original immutable `reference/inference.json` remains unchanged. Modern capture
matches all six rows: action atol=1e-7/rtol=0, rotated input atol=1e-7/rtol=1e-6,
network values atol=1e-6/rtol=1e-5. Expanded evidence is separate under migration.

Expanded frozen inputs contain three recorded episode starts, the three lowest
legacy top-two candidate margins found by scanning all recorded nonterminal
frames, and 1/2/8/24-neighbour synthetic inputs. Every sampled network call and
all candidate scores are retained in expanded-legacy.json and expanded-modern.json.
Smallest selected legacy margin: 3.076922807726534e-5; largest candidate score
drift: 1.1611029462899047e-7. All ten actions match at the strict action tolerance.
There is no near-tie waiver. See m3-summary.json for each margin and difference.

Three complete bounded controller traces use those same episode starts and
constant-velocity neighbours (frozen independently of controller). The 64-step
clock spans 16 seconds, continuous state, SI units, no interventions, perfect
observations. Collision scoring uses minimum swept-disc clearance across each
step. Legacy and modern full traces are exactly equal. All three are TIMEOUTS;
this verifies migration consistency, not successful navigation. This minimal
harness does not claim to reproduce CrowdSim's reactive humans or original
trajectory semantics and is not the later full simulator.

Finite checkpoint tensors, strict loading, finite network output, explicit
empty-neighbour goal steering and native ORCA avoidance execute in modern tests.
The empty-neighbour adapter adds new behavior outside the copied compatibility
closure; there is no legacy empty-input parity claim.

RVO2 compiled and ran on local macOS arm64/CPython 3.14.7. See rvo2/README.md for
patch/license/compiler evidence and separate optional wheel installation. No
Linux native or accelerator acceptance is claimed. Source hashes before/after
match, excluding pre-existing bytecode as in the baseline manifest. All legacy
commands used PYTHONDONTWRITEBYTECODE=1. Original reference files were not edited.

Re-run from rebuild (modern environment already synchronized and native wheel installed):

```
.venv-modern/bin/python tools/parity_fixture.py --modern --output migration/inference.json
.venv-modern/bin/python tools/expanded_parity.py --modern --output migration/expanded-modern.json
.venv-modern/bin/python tools/replay_migration.py --modern --output migration/replay-modern.json
.venv-modern/bin/python tools/migration_summary.py
.venv-modern/bin/python -m pytest -v
```

Legacy capture is complete. The M4 controller owns retirement and removal of
legacy capture entry points; M3 has not deleted or retired that environment.
