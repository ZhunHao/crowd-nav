# Checkpoint-compatible inference port

This closure is copied from the supplied, modified `CrowdNav-20250813-DIP`
source tree. Original bytes and checkpoint hashes are recorded in
`rebuild/reference/manifest.json`. `rebuild/tools/port_inference.py` checks the
seven source file hashes against that manifest before writing the port.

Upstream project: https://github.com/vita-epfl/CrowdNav (VITA lab at EPFL).
The supplied tree omits CrowdNav's root license. The attached MIT license was
retrieved on 2026-09-27 from
https://raw.githubusercontent.com/vita-epfl/CrowdNav/master/LICENSE and retains
its Copyright (c) 2018 VITA lab at EPFL attribution. This identifies upstream
licensing; it does not assert authorship or additional permissions for the
supplied local modifications. Python-RVO2 is a separate optional dependency;
its bundled license is not used as the CrowdNav license.

## Source differences

- Text reads normalize CRLF to LF and remove trailing blank lines.
- Internal `from crowd_nav.` and `from crowd_sim.` imports gain the
  `shipnav.compat.` prefix.
- Package initializers are empty to avoid Gym registration and eager
  simulator/native dependency imports.
- ORCA's `import rvo2` moves inside `predict`; executing ORCA still requires
  the optional native dependency, with the original native calculations.
- No network layers, attention masking, normalization, sampling, or numerical
  formulas are changed. No NumPy 2 or Torch API correction was needed to load
  the supplied checkpoint on the locked modern stack.

`shipnav.model.load_policy` is extracted from the baseline loader, imports
only this closure, uses CPU `weights_only=True`, explicitly strict state-dict
loading, eval mode and test phase, and rejects non-finite checkpoint tensors.
Finite output and empty-neighbour adapter acceptance are tracked in M3.

Run the copy tool from any working directory. Its default source and target
are derived from the script path. Re-running reproduces this closure and the
license; keep any future approved source corrections in the copy tool too.
