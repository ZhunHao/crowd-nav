# 01b — Newest stable stack migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the verified inference baseline to the newest stable application stack and retire legacy once the modern stack is verified.

**Architecture:** Use an isolated legacy interpreter temporarily for comparison, then remove it at acceptance. Copy only the inference dependency closure into a new namespace, prove CPU parity, then use modern locks for every later phase.

**Tech Stack:** Latest stable CPython, uv, PyTorch, NumPy, Gymnasium, PySide6 and pytest; versions discovered again at execution

**Spec:** [Revised design](../specs/2026-09-27-solo-rebuild-design.md)

## Global Constraints

- Never use `codex` in branch names or worktree directory names.
- Preserve `CrowdNav-20250813-DIP/` and the supplied briefing unchanged.
- Implement only inside `rebuild/`; keep planning documents inside `docs/superpowers/` and Git-ignored.
- Use `.venv-legacy` only for baseline verification and migration comparison; remove the legacy environment, build dependencies and runtime support after modern acceptance passes.
- Retain baseline results, hashes and frozen comparison fixtures as verification evidence; do not retain a legacy fallback or maintain a second runtime.
- After that gate, attempt the newest stable Python and dependency releases available at execution time; prereleases require an explicit separate experiment.
- All subsequent phases use `.venv-modern` and its committed lockfile; Linux EC2 training uses a separately verified CUDA environment with the same application version and Python minor.
- Do not silently downgrade the modern stack. Record compatibility blockers, isolate optional native dependencies, and re-run parity checks after any port.
- Run migration parity on CPU first; validate MPS and CUDA separately before using their results.
- Use metres, seconds, radians, and Cartesian East–North `(x, y)` coordinates in the rebuilt simulation.
- Keep one episode clock and continuous vessel state across intermediate goals.
- Freeze scenarios independently of planners; pair initial conditions and exogenous randomness across comparisons.
- Keep perceived observations separate from scoring truth; report collisions, failures, safety interventions and missed deadlines honestly.
- Use AWS CLI for EC2 retraining operations; choose account, region, resource sizes and an explicit spending limit when executing that phase.
- Do not implement literature work, academic report writing, or competition presentation preparation in these plans.

---

All implementation paths below are relative to `rebuild/`. Run commands there. These are instructions and reference snippets, not installed software or benchmark results.

## File map and release evidence

Create `tools/release_inventory.py`, `tools/port_inference.py`, `tools/parity_fixture.py`, `tests/test_migration.py`, `migration/compatibility.md`; modify `pyproject.toml`, `.python-version`, `uv.lock`; keep `legacy/pyproject.toml` and `.venv-legacy` only during comparison; keep `reference/` evidence after retirement.

Direct PyPI JSON queried 2026-09-27 returned these candidate releases. They are candidates, not a tested compatible set: Python **3.14.7** (official Python downloads), torch **2.14.0**, NumPy **2.5.3**, Gymnasium **1.3.0**, SB3 **2.9.0**, PySide6 **6.11.2**, Matplotlib **3.11.2**, SciPy **1.18.1**, CasADi **3.8.1**, Cython **3.3.0**, pytest **9.1.1**, pytest-qt **4.5.0**, uv **0.12.19**. A PyTorch announcement search also surfaced 2.14.1; the live PyPI metadata returned 2.14.0, so do not pin an unverified wheel on that announcement alone. Python 3.15 was still a prerelease at this review. Refresh releases when executing, because the user asked for the newest available then.

Sources: [Python downloads](https://www.python.org/downloads/), package-owner metadata at [PyPI JSON](https://pypi.org/pypi/torch/json) (substitute the package name), [uv platform resolution](https://docs.astral.sh/uv/concepts/resolution/), [PyTorch installation selector](https://docs.pytorch.org/get-started/locally/). Use stable release metadata, not a repository's development-version documentation.

### M1 — Resolve and smoke-test modern environments

- [ ] Save the release inventory code below. It captures installable-release metadata and wheel names so missing Python/platform wheels are visible. Recheck the Python downloads page; choose standard GIL-enabled CPython, leaving free-threaded builds for a separately labelled experiment.

File: `rebuild/tools/release_inventory.py`

```python
import json
from pathlib import Path
from urllib.request import urlopen

PACKAGES = ['torch', 'numpy', 'gymnasium', 'stable-baselines3', 'PySide6',
            'matplotlib', 'scipy', 'casadi', 'Cython', 'pytest', 'pytest-qt', 'uv']

def inventory(fetch):
    result = {}
    for name in PACKAGES:
        data = fetch(name)
        result[name] = {'version': data['info']['version'],
                        'requires_python': data['info']['requires_python'],
                        'files': [{'name': f['filename'], 'sha256': f['digests']['sha256'],
                                   'yanked': f['yanked']} for f in data['urls']]}
    return result

if __name__ == '__main__':
    def fetch(name):
        with urlopen(f'https://pypi.org/pypi/{name}/json', timeout=30) as response:
            return json.load(response)
    Path('migration').mkdir(exist_ok=True)
    Path('migration/releases.json').write_text(json.dumps(inventory(fetch), indent=2))
```

- [ ] Run `python tools/release_inventory.py`; inspect release status, `Requires-Python`, macOS ARM64 wheels and Linux x86_64 wheels. Exclude yanked/prerelease candidates. Capture current uv/Python/FFmpeg versions. Keep optional RVO2, CasADi and geospatial packages outside core dependency resolution until their spikes pass.
- [ ] Copy the existing baseline project metadata into `legacy/pyproject.toml`, then replace the root metadata with this starting point; update exact candidate versions if the fresh inventory differs. Keep package discovery at `src` and existing pytest markers. Add `legacy` and `external` pytest markers and configure default test collection to exclude `tests/legacy`.

```toml
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "shipnav-rebuild"
version = "0.2.0"
requires-python = ">=3.14,<3.15"
dependencies = ["numpy==2.5.3", "torch==2.14.0", "gymnasium==1.3.0"]

[project.optional-dependencies]
gui = ["PySide6==6.11.2", "matplotlib==3.11.2"]
train = ["stable-baselines3==2.9.0"]

[dependency-groups]
dev = ["pytest==9.1.1", "pytest-qt==4.5.0"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
norecursedirs = ["tests/legacy"]
markers = ["integration: checkpoint/native dependency", "external: upstream maritime tools", "video: FFmpeg"]

[tool.uv]
environments = ["sys_platform == 'darwin' and platform_machine == 'arm64'", "sys_platform == 'linux' and platform_machine == 'x86_64'"]
required-environments = ["sys_platform == 'darwin' and platform_machine == 'arm64'", "sys_platform == 'linux' and platform_machine == 'x86_64'"]
```

- [ ] Move `tests/test_baseline.py` to `tests/legacy/test_baseline.py`. Keep its model path calculation correct after the extra directory (`parents[3]` reaches workspace root). Run it only by explicit path in the legacy interpreter. Do not import `shipnav.baseline` from modern modules.
- [ ] Create and resolve the modern project from `rebuild/`:

```bash
uv python install 3.14.7
uv python pin 3.14.7
uv venv --python 3.14.7 .venv-modern
UV_PROJECT_ENVIRONMENT=.venv-modern uv lock --upgrade
UV_PROJECT_ENVIRONMENT=.venv-modern uv sync --locked --extra gui --extra train
.venv-modern/bin/python -c 'import torch, numpy, gymnasium, PySide6; print(torch.__version__, numpy.__version__, gymnasium.__version__, PySide6.__version__)'
```

- [ ] Run an actual tensor forward/backward, Qt native window, Matplotlib PNG and FFmpeg export; a resolver success is not runtime evidence. On Linux EC2 run CPU and CUDA tensor tests separately and save `torch.version.cuda`, driver/GPU model and `torch.cuda.is_available()`.
- [ ] If a newest release cannot install, first patch the copied integration or isolate the optional native dependency; document exact resolver/build errors in `migration/compatibility.md`. Do not silently pin older Python. Mark that platform blocked until its chosen solution passes. If the newest stable set is fundamentally incompatible, present the concrete alternatives and affected phases at implementation time.
- [ ] Store exact `uv.lock`, `.python-version`, platform smoke output and native toolchain versions. Future phases use `UV_PROJECT_ENVIRONMENT=.venv-modern uv run --locked ...` or `.venv-modern/bin/python`; upgrades require a new parity run. Linux CUDA may need a separate project at `environments/ec2/` with the same pure-Python pins and local source dependency, explicit verified PyTorch CUDA index, and its own lock. Do not globally substitute a CUDA-only index for PyPI.

### M2 — Port the checkpoint-compatible inference closure

- [ ] Create and run this source-copy tool after recording the baseline manifest. Copy upstream license/attribution alongside the port. It deliberately creates empty package initializers to avoid Gym registration and eager RVO2 imports.

File: `rebuild/tools/port_inference.py`

```python
from pathlib import Path

FILES = ['crowd_nav/policy/sarl.py', 'crowd_nav/policy/cadrl.py',
         'crowd_nav/policy/multi_human_rl.py', 'crowd_sim/envs/policy/policy.py',
         'crowd_sim/envs/policy/orca.py', 'crowd_sim/envs/utils/state.py',
         'crowd_sim/envs/utils/action.py']

def port(source: Path, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        out = target/name
        out.parent.mkdir(parents=True, exist_ok=True)
        code = (source/name).read_text()
        code = code.replace('from crowd_nav.', 'from shipnav.compat.crowd_nav.')
        code = code.replace('from crowd_sim.', 'from shipnav.compat.crowd_sim.')
        out.write_text(code)
        parent = out.parent
        while parent != target.parent:
            (parent/'__init__.py').touch()
            parent = parent.parent

if __name__ == '__main__':
    port(Path('../CrowdNav-20250813-DIP'), Path('src/shipnav/compat'))
```

- [ ] Extract only `load_policy` from `baseline.py` into `src/shipnav/model.py`. Retain its config/weights validation, CPU load with `weights_only=True`, state-dict strict loading, evaluation mode and test phase. Change its SARL import to `shipnav.compat.crowd_nav.policy.sarl`; import `Path`, `configparser` and `torch`. Do not import baseline/CrowdSim. Update plan 03's `Learned` and `Reciprocal` imports to this namespace (already reflected in the revised snippets).
- [ ] Do not change network layers, attention masking, feature normalization or action sampling while proving parity. NumPy 2 removals and modern torch API incompatibilities must be minimal, individually tested port fixes with source-diff notes. Optimize only after parity passes.

### M3 — Replay identical inference inputs on both stacks

Create the fixture script below. Legacy mode runs before migration; modern mode reads the same configs and uses the namespaced classes. Hook outputs cover every sampled candidate value evaluation, not only the final action.

File: `rebuild/tools/parity_fixture.py`

```python
from pathlib import Path
import argparse, importlib, json
import torch


def capture(modern, model_dir):
    if modern:
        from shipnav.model import load_policy
        prefix = 'shipnav.compat.'
    else:
        from shipnav.baseline import load_policy
        prefix = ''
    states = importlib.import_module(prefix+'crowd_sim.envs.utils.state')
    policy = load_policy(model_dir)
    policy.query_env = False
    policy.time_step = .25
    rows = []
    for n in (1, 4, 12):
        for vx in (0., .7):
            calls = []
            def hook(module, args, result):
                calls.append({'input': args[0].detach().cpu().tolist(),
                              'value': result.detach().cpu().tolist()})
            handle = policy.model.register_forward_hook(hook)
            own = states.FullState(2., 2., vx, 0., .5, 12., 9., 1., 0.)
            others = [states.ObservableState(4.+i, 4.+i%3, -.2, .1, .6) for i in range(n)]
            with torch.inference_mode():
                action = policy.predict(states.JointState(own, others))
            handle.remove()
            rows.append({'n': n, 'vx': vx, 'action': list(action), 'calls': calls})
    return rows

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modern', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = capture(args.modern, Path('../CrowdNav-20250813-DIP/crowd_nav/data/output_trained'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, allow_nan=False))
```

File: `rebuild/tests/test_migration.py`

```python
from pathlib import Path
import json
import numpy as np

def test_cpu_inference_parity():
    old = json.loads(Path('reference/inference.json').read_text())
    new = json.loads(Path('migration/inference.json').read_text())
    assert len(old) == len(new) == 6
    for a, b in zip(old, new):
        assert (a['n'], a['vx']) == (b['n'], b['vx'])
        np.testing.assert_allclose(a['action'], b['action'], rtol=0, atol=1e-7)
        assert len(a['calls']) == len(b['calls']) > 0
        for x, y in zip(a['calls'], b['calls']):
            np.testing.assert_allclose(x['input'], y['input'], rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(x['value'], y['value'], rtol=1e-5, atol=1e-6)
```

- [ ] Run `.venv-legacy/bin/python tools/parity_fixture.py --output reference/inference.json` before porting, then `.venv-modern/bin/python tools/parity_fixture.py --modern --output migration/inference.json` and `.venv-modern/bin/python -m pytest tests/test_migration.py -v`. Expand inputs with actual recorded baseline states, near ties and neighbour counts before accepting the port. If action changes occur near a tie, record candidate value margins; do not silently approve them based on a loose tensor tolerance.
- [ ] Verify `import shipnav.model` does not import Gym or RVO2 by checking `sys.modules`. Add nonfinite-weight/output and no-neighbour adapter tests. Compare complete modern closed-loop traces on frozen scenarios separately; do not conflate a new simulator with exact original trajectory reproduction.
- [ ] Spike bundled RVO2 in a disposable directory with current Cython/CMake. If it needs a patch, keep the patch and compiler log under `migration/rvo2/` and install only the built wheel in modern. If blocked, mark ORCA unavailable with an explicit error; the core/GUI can pass while the ORCA benchmark gate stays open. A legacy subprocess comparator is not an accepted solution. Resolve required ORCA support in the modern environment before retiring legacy; keep the affected acceptance gate open while it is blocked.
- [ ] Commit the modern port, license, fixtures, release inventory and locks once gates pass, then complete M4. Temporary legacy setup metadata is removed at retirement. No subsequent phase installs, restores or runs the legacy stack.

### Exact modern model loader

Create this file in M2; it is the sole modern checkpoint loader. Baseline-only imports stay in `baseline.py`.

File: `rebuild/src/shipnav/model.py`

```python
from pathlib import Path
import configparser
import torch
from shipnav.compat.crowd_nav.policy.sarl import SARL

def load_policy(model_dir: Path) -> SARL:
    config = configparser.RawConfigParser()
    config_path = model_dir / 'policy.config'
    weights = model_dir / 'rl_model.pth'
    for path in (config_path, weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    config.read(config_path)
    policy = SARL()
    policy.configure(config)
    if policy.kinematics != 'holonomic':
        raise ValueError('This rebuild requires a holonomic checkpoint')
    policy.set_device(torch.device('cpu'))
    policy.get_model().load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    policy.get_model().eval()
    policy.set_phase('test')
    return policy
```

### M4 — Retire legacy after modern acceptance

This is the required end of migration, not an optional cleanup. “Modern works” means the currently targeted local platform passes all checks below, not just package installation. Linux/CUDA acceptance belongs to its later deployment phase and must also use modern dependencies; do not keep legacy waiting for future cloud work.

**Files:** remove `rebuild/.venv-legacy/`, `rebuild/legacy/`, `rebuild/locks/legacy.txt`, `rebuild/src/shipnav/baseline.py`, `rebuild/tests/legacy/`, `rebuild/tools/reference_manifest.py`; simplify `rebuild/tools/parity_fixture.py`, `rebuild/pyproject.toml` and `rebuild/README.md`. Keep `rebuild/reference/`, the modern checkpoint port and its upstream license. The user-supplied original source/checkpoints remain unchanged as input assets, not an installed or supported legacy runtime.

- [ ] Pass the CPU tensor/value/action parity tests against frozen baseline fixtures, using identical inference settings. Verify modern checkpoint loading, empty-neighbour handling and a bounded modern closed-loop replay. Save outcome/trace differences with explanations; a changed simulator need not reproduce the historical trajectory exactly.
- [ ] Recreate the modern environment from its lockfile in a clean environment without any legacy `.pth` or `PYTHONPATH` entries. Pass required controller/native extension checks, a native Qt window, PNG and FFmpeg smoke exports, and modern import tests. Required controller failures block acceptance; a successful package import alone is insufficient.
- [ ] Save acceptance results, versions and lock/source/checkpoint hashes in `migration/acceptance.json`. Keep the already captured baseline manifest (including dependency versions), results and immutable input/output fixtures in `reference/`. These are evidence files, not a maintained legacy installation recipe.
- [ ] Change `tools/parity_fixture.py` to modern-only imports (`shipnav.model` and `shipnav.compat...`), remove its original-loader branch and `--modern` switch, and update its command in active operating instructions. It generates `migration/inference.json`; `reference/inference.json` remains the immutable comparison input. Historical commands in phases 01/M3 describe work completed before retirement and are not ongoing setup instructions.
- [ ] Stop processes using the temporary legacy environment and leave its activated shell. Remove the specific legacy paths listed above, including its copied native build tree. Remove the legacy pytest exclusion/markers, legacy environment setup commands, CI jobs and fallback paths. Do not uninstall a shared system Python or shared build tools used by other projects.
- [ ] Run the modern parity suite and smoke checks again after removal. Confirm modern execution does not import `shipnav.baseline`, original `crowd_nav`/`crowd_sim` packages or legacy Gym; migrated `shipnav.compat` modules remain modern code preserving checkpoint compatibility. Assert the retired environment/setup/runtime paths no longer exist.
- [ ] Update README and the migration record to show one supported modern application stack plus its platform-specific locks. Commit the retirement diff with the verification evidence. Subsequent debugging uses modern code and recorded fixtures; no legacy fallback is retained.
