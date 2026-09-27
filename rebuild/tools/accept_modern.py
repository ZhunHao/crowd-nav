"""Record the completed local acceptance gates and modern runtime provenance."""
from hashlib import sha256
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
RETIRED = ('.venv-legacy', 'legacy', 'locks/legacy.txt', 'src/shipnav/baseline.py',
           'tests/legacy', 'tools/reference_manifest.py', 'tools/capture_reference.py')


def hashes(root):
    return {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()
            and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def record():
    from shipnav.model import load_policy
    from shipnav.compat.crowd_sim.envs.policy.orca import ORCA
    from shipnav.replay import replay
    absent = {p: not (ROOT / p).exists() for p in RETIRED}
    imports = {p: importlib.util.find_spec(p) is None
               for p in ('shipnav.baseline', 'crowd_nav', 'crowd_sim', 'gym')}
    assert all(absent.values()) and all(imports.values())
    assert not os.environ.get('PYTHONPATH')
    assert not any('CrowdNav-20250813-DIP' in p or '.venv-legacy' in p for p in sys.path)
    source = hashes(ROOT.parent / 'CrowdNav-20250813-DIP')
    assert source == json.loads((ROOT / 'migration/source-before-m3.json').read_text())
    names = ('numpy', 'torch', 'gymnasium', 'PySide6', 'matplotlib',
             'stable-baselines3', 'pytest', 'pyrvo2')
    evidence = ('m4-clean-sync.txt', 'm4-native-install.txt',
                'm4-pre-retirement-tests.txt', 'm4-pre-retirement-parity.txt',
                'm4-pre-retirement-smoke.txt', 'm4-post-retirement-tests.txt',
                'm4-post-retirement-smoke.txt', 'm4-retirement-red.txt',
                'm4-retirement-green.txt', 'm4-summary-output.txt')
    for name in ('m4-pre-retirement-tests.txt', 'm4-pre-retirement-parity.txt',
                 'm4-post-retirement-tests.txt', 'm4-retirement-green.txt'):
        output = (ROOT / 'migration' / name).read_text()
        assert 'passed' in output and 'failed' not in output and 'ERROR' not in output, name
    data = {'accepted_platform': 'macOS arm64 CPU', 'python': sys.version,
            'platform': platform.platform(), 'executable': sys.executable,
            'versions': {n: importlib.metadata.version(n) for n in names},
            'clean_environment': {'recreated_from': 'uv.lock', 'pythonpath': None,
                                  'sys_path': sys.path, 'original_path_absent': True},
            'retired_paths_absent': absent, 'unavailable_original_imports': imports,
            'source_unchanged': True, 'source_sha256': source,
            'modern_source_sha256': hashes(ROOT / 'src/shipnav'),
            'frozen_reference_sha256': hashes(ROOT / 'reference'),
            'frozen_comparison_sha256': {p.name: sha256(p.read_bytes()).hexdigest()
                for p in (ROOT / 'migration').glob('*legacy.json')} |
                {'expanded-inputs.json': sha256((ROOT / 'migration/expanded-inputs.json').read_bytes()).hexdigest()},
            'lock_sha256': sha256((ROOT / 'uv.lock').read_bytes()).hexdigest(),
            'checkpoint_sha256': {p: h for p, h in source.items() if p.endswith(('.pth', '.config'))},
            'native_wheels': json.loads((ROOT / 'migration/rvo2/wheels.json').read_text()),
            'evidence_sha256': {p: sha256((ROOT / 'migration' / p).read_bytes()).hexdigest() for p in evidence},
            'smoke': json.loads((ROOT / 'migration/macos-arm64/smoke.json').read_text()),
            'video_probe': json.loads((ROOT / 'migration/macos-arm64/video-probe.json').read_text()),
            'parity_and_replay': json.loads((ROOT / 'migration/m3-summary.json').read_text()),
            'deferred': ['Linux execution', 'CUDA', 'MPS parity',
                         'Native wheel macOS deployment-target portability']}
    (ROOT / 'migration/acceptance.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    print('Accepted modern macOS arm64 CPU; original source unchanged; retired paths and imports absent.')


if __name__ == '__main__':
    record()
