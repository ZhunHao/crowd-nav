"""The accepted application must run without the retired original runtime."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RETIRED = ('.venv-legacy', 'legacy', 'locks/legacy.txt', 'src/shipnav/baseline.py',
           'tests/legacy', 'tools/reference_manifest.py', 'tools/capture_reference.py')


def test_retired_paths_absent():
    assert not [p for p in RETIRED if (ROOT / p).exists()]


def test_modern_import_isolation():
    code = '''
import importlib.util, sys
from shipnav.model import load_policy
from shipnav.replay import replay
from shipnav.compat.crowd_sim.envs.policy.orca import ORCA
for name in ('shipnav.baseline', 'crowd_nav', 'crowd_sim', 'gym'):
    assert importlib.util.find_spec(name) is None, name
assert not any(n == 'gym' or n.startswith(('gym.', 'crowd_nav', 'crowd_sim')) for n in sys.modules)
assert not any('CrowdNav-20250813-DIP' in p or '.venv-legacy' in p for p in sys.path)
'''
    subprocess.run([sys.executable, '-c', code], check=True)


def test_capture_tools_have_one_loader():
    for name in ('parity_fixture.py', 'expanded_parity.py', 'replay_migration.py'):
        text = (ROOT / 'tools' / name).read_text()
        assert 'shipnav.baseline' not in text
        assert '--modern' not in text
        assert '--freeze' not in text
