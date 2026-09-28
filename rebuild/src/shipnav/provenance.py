"""Reproducibility metadata attached to service/benchmark results.

Captures enough to reproduce a run: the exact application revision and
working-tree diff, a hash of the settings that drove the run, the
interpreter/platform, and the locked dependency set. Git fields are
``None`` with a ``git_error`` reason when git is unavailable, ``root``
is not a git repository, or any git call fails (all-or-nothing: no
partial revision without its diff) -- never fabricated.
"""
from hashlib import sha256
from pathlib import Path
import platform
import subprocess

from shipnav.maps import canonical_json


def _git(args, root):
    return subprocess.run(['git', *args], cwd=root, capture_output=True, check=True)


def provenance(settings: dict, root: Path) -> dict:
    root = Path(root)
    git_revision = dirty_diff_sha256 = git_error = None
    try:
        git_revision = _git(['rev-parse', 'HEAD'], root).stdout.decode().strip()
        dirty_diff_sha256 = sha256(_git(['diff', 'HEAD'], root).stdout).hexdigest()
    except (OSError, subprocess.CalledProcessError) as error:
        # All-or-nothing: never report a revision without its matching diff.
        git_revision = dirty_diff_sha256 = None
        git_error = str(error)
    uv_lock = root/'rebuild'/'uv.lock'
    uv_lock_sha256 = sha256(uv_lock.read_bytes()).hexdigest() if uv_lock.exists() else None
    return {'git_revision': git_revision, 'dirty_diff_sha256': dirty_diff_sha256, 'git_error': git_error,
            'settings_sha256': sha256(canonical_json(settings).encode()).hexdigest(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
            'platform': platform.platform(),
            'uv_lock_sha256': uv_lock_sha256}
