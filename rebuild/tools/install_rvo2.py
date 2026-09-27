"""Install only the optional wheel whose SHA-256 is in the retained manifest."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]



def require_local_platform():
    """These artifacts record only the accepted macOS 26 arm64 CPU host."""
    if (platform.system() != 'Darwin' or platform.machine() != 'arm64'
            or platform.mac_ver()[0].split('.')[0] != '26'):
        raise RuntimeError('This recorder supports only macOS 26 arm64; other platforms require separate acceptance tooling')



def install(directory, python):
    require_local_platform()
    entries = json.loads((directory/'wheels.json').read_text())
    if len(entries) != 1:
        raise ValueError('Expected exactly one platform wheel in manifest')
    name,digest = next(iter(entries.items()))
    if Path(name).name != name or not name.endswith('.whl'):
        raise ValueError('Invalid wheel filename')
    if not name.endswith('-cp314-cp314-macosx_26_0_arm64.whl'):
        raise ValueError('Expected the accepted CPython 3.14 macOS 26 arm64 wheel')
    wheel = directory/name
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != digest:
        raise ValueError('Native wheel hash mismatch')
    subprocess.run(['uv','pip','install','--reinstall','--python',str(python),str(wheel)],check=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=ROOT/'migration/rvo2')
    p.add_argument('--python',type=Path,default=Path(sys.executable))
    args = p.parse_args()
    install(args.directory,args.python)
