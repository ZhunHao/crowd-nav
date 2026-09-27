"""Install only the optional wheel whose SHA-256 is in the retained manifest."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def install(directory, python):
    entries = json.loads((directory/'wheels.json').read_text())
    if len(entries) != 1:
        raise ValueError('Expected exactly one platform wheel in manifest')
    name,digest = next(iter(entries.items()))
    if Path(name).name != name or not name.endswith('.whl'):
        raise ValueError('Invalid wheel filename')
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
