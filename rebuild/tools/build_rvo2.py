"""Build the optional native wheel from an isolated, patched source copy."""
from pathlib import Path
import argparse
import difflib
import json
import platform
import os
import shutil
import subprocess
import tempfile
import hashlib

ROOT = Path(__file__).resolve().parents[1]


def require_local_platform():
    """These artifacts record only the accepted macOS 26 arm64 CPU host."""
    if (platform.system() != 'Darwin' or platform.machine() != 'arm64'
            or platform.mac_ver()[0].split('.')[0] != '26'):
        raise RuntimeError('This recorder supports only macOS 26 arm64; other platforms require separate acceptance tooling')



def build(output):
    require_local_platform()
    source = ROOT.parent / 'CrowdNav-20250813-DIP/Python-RVO2-main'
    expected = json.loads((ROOT/'reference/manifest.json').read_text())['source']
    prefix = 'Python-RVO2-main/'
    for name, digest in expected.items():
        if name.startswith(prefix):
            path = source / name[len(prefix):]
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Native source differs from baseline: {name}')
    output.mkdir(parents=True, exist_ok=True)
    # Refuse mixed or obsolete artifacts before overwriting any evidence.
    expected_wheel = 'pyrvo2-0.0.0-cp314-cp314-macosx_26_0_arm64.whl'
    if any(p.name != expected_wheel for p in output.glob('*.whl')):
        raise ValueError('Output contains another platform/target wheel; use a clean --output directory')
    with tempfile.TemporaryDirectory(prefix='shipnav-rvo2-') as temp:
        tmp = Path(temp)
        copy = tmp / 'source'
        shutil.copytree(source, copy, ignore=shutil.ignore_patterns('build', '*.so', '*.egg-info', '__pycache__', '.git'))
        patches = []
        for name, old, new in [('CMakeLists.txt', 'VERSION 2.8', 'VERSION 3.5'),
                               ('setup.py', "'-DCMAKE_CXX_FLAGS=-fPIC'", "'-DCMAKE_CXX_FLAGS=-fPIC', '-DCMAKE_OSX_DEPLOYMENT_TARGET=26.0'"),
                               ('setup.py', "extra_compile_args=['-fPIC']", "extra_compile_args=['-fPIC', '-mmacosx-version-min=26.0'], extra_link_args=['-mmacosx-version-min=26.0']"),
                               ('setup.py', 'name="pyrvo2",', 'name="pyrvo2",\n    version="0.0.0",')]:
            path = copy / name
            before = path.read_text()
            after = before.replace(old, new)
            path.write_text(after)
            patches.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='a/'+name, tofile='b/'+name, n=0))
        (output / 'modern.patch').write_text(''.join(patches))
        for path in source.glob('*LICENSE*'):
            shutil.copyfile(path, output / path.name)
        env_dir = tmp / 'build-env'
        with (output / 'build.log').open('w') as log:
            def run(args, **kwargs):
                log.write('$ ' + ' '.join(map(str, args)) + '\n'); log.flush()
                subprocess.run(args, check=True, stdout=log, stderr=subprocess.STDOUT, **kwargs)
            run(['uv', 'venv', '--python', str(ROOT / '.venv-modern/bin/python'), str(env_dir)])
            python = env_dir / 'bin/python'
            run(['uv', 'pip', 'install', '--python', str(python), '--require-hashes', '-r', str(ROOT/'migration/rvo2/build-requirements.txt')])
            env = dict(os.environ, MACOSX_DEPLOYMENT_TARGET='26.0', PATH=str(env_dir/'bin')+os.pathsep+os.environ['PATH'])
            run([str(python), 'setup.py', 'bdist_wheel', '--dist-dir', str(output.resolve())], cwd=copy, env=env)
        wheels = list(output.glob('*.whl'))
        (output / 'wheels.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in wheels}, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'migration/rvo2')
    build(parser.parse_args().output)
