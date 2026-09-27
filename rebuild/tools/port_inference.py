"""Copy the frozen inference closure; no original package initializers are used."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

FILES = ['crowd_nav/policy/sarl.py', 'crowd_nav/policy/cadrl.py',
         'crowd_nav/policy/multi_human_rl.py', 'crowd_sim/envs/policy/policy.py',
         'crowd_sim/envs/policy/orca.py', 'crowd_sim/envs/utils/state.py',
         'crowd_sim/envs/utils/action.py']


def port(source: Path, target: Path):
    # Require the recorded baseline before copying any supplied source.
    manifest_path = Path(__file__).resolve().parents[1] / 'reference/manifest.json'
    manifest = json.loads(manifest_path.read_text())['source']
    for name in FILES:
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != manifest[name]:
            raise ValueError(f'Source differs from baseline manifest: {name}')
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        out = target / name
        out.parent.mkdir(parents=True, exist_ok=True)
        code = (source / name).read_text()
        code = code.replace('from crowd_nav.', 'from shipnav.compat.crowd_nav.')
        code = code.replace('from crowd_sim.', 'from shipnav.compat.crowd_sim.')
        if name.endswith('/orca.py'):
            code = code.replace('import rvo2\n', '')
            code = code.replace('        self_state = state.self_state\n',
                                '        import rvo2  # Optional native dependency, needed only for ORCA execution.\n\n        self_state = state.self_state\n')
        out.write_text(code.rstrip() + '\n')
        parent = out.parent
        while parent != target.parent:
            (parent / '__init__.py').write_text('')
            parent = parent.parent
    shutil.copyfile(Path(__file__).with_name('CROWDNAV-LICENSE'), target / 'LICENSE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument('--source', type=Path, default=root / 'CrowdNav-20250813-DIP')
    parser.add_argument('--target', type=Path, default=root / 'rebuild/src/shipnav/compat')
    args = parser.parse_args()
    port(args.source, args.target)
