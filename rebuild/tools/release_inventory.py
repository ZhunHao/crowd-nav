"""Capture fresh PyPI release metadata, including artifact hashes and yank status."""
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
    Path('migration/releases.json').write_text(json.dumps(inventory(fetch), indent=2) + '\n')
