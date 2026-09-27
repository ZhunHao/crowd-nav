from hashlib import sha256
from pathlib import Path
import json, platform, subprocess, sys

def manifest(root: Path) -> dict:
    return {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()
            and '__pycache__' not in p.parts and '.pyc' != p.suffix}

if __name__ == '__main__':
    root = Path('../CrowdNav-20250813-DIP')
    out = Path('reference'); out.mkdir(exist_ok=True)
    data = {'source': manifest(root), 'python': sys.version,
            'platform': platform.platform(),
            'packages': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)}
    (out/'manifest.json').write_text(json.dumps(data, indent=2))
