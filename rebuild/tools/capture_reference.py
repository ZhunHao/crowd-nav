"""Capture bounded legacy episodes and verify paired deterministic reruns."""
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
from shipnav.baseline import run_baseline


def compare(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            compare(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            compare(a, b)
    elif isinstance(left, (float, int)) and not isinstance(left, bool):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
    else:
        assert left == right


if __name__ == '__main__':
    model = Path('../CrowdNav-20250813-DIP/crowd_nav/data/output_trained')
    reference = Path('reference/episodes')
    reference.mkdir(parents=True, exist_ok=True)
    checks = []
    for seed in (0, 1, 2):
        outputs = [Path(f'results/determinism/seed-{seed}-run-{run}') for run in (1, 2)]
        for output in outputs:
            run_baseline(model, output, seed)
        for filename in ('baseline.json', 'trace.json'):
            compare(*(json.loads((output / filename).read_text()) for output in outputs))
            target = reference / f'seed-{seed}-{filename}'
            target.write_bytes((outputs[0] / filename).read_bytes())
        checks.append({'seed': seed, 'runs': 2, 'passed': True})
    hashes = {str(p): sha256(p.read_bytes()).hexdigest() for p in sorted(reference.glob('*.json'))}
    Path('reference/determinism.json').write_text(json.dumps(
        {'rtol': 1e-6, 'atol': 1e-7, 'excluded': ['wall_clock', 'video_bytes'],
         'checks': checks, 'output_sha256': hashes}, indent=2))
    print(json.dumps(checks))
