"""SSM AWS-RunShellScript parameter documents. Values are shell-quoted; nothing is concatenated raw."""
from pathlib import Path
from shlex import quote
import argparse
import json

WORK = '/opt/shipnav'
UV_SYNC = 'uv sync --locked --extra train --extra maps'
# systemd waits this long after TERM before SIGKILL: the trainer's 120 s kill-after (run-training.sh) plus
# time for the partial-checkpoint upload and the status retries.
STOP_TIMEOUT_S = 300
VARIANTS = ('ppo', 'ppo_lagrangian')


def _check(s):
    if not s['seconds'] > 0 or s['chunks'] < 1 or s['envs'] < 1:
        raise ValueError('Positive seconds, chunks and envs required')
    if s['variant'] not in VARIANTS or not isinstance(s['seed'], int) or isinstance(s['seed'], bool) or s['seed'] < 0:
        raise ValueError('variant must be ppo or ppo_lagrangian and seed a non-negative integer')
    if s['variant'] == 'ppo_lagrangian' and s.get('budget') is None:
        raise ValueError('ppo_lagrangian needs the preregistered budget')


def commands(kind, s):
    _check(s)
    uri = quote(f"s3://{s['bucket']}/{s['prefix']}")
    if kind == 'setup':
        body = ['set -euo pipefail', f'mkdir -p {WORK} && cd {WORK}',
                f'aws s3 cp {uri}/source.tar.gz source.tar.gz', f'aws s3 cp {uri}/source.sha256 source.sha256',
                'sha256sum -c source.sha256', 'tar -xzf source.tar.gz', 'ldd --version | head -1',
                f"command -v uv || curl -LsSf {quote('https://astral.sh/uv/'+s['uv_version']+'/install.sh')} | sh",
                f'cd rebuild && $HOME/.local/bin/uv python install 3.14 && $HOME/.local/bin/{UV_SYNC}']
        timeout = 3600
    elif kind == 'smoke':
        body = ['set -euo pipefail', f'cd {WORK}/rebuild', 'nvidia-smi || true',
                '.venv/bin/python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"',
                'MPLBACKEND=Agg .venv/bin/python -m pytest -p no:pytestqt -q tests/test_training_env.py '
                'tests/test_checkpoint.py tests/test_train.py tests/test_modern_learned.py']
        timeout = 3600
    elif kind == 'start':
        minutes = int(s['seconds']//60) + 30  # training deadline + upload grace
        env = {'SHIPNAV_PYTHON': f'{WORK}/rebuild/.venv/bin/python', 'SHIPNAV_SCENARIOS': f'{WORK}/rebuild/scenarios/training',
               'SHIPNAV_OUTPUT': f"{WORK}/out/{s['variant']}/seed{s['seed']}",  # one directory per run: sequential runs never mix
               'SHIPNAV_S3_URI': f"s3://{s['bucket']}/{s['prefix']}",
               'SHIPNAV_SEED': s['seed'], 'SHIPNAV_VARIANT': s['variant'], 'SHIPNAV_CHUNKS': s['chunks'],
               'SHIPNAV_ENVS': s['envs'], 'SHIPNAV_DEVICE': s['device'], 'SHIPNAV_SECONDS': int(s['seconds']),
               'SHIPNAV_BUDGET': '' if s.get('budget') is None else s['budget']}
        setenv = ' '.join(f'--setenv={k}={quote(str(v))}' for k, v in env.items())
        body = ['set -euo pipefail', f'sudo shutdown -h +{minutes}',
                f"sudo systemd-run --unit={quote('shipnav-'+s['run_id'])} --property=RuntimeMaxSec={minutes*60} "
                f'--property=TimeoutStopSec={STOP_TIMEOUT_S} --working-directory={WORK}/rebuild {setenv} '
                f'/bin/bash {WORK}/rebuild/cloud/run-training.sh']
        timeout = 600
    else:
        raise ValueError('kind must be setup, smoke or start')
    return {'commands': body, 'executionTimeout': [str(timeout)]}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('kind', choices=('setup', 'smoke', 'start'))
    p.add_argument('--settings', type=Path, required=True)
    a = p.parse_args()
    out = Path(f'cloud/local/{a.kind}-commands.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(commands(a.kind, json.loads(a.settings.read_text())), indent=2))
    print(out)
