"""SSM AWS-RunShellScript parameter documents. Values are shell-quoted; nothing is concatenated raw."""
from pathlib import Path
from shlex import quote
import argparse
import json
import re

from identifiers import check_run_id

WORK = '/opt/shipnav'
HOME = '/root'  # SSM runs as root without a login shell, so HOME is set explicitly
UV_DIR = '/usr/local/bin'  # explicit install path: never depends on $HOME or PATH
UV = f'{UV_DIR}/uv'
UV_SYNC = 'sync --locked --extra train --extra maps'
# systemd waits this long after TERM before SIGKILL: the trainer's 120 s kill-after (run-training.sh) plus
# time for the partial-checkpoint upload and the status retries.
STOP_TIMEOUT_S = 300
VARIANTS = ('ppo', 'ppo_lagrangian')
COMMIT = re.compile(r'[0-9a-f]{40}')
REGION = re.compile(r'[a-z]{2}(-[a-z]+)+-[0-9]')
TRAINING_CONFIG = Path(__file__).resolve().parents[1]/'evidence/retraining/training_config.json'


def _check(s):
    check_run_id(s['run_id'])
    if not s['seconds'] > 0 or s['chunks'] < 1 or s['envs'] < 1:
        raise ValueError('Positive seconds, chunks and envs required')
    if s['variant'] not in VARIANTS or not isinstance(s['seed'], int) or isinstance(s['seed'], bool) or s['seed'] < 0:
        raise ValueError('variant must be ppo or ppo_lagrangian and seed a non-negative integer')
    if s['variant'] == 'ppo_lagrangian' and s.get('budget') is None:
        raise ValueError('ppo_lagrangian needs the preregistered budget')
    if not isinstance(s.get('region'), str) or not REGION.fullmatch(s['region']):
        raise ValueError('region must be an AWS region name such as eu-west-1')
    if not isinstance(s.get('source_commit'), str) or not COMMIT.fullmatch(s['source_commit']):
        raise ValueError('source_commit must be the 40-hex git commit that was archived')


def check_training_config(s, path=TRAINING_CONFIG):
    """Refuse settings that disagree with the frozen training configuration (when one exists)."""
    if not Path(path).is_file():
        return
    config = json.loads(Path(path).read_text())
    problems = []
    if s['variant'] not in config['variants']:
        problems.append(f"variant {s['variant']!r} is not in {config['variants']}")
    if s['seed'] not in config['seeds']:
        problems.append(f"seed {s['seed']} is not in {config['seeds']}")
    if s['chunks'] != config['chunks']:
        problems.append(f"chunks {s['chunks']} != {config['chunks']}")
    if s['envs'] != config['envs']:
        problems.append(f"envs {s['envs']} != {config['envs']}")
    budget = config['budget_domain_s'] if s['variant'] == 'ppo_lagrangian' else None
    if s.get('budget') != budget:
        problems.append(f"budget {s.get('budget')!r} != {budget!r}")
    if s['source_commit'] != config['source_commit']:
        problems.append(f"source_commit {s['source_commit']} != the frozen {config['source_commit']} "
                        '(archive that commit: the checkpoints must carry it)')
    if problems:
        raise ValueError('Settings disagree with the frozen training_config.json: '+'; '.join(problems))


def commands(kind, s):
    _check(s)
    uri = quote(f"s3://{s['bucket']}/{s['prefix']}")
    base = ['set -euo pipefail', f'export HOME={HOME}', f"export AWS_DEFAULT_REGION={quote(s['region'])}"]
    if kind == 'setup':
        body = [*base, f'mkdir -p {WORK} && cd {WORK}',
                f'aws s3 cp {uri}/source.tar.gz source.tar.gz', f'aws s3 cp {uri}/source.sha256 source.sha256',
                'sha256sum -c source.sha256', 'tar -xzf source.tar.gz', 'ldd --version | head -1',
                f"[ -x {UV} ] || curl -LsSf {quote('https://astral.sh/uv/'+s['uv_version']+'/install.sh')} "
                f'| env UV_INSTALL_DIR={UV_DIR} UV_NO_MODIFY_PATH=1 sh',
                f'cd rebuild && {UV} python install 3.14 && {UV} {UV_SYNC}']
        timeout = 3600
    elif kind == 'smoke':
        body = [*base, f'cd {WORK}/rebuild', 'nvidia-smi || true',
                '.venv/bin/python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"',
                'MPLBACKEND=Agg .venv/bin/python -m pytest -p no:pytestqt -q tests/test_training_env.py '
                'tests/test_checkpoint.py tests/test_train.py tests/test_modern_learned.py']
        timeout = 3600
    elif kind == 'pilot':
        out = f"{WORK}/out/pilot-{s['run_id']}.json"
        body = [*base, f'mkdir -p {WORK}/out && cd {WORK}/rebuild',
                f"MPLBACKEND=Agg .venv/bin/python tools/training_pilot.py --devices {quote(s['device'])} "
                f"--output {quote(out)}",
                f"aws s3 cp {quote(out)} {uri}/pilot-{s['run_id']}.json --only-show-errors"]
        timeout = 3600
    elif kind == 'start':
        minutes = int(s['seconds']//60) + 30  # training deadline + upload grace
        env = {'HOME': HOME, 'AWS_DEFAULT_REGION': s['region'], 'SHIPNAV_SOURCE_COMMIT': s['source_commit'],
               'SHIPNAV_PYTHON': f'{WORK}/rebuild/.venv/bin/python', 'SHIPNAV_SCENARIOS': f'{WORK}/rebuild/scenarios/training',
               'SHIPNAV_OUTPUT': f"{WORK}/out/{s['variant']}/seed{s['seed']}",  # one directory per run: sequential runs never mix
               'SHIPNAV_S3_URI': f"s3://{s['bucket']}/{s['prefix']}",
               'SHIPNAV_SEED': s['seed'], 'SHIPNAV_VARIANT': s['variant'], 'SHIPNAV_CHUNKS': s['chunks'],
               'SHIPNAV_ENVS': s['envs'], 'SHIPNAV_DEVICE': s['device'], 'SHIPNAV_SECONDS': int(s['seconds']),
               'SHIPNAV_BUDGET': '' if s.get('budget') is None else s['budget']}
        setenv = ' '.join(f'--setenv={k}={quote(str(v))}' for k, v in env.items())
        body = ['set -euo pipefail', f'sudo shutdown -h +{minutes}',
                f"sudo systemd-run --unit={quote('shipnav-'+s['run_id'])} --property=RuntimeMaxSec={minutes*60} "
                '--property=After=network-online.target --property=Wants=network-online.target '
                f'--property=TimeoutStopSec={STOP_TIMEOUT_S} --working-directory={WORK}/rebuild {setenv} '
                f'/bin/bash {WORK}/rebuild/cloud/run-training.sh']
        timeout = 600
    else:
        raise ValueError('kind must be setup, smoke, pilot or start')
    return {'commands': body, 'executionTimeout': [str(timeout)]}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('kind', choices=('setup', 'smoke', 'pilot', 'start'))
    p.add_argument('--settings', type=Path, required=True)
    a = p.parse_args()
    settings = json.loads(a.settings.read_text())
    _check(settings)
    check_training_config(settings)
    out = Path(f'cloud/local/{a.kind}-commands.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(commands(a.kind, settings), indent=2))
    print(out)
