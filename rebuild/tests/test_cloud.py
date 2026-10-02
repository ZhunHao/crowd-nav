from hashlib import sha256
from pathlib import Path
import json
import os
import stat
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'cloud'))
from budget import estimate  # noqa: E402
from prepare_launch import request  # noqa: E402
from resume_target import first_target  # noqa: E402
from ssm_commands import commands  # noqa: E402
from shlex import quote  # noqa: E402

SETTINGS = {'bucket': 'example-bucket', 'prefix': 'shipnav/run-1', 'seed': 0, 'variant': 'ppo',
            'chunks': 40, 'envs': 8, 'device': 'cpu', 'seconds': 36000, 'run_id': 'run-1', 'budget': None,
            'uv_version': '0.0.0'}
RUNNER = ROOT/'cloud/run-training.sh'


def test_launch_request_is_private_encrypted_and_bounded():
    r = request('ami-0', 'c7i.4xlarge', 'subnet-0', 'sg-0', 'shipnav-profile', '/dev/xvda', 'run-1', 10., 60)
    assert (r['MinCount'], r['MaxCount']) == (1, 1) and 'KeyName' not in r
    assert r['MetadataOptions']['HttpTokens'] == 'required'
    ebs = r['BlockDeviceMappings'][0]['Ebs']
    assert ebs['Encrypted'] and ebs['DeleteOnTermination'] and ebs['VolumeSize'] == 60
    assert r['InstanceInitiatedShutdownBehavior'] == 'stop' and r['ClientToken'] == 'run-1'
    with pytest.raises(ValueError):
        request('ami-0', 't', 's', 'g', 'p', '/dev/xvda', 'run-1', 0., 60)


def test_ssm_documents_set_timeouts_and_quote_values():
    setup = commands('setup', SETTINGS)
    assert setup['executionTimeout'] == ['3600']
    start = commands('start', {**SETTINGS, 'prefix': 'a b; rm -rf /'})
    script = '\n'.join(start['commands'])
    assert 'systemd-run' in script and 'shutdown -h +' in script
    assert quote('s3://example-bucket/a b; rm -rf /') in script and '\nrm -rf' not in script
    with pytest.raises(ValueError):
        commands('start', {**SETTINGS, 'seconds': 0})
    with pytest.raises(ValueError):
        commands('start', {**SETTINGS, 'variant': 'ppo_lagrangian'})  # missing budget


def test_smoke_document_runs_existing_qt_free_test_files():
    script = '\n'.join(commands('smoke', SETTINGS)['commands'])
    assert '-p no:pytestqt' in script
    listed = [word for word in script.split() if word.startswith('tests/')]
    assert listed and all((ROOT/name).is_file() for name in listed)
    for name in listed:
        assert 'Qt' not in (ROOT/name).read_text() and 'qtbot' not in (ROOT/name).read_text()


def test_budget_includes_storage_and_extras():
    e = estimate(hours=10, hourly=1., volume_gib=60, ebs_gib_month=.08, s3_gib=5, s3_gib_month=.025, extra=2.)
    assert e['compute'] == 10. and e['total'] > 12.


def test_runner_script_parses_and_requires_settings(tmp_path):
    subprocess.run(['bash', '-n', str(RUNNER)], check=True)
    result = subprocess.run(['bash', str(RUNNER)], capture_output=True, text=True,
                            env={'PATH': '/usr/bin:/bin'})
    assert result.returncode != 0 and 'required' in result.stderr


# ---- Ruling B: first --chunks target after an instance stop/start ------------------------------------------

def make_checkpoint(root, name, chunk, complete, steps):
    """A verifiable checkpoint folder (payload hashes match) without training a model."""
    directory = Path(root)/name
    directory.mkdir(parents=True)
    payload = {n: f'{name}:{n}'.encode() for n in ('policy.zip', 'state.pt', 'rng.json')}
    for n, data in payload.items():
        (directory/n).write_bytes(data)
    (directory/'manifest.json').write_text(json.dumps({
        'chunk': chunk, 'complete_chunk': complete, 'num_timesteps': steps,
        'files': {n: sha256(data).hexdigest() for n, data in payload.items()}}))
    return directory


def test_first_target_is_one_without_a_checkpoint(tmp_path):
    assert first_target(tmp_path/'missing') == 1
    assert first_target(tmp_path) == 1


def test_first_target_reenters_a_partial_chunk(tmp_path):
    make_checkpoint(tmp_path, 'chunk-0000', 0, True, 100)
    make_checkpoint(tmp_path, 'chunk-0001', 1, True, 200)
    make_checkpoint(tmp_path, 'partial-0002-250', 2, False, 250)
    assert first_target(tmp_path) == 3  # --chunks 3 re-enters chunk 2 and trains the remainder


def test_first_target_starts_the_next_chunk_after_a_complete_one(tmp_path):
    make_checkpoint(tmp_path, 'chunk-0000', 0, True, 100)
    make_checkpoint(tmp_path, 'chunk-0001', 1, True, 200)
    assert first_target(tmp_path) == 3  # chunk 2 is next, i.e. target 3


def test_first_target_can_exceed_the_chunk_count_when_the_run_is_complete(tmp_path):
    chunks = 3
    for c in range(chunks):
        make_checkpoint(tmp_path, f'chunk-{c:04d}', c, True, (c+1)*100)
    assert first_target(tmp_path) == chunks+1 > chunks


def test_first_target_ignores_incomplete_folders_and_rejects_bad_manifests(tmp_path):
    make_checkpoint(tmp_path, 'chunk-0000', 0, True, 100)
    half = make_checkpoint(tmp_path, 'chunk-0001', 1, True, 200)
    (half/'manifest.json').unlink()  # a half-written copy is never a checkpoint
    assert first_target(tmp_path) == 2
    bad = make_checkpoint(tmp_path, 'chunk-0002', 2, True, 300)
    manifest = json.loads((bad/'manifest.json').read_text())
    del manifest['chunk']
    (bad/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='chunk'):
        first_target(tmp_path)


def test_first_target_cli_prints_the_target(tmp_path):
    make_checkpoint(tmp_path, 'partial-0004-450', 4, False, 450)
    done = subprocess.run([sys.executable, str(ROOT/'cloud/resume_target.py'), str(tmp_path)],
                          capture_output=True, text=True, check=True)
    assert done.stdout.strip() == '5'


# ---- the runner's control flow with stand-ins for aws, timeout and the trainer -----------------------------

FAKE_TRAINER = textwrap.dedent('''
    import json, os, sys
    from pathlib import Path
    sys.path.insert(0, {tests!r})
    from test_cloud import make_checkpoint
    from shipnav.training.checkpoints import latest_checkpoint, verify_checkpoint
    argv = sys.argv[1:]
    value = lambda flag: argv[argv.index(flag)+1]
    out, chunks = Path(value('--output')), int(value('--chunks'))
    with open(os.environ['FAKE_LOG'], 'a') as log:
        log.write(json.dumps({{'chunks': chunks, 'resume': '--resume' in argv}})+'\\n')
    last = latest_checkpoint(out)
    first = 0
    if last:
        m = verify_checkpoint(last)
        if not m['complete_chunk'] and m['chunk'] >= chunks:
            raise SystemExit('partial beyond --chunks')
        first = m['chunk']+(1 if m['complete_chunk'] else 0)
    stop_at = int(os.environ.get('FAKE_STOP_AT_CHUNK', -1))
    for chunk in range(first, chunks):
        if chunk == stop_at:
            make_checkpoint(out, f'partial-{{chunk:04d}}-{{chunk*100+50}}', chunk, False, chunk*100+50)
            raise SystemExit(3)
        make_checkpoint(out, f'chunk-{{chunk:04d}}', chunk, True, (chunk+1)*100)
        with (out/'curve.jsonl').open('a') as stream:
            stream.write(json.dumps({{'chunk': chunk}})+'\\n')
''')


@pytest.fixture
def harness(tmp_path):
    """Stub aws (logs calls, captures status.json), timeout (drops its options) and a fake trainer."""
    bin_dir, out, captured = tmp_path/'bin', tmp_path/'out', tmp_path/'captured'
    for d in (bin_dir, out, captured):
        d.mkdir()
    (tmp_path/'trainer.py').write_text(FAKE_TRAINER.format(tests=str(ROOT/'tests')))
    scripts = {
        'aws': f'#!/bin/bash\necho "$*" >> "$AWS_LOG"\n'
               f'[[ "$4" == */status.json ]] && cp "$3" {captured}/status.json\nexit 0\n',
        'timeout': '#!/bin/bash\nwhile [[ "$1" == --* ]]; do shift; done\nshift\nexec "$@"\n',
        'python': f'#!/bin/bash\nif [[ "$1" == -m && "$2" == shipnav.training.train ]]; then\n'
                  f'    exec {sys.executable} {tmp_path}/trainer.py "${{@:3}}"\nfi\nexec {sys.executable} "$@"\n'}
    for name, body in scripts.items():
        (bin_dir/name).write_text(body)
        (bin_dir/name).chmod((bin_dir/name).stat().st_mode | stat.S_IXUSR)
    env = {'PATH': f'{bin_dir}:/usr/bin:/bin', 'AWS_LOG': str(tmp_path/'aws.log'), 'FAKE_LOG': str(tmp_path/'trainer.log'),
           'SHIPNAV_PYTHON': str(bin_dir/'python'), 'SHIPNAV_SCENARIOS': 'scenarios', 'SHIPNAV_OUTPUT': str(out),
           'SHIPNAV_S3_URI': 's3://bucket/prefix', 'SHIPNAV_SEED': '0', 'SHIPNAV_VARIANT': 'ppo', 'SHIPNAV_CHUNKS': '3',
           'SHIPNAV_ENVS': '8', 'SHIPNAV_DEVICE': 'cpu', 'SHIPNAV_SECONDS': '36000'}

    def run(**overrides):
        done = subprocess.run(['bash', str(RUNNER)], capture_output=True, text=True, env={**env, **overrides})
        trainer_log = tmp_path/'trainer.log'
        targets = ([json.loads(line)['chunks'] for line in trainer_log.read_text().splitlines()]
                   if trainer_log.exists() else [])
        uploads = [line.split()[2:4] for line in (tmp_path/'aws.log').read_text().splitlines()]
        status = json.loads((captured/'status.json').read_text())
        return done, targets, uploads, status
    return run, out


def published(uploads, name):
    return [Path(dst).name for _, dst in uploads if f'/checkpoints/{name}/' in dst]


def test_runner_trains_every_chunk_publishing_manifest_last(harness):
    run, _ = harness
    done, targets, uploads, status = run()
    assert done.returncode == 0, done.stderr
    assert targets == [1, 2, 3] and status == {'state': 'complete', 'exit_code': 0}
    for c in range(3):
        assert published(uploads, f'chunk-{c:04d}')[-1] == 'manifest.json'
        assert set(published(uploads, f'chunk-{c:04d}')) == {'policy.zip', 'state.pt', 'rng.json', 'manifest.json'}
    assert uploads[-1] == ['/'.join(uploads[-1][0].split('/')[:-1])+'/run-status.json', 's3://bucket/prefix/status.json']
    assert ['s3://bucket/prefix/curve.jsonl'] == [dst for _, dst in uploads if dst.endswith('curve.jsonl')]


def test_runner_resumes_after_a_complete_chunk_without_restarting_at_one(harness):
    run, out = harness
    make_checkpoint(out, 'chunk-0000', 0, True, 100)
    make_checkpoint(out, 'chunk-0001', 1, True, 200)
    done, targets, _, status = run(SHIPNAV_CHUNKS='4')
    assert done.returncode == 0, done.stderr
    assert targets == [3, 4] and status['state'] == 'complete'


def test_runner_reenters_a_partial_chunk(harness):
    run, out = harness
    make_checkpoint(out, 'chunk-0000', 0, True, 100)
    make_checkpoint(out, 'partial-0001-150', 1, False, 150)
    done, targets, uploads, status = run()
    assert done.returncode == 0, done.stderr
    assert targets == [2, 3] and status['state'] == 'complete'
    assert 'chunk-0001' in [Path(dst).parent.name for _, dst in uploads]


def test_runner_reports_an_already_complete_run_without_training(harness):
    run, out = harness
    for c in range(3):
        make_checkpoint(out, f'chunk-{c:04d}', c, True, (c+1)*100)
    done, targets, uploads, status = run()
    assert done.returncode == 0, done.stderr
    assert targets == [] and status == {'state': 'complete', 'exit_code': 0}
    assert published(uploads, 'chunk-0002')[-1] == 'manifest.json'  # an unpublished final checkpoint still goes up


def test_runner_publishes_the_partial_checkpoint_on_a_stop(harness):
    run, out = harness
    done, targets, uploads, status = run(FAKE_STOP_AT_CHUNK='1')
    assert done.returncode == 0, done.stderr
    assert targets == [1, 2] and status == {'state': 'stopped', 'exit_code': 0}
    assert published(uploads, 'partial-0001-150') == ['policy.zip', 'state.pt', 'rng.json', 'manifest.json']
    assert not (out/'chunk-0001').exists()


def test_runner_uploads_failed_status_when_the_trainer_fails(harness):
    run, out = harness
    make_checkpoint(out, 'chunk-0000', 0, True, 100)
    (out/'chunk-0000'/'policy.zip').write_bytes(b'tampered')  # hash mismatch propagates from latest_checkpoint
    done, _, _, status = run()
    assert done.returncode != 0 and status['state'] == 'failed' and status['exit_code'] == done.returncode
