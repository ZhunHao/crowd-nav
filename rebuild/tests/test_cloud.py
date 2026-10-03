from hashlib import sha256
from pathlib import Path
import json
import os
import shlex
import signal
import stat
import time
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
    assert '--property=TimeoutStopSec=300 ' in script  # outlasts the trainer's 120 s kill-after plus uploads
    unit = shlex.split(next(line for line in start['commands'] if 'systemd-run' in line))
    assert 'SHIPNAV_S3_URI=s3://example-bucket/a b; rm -rf /' in ' '.join(unit)  # quoting survives splitting
    assert any(w == '--setenv=SHIPNAV_S3_URI=s3://example-bucket/a b; rm -rf /' for w in unit)
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

RUN_IDENTITY = {'variant': 'ppo', 'seed': 0, 'envs': 8, 'chunk_steps': 20480, 'budget': None}


def make_checkpoint(root, name, chunk, complete, steps, **identity):
    """A verifiable checkpoint folder (payload hashes match) without training a model."""
    directory = Path(root)/name
    directory.mkdir(parents=True)
    payload = {n: f'{name}:{n}'.encode() for n in ('policy.zip', 'state.pt', 'rng.json')}
    for n, data in payload.items():
        (directory/n).write_bytes(data)
    (directory/'manifest.json').write_text(json.dumps({
        **RUN_IDENTITY, **identity, 'chunk': chunk, 'complete_chunk': complete, 'num_timesteps': steps,
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
    ident = dict(variant=value('--variant'), seed=int(value('--seed')), envs=int(value('--envs')),
                 chunk_steps=int(value('--chunk-steps')) if '--chunk-steps' in argv else 20480,
                 budget=float(value('--budget')) if '--budget' in argv else None)
    last = latest_checkpoint(out)
    first = 0
    if last:
        m = verify_checkpoint(last)
        if not m['complete_chunk'] and m['chunk'] >= chunks:
            raise SystemExit('partial beyond --chunks')
        first = m['chunk']+(1 if m['complete_chunk'] else 0)
    stop_at = int(os.environ.get('FAKE_STOP_AT_CHUNK', -1))
    hang_at = int(os.environ.get('FAKE_HANG_AT_CHUNK', -1))
    for chunk in range(first, chunks):
        if chunk == hang_at:  # "running": on TERM save a partial checkpoint and exit 3, like the real trainer
            import signal, time
            def on_term(*_):
                signal.signal(signal.SIGTERM, signal.SIG_IGN)  # a repeat TERM is harmless
                make_checkpoint(out, f'partial-{{chunk:04d}}-{{chunk*100+50}}', chunk, False, chunk*100+50, **ident)
                raise SystemExit(3)
            signal.signal(signal.SIGTERM, on_term)
            Path(os.environ['FAKE_READY']).write_text('ready')
            for _ in range(600):  # at most 30 s: never hangs the suite
                time.sleep(0.05)
            raise SystemExit(99)
        if chunk == stop_at:
            make_checkpoint(out, f'partial-{{chunk:04d}}-{{chunk*100+50}}', chunk, False, chunk*100+50, **ident)
            raise SystemExit(3)
        make_checkpoint(out, f'chunk-{{chunk:04d}}', chunk, True, (chunk+1)*100, **ident)
        with (out/'curve.jsonl').open('a') as stream:
            stream.write(json.dumps({{'chunk': chunk}})+'\\n')
''')


def build_harness(tmp_path):
    """Stub aws (logs calls, captures status.json, can fail it N times), timeout (drops its options) and a fake trainer."""
    bin_dir, out, captured = tmp_path/'bin', tmp_path/'out', tmp_path/'captured'
    for d in (bin_dir, out, captured):
        d.mkdir()
    (tmp_path/'trainer.py').write_text(FAKE_TRAINER.format(tests=str(ROOT/'tests')))
    scripts = {
        'aws': f'#!/bin/bash\necho "$*" >> "$AWS_LOG"\n'
               f'if [[ -n "${{AWS_FAIL_MATCH:-}}" && "$4" == *"$AWS_FAIL_MATCH"* ]]; then exit 1; fi\n'
               f'if [[ "$4" == */status.json ]]; then\n'
               f'    n=$(cat "$AWS_COUNT" 2>/dev/null || echo 0)\n'
               f'    if ((n < ${{AWS_FAIL_STATUS:-0}})); then echo $((n+1)) > "$AWS_COUNT"; exit 1; fi\n'
               f'    cp "$3" {captured}/status.json\nfi\nexit 0\n',
        'timeout': '#!/bin/bash\nwhile [[ "$1" == --* ]]; do shift; done\nshift\nexec "$@"\n',
        'python': f'#!/bin/bash\nif [[ "$1" == -m && "$2" == shipnav.training.train ]]; then\n'
                  f'    exec {sys.executable} {tmp_path}/trainer.py "${{@:3}}"\nfi\nexec {sys.executable} "$@"\n'}
    for name, body in scripts.items():
        (bin_dir/name).write_text(body)
        (bin_dir/name).chmod((bin_dir/name).stat().st_mode | stat.S_IXUSR)
    env = {'PATH': f'{bin_dir}:/usr/bin:/bin', 'AWS_LOG': str(tmp_path/'aws.log'), 'AWS_COUNT': str(tmp_path/'aws.count'),
           'FAKE_LOG': str(tmp_path/'trainer.log'), 'FAKE_READY': str(tmp_path/'ready'), 'SHIPNAV_RETRY_SLEEP': '0',
           'SHIPNAV_PYTHON': str(bin_dir/'python'), 'SHIPNAV_SCENARIOS': 'scenarios', 'SHIPNAV_OUTPUT': str(out),
           'SHIPNAV_S3_URI': 's3://bucket/prefix', 'SHIPNAV_SEED': '0', 'SHIPNAV_VARIANT': 'ppo', 'SHIPNAV_CHUNKS': '3',
           'SHIPNAV_ENVS': '8', 'SHIPNAV_DEVICE': 'cpu', 'SHIPNAV_SECONDS': '36000'}

    def results():
        trainer_log = tmp_path/'trainer.log'
        targets = ([json.loads(line)['chunks'] for line in trainer_log.read_text().splitlines()]
                   if trainer_log.exists() else [])
        uploads = [line.split()[2:4] for line in (tmp_path/'aws.log').read_text().splitlines()]
        status_file = captured/'status.json'
        return targets, uploads, json.loads(status_file.read_text()) if status_file.exists() else None

    def run(**overrides):
        done = subprocess.run(['bash', str(RUNNER)], capture_output=True, text=True, env={**env, **overrides})
        return (done, *results())

    def start(**overrides):  # own session, so os.killpg can stand in for systemd's control-group TERM
        return subprocess.Popen(['bash', str(RUNNER)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                env={**env, **overrides}, start_new_session=True)
    return run, start, results, out, tmp_path/'ready'


@pytest.fixture
def harness(tmp_path):
    run, _, _, out, _ = build_harness(tmp_path)
    return run, out


@pytest.fixture
def terminable(tmp_path):
    _, start, results, out, ready = build_harness(tmp_path)
    return start, results, out, ready


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


def terminate_while_training(terminable, group):
    start, results, out, ready = terminable
    proc = start(FAKE_HANG_AT_CHUNK='1')
    try:
        deadline = time.monotonic()+60
        while not ready.exists():
            assert proc.poll() is None and time.monotonic() < deadline, 'fake trainer never started'
            time.sleep(.05)
        (os.killpg if group else os.kill)(proc.pid, signal.SIGTERM)
        stdout, stderr = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    return proc.returncode, stderr, *results(), out


@pytest.mark.parametrize('group', [False, True], ids=['runner-only', 'whole-group'])
def test_runner_publishes_partial_and_a_true_status_when_terminated(terminable, group):
    code, stderr, targets, uploads, status, out = terminate_while_training(terminable, group)
    assert code == 143, stderr
    assert targets == [1, 2] and status == {'state': 'terminated', 'exit_code': 143}
    assert (out/'partial-0001-150').is_dir() and not (out/'chunk-0001').exists()
    partial = [i for i, (_, dst) in enumerate(uploads) if '/checkpoints/partial-0001-150/' in dst]
    assert [Path(uploads[i][1]).name for i in partial] == ['policy.zip', 'state.pt', 'rng.json', 'manifest.json']
    assert len(partial) == 4  # published once: the EXIT trap does not repeat an upload that succeeded
    assert uploads[-1][1] == 's3://bucket/prefix/status.json' and partial[-1] < len(uploads)-1


def test_status_upload_is_retried_then_succeeds(harness, tmp_path):
    run, _ = harness
    done, _, uploads, status = run(AWS_FAIL_STATUS='2')
    assert done.returncode == 0, done.stderr
    assert status == {'state': 'complete', 'exit_code': 0}
    assert [dst for _, dst in uploads].count('s3://bucket/prefix/status.json') == 3


def test_status_upload_failure_after_three_attempts_is_a_failure(harness):
    run, _ = harness
    done, _, uploads, _ = run(AWS_FAIL_STATUS='99')
    assert done.returncode == 1 and 'status.json upload failed' in done.stderr
    assert [dst for _, dst in uploads].count('s3://bucket/prefix/status.json') == 3


def test_failed_payload_upload_never_leads_to_a_manifest_in_the_best_effort_path(terminable):
    """EXIT-trap publish runs as `publish_latest || true`, where errexit is off: it must still stop at the failure."""
    start, results, out, ready = terminable
    proc = start(FAKE_HANG_AT_CHUNK='1', AWS_FAIL_MATCH='partial-0001-150/policy.zip')
    try:
        deadline = time.monotonic()+60
        while not ready.exists():
            assert proc.poll() is None and time.monotonic() < deadline, 'fake trainer never started'
            time.sleep(.05)
        os.kill(proc.pid, signal.SIGTERM)
        proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    _, uploads, status = results()
    assert proc.returncode == 143 and status == {'state': 'terminated', 'exit_code': 143}
    assert published(uploads, 'partial-0001-150') == ['policy.zip']  # attempted, failed, nothing after it
    assert published(uploads, 'chunk-0000')[-1] == 'manifest.json'  # the earlier, healthy checkpoint is whole
    assert uploads[-1][1] == 's3://bucket/prefix/status.json'


@pytest.mark.parametrize('failing', ['policy.zip', 'state.pt', 'rng.json'])
def test_failed_payload_upload_in_the_loop_fails_the_run_without_a_manifest(harness, failing):
    run, _ = harness
    done, targets, uploads, status = run(AWS_FAIL_MATCH=f'chunk-0000/{failing}')
    assert done.returncode != 0 and targets == [1]
    assert 'manifest.json' not in published(uploads, 'chunk-0000')
    assert published(uploads, 'chunk-0000')[-1] == failing
    assert status == {'state': 'failed', 'exit_code': done.returncode}
    assert uploads[-1][1] == 's3://bucket/prefix/status.json'


# ---- C1: a checkpoint from another run is never published, resumed from or reported complete ---------------

EXPECTED = {'variant': 'ppo', 'seed': 0, 'envs': 8, 'chunk_steps': 20480, 'budget': None}


@pytest.mark.parametrize('field, other', [('variant', 'ppo_lagrangian'), ('seed', 1), ('envs', 4),
                                          ('chunk_steps', 2048), ('budget', 30.)])
def test_first_target_refuses_a_checkpoint_from_another_run(tmp_path, field, other):
    make_checkpoint(tmp_path, 'chunk-0000', 0, True, 100, **{field: other})
    with pytest.raises(ValueError, match=field):
        first_target(tmp_path, expected=EXPECTED)
    assert first_target(tmp_path) == 2  # without the run's settings only the chunk bookkeeping is checked


def test_first_target_accepts_a_checkpoint_of_this_run(tmp_path):
    make_checkpoint(tmp_path, 'chunk-0000', 0, True, 100)
    assert first_target(tmp_path, expected=EXPECTED, chunks=3) == 2


def test_first_target_is_complete_only_with_a_complete_final_chunk(tmp_path):
    for c in range(3):
        make_checkpoint(tmp_path, f'chunk-{c:04d}', c, True, (c+1)*100)
    assert first_target(tmp_path, expected=EXPECTED, chunks=3) == 4
    lone = tmp_path/'lone'
    make_checkpoint(lone, 'partial-0003-350', 3, False, 350)
    with pytest.raises(ValueError, match='beyond'):  # a count above --chunks is not "complete" without chunk-0002
        first_target(lone, expected=EXPECTED, chunks=3)


def test_first_target_cli_passes_the_run_settings(tmp_path):
    make_checkpoint(tmp_path, 'chunk-0000', 0, True, 100, seed=1)
    cli = [sys.executable, str(ROOT/'cloud/resume_target.py'), str(tmp_path), '--variant', 'ppo', '--seed', '0',
           '--envs', '8', '--chunks', '3']
    done = subprocess.run(cli, capture_output=True, text=True)
    assert done.returncode != 0 and 'seed' in done.stderr
    ok = subprocess.run(cli[:6]+['1']+cli[7:], capture_output=True, text=True)
    assert ok.returncode == 0 and ok.stdout.strip() == '2', ok.stderr


def test_runner_refuses_a_foreign_checkpoint_and_publishes_nothing(harness):
    run, out = harness
    for c in range(3):
        make_checkpoint(out, f'chunk-{c:04d}', c, True, (c+1)*100, seed=7)  # another run's finished output
    (out/'curve.jsonl').write_text('{"chunk": 2}\n')
    done, targets, uploads, status = run()
    assert done.returncode != 0 and 'seed' in done.stderr
    assert targets == [] and status == {'state': 'failed', 'exit_code': done.returncode}
    assert [dst for _, dst in uploads] == ['s3://bucket/prefix/status.json']  # no checkpoint, no foreign curve


def test_runner_does_not_report_complete_without_a_complete_final_chunk(harness):
    run, out = harness
    make_checkpoint(out, 'partial-0002-250', 2, False, 250)  # beyond SHIPNAV_CHUNKS=2, chunk-0001 never finished
    done, targets, uploads, status = run(SHIPNAV_CHUNKS='2')
    assert done.returncode != 0 and targets == [] and status['state'] == 'failed'
    assert not any('/checkpoints/' in dst for _, dst in uploads)


def test_start_document_gives_each_run_its_own_output_directory():
    def output(settings):
        unit = shlex.split(next(line for line in commands('start', settings)['commands'] if 'systemd-run' in line))
        return next(w.split('=', 2)[2] for w in unit if w.startswith('--setenv=SHIPNAV_OUTPUT='))
    first, second = output(SETTINGS), output({**SETTINGS, 'seed': 1})
    assert first == '/opt/shipnav/out/ppo/seed0' and second == '/opt/shipnav/out/ppo/seed1'
    assert output({**SETTINGS, 'variant': 'ppo_lagrangian', 'budget': 30.}) == '/opt/shipnav/out/ppo_lagrangian/seed0'
    with pytest.raises(ValueError):
        commands('start', {**SETTINGS, 'variant': '../x'})
