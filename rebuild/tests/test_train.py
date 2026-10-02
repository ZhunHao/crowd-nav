import json

import pytest

from shipnav.maps import SeaMap
from shipnav.scenarios import make_scenario
from shipnav.training import train
from shipnav.training.checkpoints import verify_checkpoint
from shipnav.training.scenarios import write_corpus


@pytest.fixture
def corpus(tmp_path):
    sea = SeaMap((0, 0, 24, 24), ((9, 9, 12, 12),))
    scenes = [{**make_scenario(sea, (2, 2), (20, 20), 1, s), 'split': 'train'} for s in (1, 2)]
    write_corpus(tmp_path/'corpus', {'train': scenes, 'select': scenes[:1]}, [])
    return tmp_path/'corpus'


def args(corpus, output, *extra):
    return ['--scenarios', str(corpus), '--output', str(output), '--seed', '0', '--chunk-steps', '128',
            '--envs', '1', '--max-seconds', '600', '--no-filter', *extra]


def test_chunks_checkpoint_and_resume(corpus, tmp_path):
    out = tmp_path/'run'
    assert train.main(args(corpus, out, '--variant', 'ppo', '--chunks', '2'))['state'] == 'complete'
    assert verify_checkpoint(out/'chunk-0001')['num_timesteps'] == 256
    status = train.main(args(corpus, out, '--variant', 'ppo', '--chunks', '3', '--resume'))
    manifest = verify_checkpoint(out/'chunk-0002')
    assert status['state'] == 'complete' and manifest['num_timesteps'] == 384
    assert manifest['resume_mode'] == 'restart-equivalent' and manifest['observation_schema'] == 1
    assert len((out/'curve.jsonl').read_text().splitlines()) == 3


def test_lagrangian_records_penalty_and_requires_budget(corpus, tmp_path):
    with pytest.raises(SystemExit):
        train.main(args(corpus, tmp_path/'a', '--variant', 'ppo_lagrangian', '--chunks', '1'))
    train.main(args(corpus, tmp_path/'b', '--variant', 'ppo_lagrangian', '--budget', '0', '--chunks', '1'))
    manifest = verify_checkpoint(tmp_path/'b'/'chunk-0000')
    assert manifest['penalty_used'] == 0. and manifest['penalty_next'] >= 0.


def test_stop_request_saves_a_partial_checkpoint(corpus, tmp_path):
    train.STOP.set()
    try:
        status = train.main(args(corpus, tmp_path/'c', '--variant', 'ppo', '--chunks', '2'))
    finally:
        train.STOP.clear()
    assert status['state'] == 'stopped'
    assert json.loads((tmp_path/'c'/'status.json').read_text())['state'] == 'stopped'


def test_resume_rejects_a_changed_configuration(corpus, tmp_path):
    out = tmp_path/'d'
    train.main(args(corpus, out, '--variant', 'ppo', '--chunks', '1'))
    with pytest.raises(SystemExit):
        train.main(args(corpus, out, '--variant', 'ppo_lagrangian', '--budget', '1', '--chunks', '2', '--resume'))


def test_worker_factory_ignores_sigterm_only_in_a_subprocess(corpus, monkeypatch):
    import signal
    from shipnav.training.scenarios import load_split
    handler = lambda *_: None
    previous = signal.signal(signal.SIGTERM, handler)
    try:
        train.make_env(load_split(corpus, 'train'), train.EnvConfig(filtered=False)).close()
        assert signal.getsignal(signal.SIGTERM) is handler  # the trainer process keeps its handler
        monkeypatch.setattr(train, 'parent_process', lambda: object())
        train.make_env(load_split(corpus, 'train'), train.EnvConfig(filtered=False)).close()
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGTERM, previous)


ROLLOUT = 2048


@pytest.fixture
def stop_after(monkeypatch):
    """Set STOP from inside the callback once more than `steps` env steps were taken (no timing)."""
    original = train.EpisodeStats._on_step

    def install(steps):
        def hooked(self):
            if self.num_timesteps > steps:
                train.STOP.set()
            return original(self)
        monkeypatch.setattr(train.EpisodeStats, '_on_step', hooked)
    yield install
    train.STOP.clear()


def long_args(corpus, output, *extra):
    return args(corpus, output, *extra, '--chunk-steps', str(2*ROLLOUT))  # argparse keeps the last occurrence


def test_mid_chunk_stop_saves_rollout_aligned_partial_then_resume_hits_the_boundary(corpus, tmp_path, stop_after, monkeypatch):
    out = tmp_path/'e'
    stop_after(ROLLOUT+5)
    status = train.main(long_args(corpus, out, '--variant', 'ppo', '--chunks', '1'))
    assert status['state'] == 'stopped' and status['num_timesteps'] == ROLLOUT
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == [f'partial-0000-{ROLLOUT}']
    partial = verify_checkpoint(out/f'partial-0000-{ROLLOUT}')
    assert partial['complete_chunk'] is False and partial['chunk'] == 0 and partial['num_timesteps'] == ROLLOUT
    train.STOP.clear()
    monkeypatch.undo()
    status = train.main(long_args(corpus, out, '--variant', 'ppo', '--chunks', '1', '--resume'))
    manifest = verify_checkpoint(out/'chunk-0000')
    assert status['state'] == 'complete' and manifest['num_timesteps'] == 2*ROLLOUT and manifest['complete_chunk'] is True
    rows = [json.loads(line) for line in (out/'curve.jsonl').read_text().splitlines()]
    assert [(r['chunk'], r['complete_chunk'], r['num_timesteps']) for r in rows] == [(0, False, ROLLOUT), (0, True, 2*ROLLOUT)]


def test_zero_step_stop_is_a_partial_and_resumes_to_a_complete_chunk(corpus, tmp_path):
    out = tmp_path/'f'
    train.STOP.set()
    try:
        assert train.main(args(corpus, out, '--variant', 'ppo', '--chunks', '1'))['state'] == 'stopped'
    finally:
        train.STOP.clear()
    assert (out/'partial-0000-0').is_dir() and not (out/'chunk-0000').exists()
    status = train.main(args(corpus, out, '--variant', 'ppo', '--chunks', '1', '--resume'))
    assert status['state'] == 'complete' and verify_checkpoint(out/'chunk-0000')['num_timesteps'] == 128


def test_repeated_zero_progress_stop_is_idempotent(corpus, tmp_path):
    out = tmp_path/'g'
    train.STOP.set()
    try:
        for extra in ((), ('--resume',)):
            assert train.main(args(corpus, out, '--variant', 'ppo', '--chunks', '1', *extra))['state'] == 'stopped'
    finally:
        train.STOP.clear()
    assert len((out/'curve.jsonl').read_text().splitlines()) == 2


def test_lagrangian_partial_keeps_the_multiplier(corpus, tmp_path, stop_after):
    out = tmp_path/'h'
    train.main(long_args(corpus, out, '--variant', 'ppo_lagrangian', '--budget', '0', '--chunks', '1'))
    carried = verify_checkpoint(out/'chunk-0000')['penalty_next']
    stop_after(3*ROLLOUT+5)
    status = train.main(long_args(corpus, out, '--variant', 'ppo_lagrangian', '--budget', '0', '--chunks', '2', '--resume'))
    partial = verify_checkpoint(out/f'partial-0001-{3*ROLLOUT}')
    assert status['state'] == 'stopped' and partial['complete_chunk'] is False
    assert partial['penalty_used'] == carried and partial['penalty_next'] == partial['penalty_used']

