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
