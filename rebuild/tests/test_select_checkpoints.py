from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from select_checkpoints import choose, evaluate, load_manifests, pick_checkpoints  # noqa: E402
from shipnav.training.checkpoints import IncompleteCheckpointError  # noqa: E402


def row(chunk, success, collision, domain):
    return {'chunk': chunk, 'success_rate': success, 'collision_rate': collision, 'mean_domain_s': domain}


def test_prefers_success_among_safe_candidates_then_lower_domain_then_earlier():
    rows = [row(4, .9, .10, 1.), row(9, .8, .04, 2.), row(14, .8, .04, 1.), row(19, .8, .04, 1.)]
    assert choose(rows, .05)['chunk'] == 14


def test_falls_back_to_lowest_collision_when_none_are_safe():
    assert choose([row(4, .9, .2, 0.), row(9, .5, .1, 0.)], .05)['chunk'] == 9


def test_fallback_ties_prefer_success_then_domain_then_earlier():
    rows = [row(9, .5, .1, 1.), row(14, .7, .1, 5.), row(19, .7, .1, 2.), row(24, .7, .1, 2.)]
    assert choose(rows, .05)['chunk'] == 19


def test_choose_rejects_no_candidates():
    with pytest.raises(ValueError, match='No candidate'):
        choose([], .05)


def complete(chunk):
    return {'chunk': chunk, 'complete_chunk': True, 'num_timesteps': (chunk+1)*100}


def partial(chunk, steps):
    return {'chunk': chunk, 'complete_chunk': False, 'num_timesteps': steps}


def names(paths):
    return [Path(p).name for p in paths]


def test_every_chunks_picks_boundaries_and_include_last_adds_final_chunk():
    manifests = {Path(f'chunk-{c:04d}'): complete(c) for c in range(12)}
    assert names(pick_checkpoints(manifests, 5, True)) == ['chunk-0004', 'chunk-0009', 'chunk-0011']
    assert names(pick_checkpoints(manifests, 5, False)) == ['chunk-0004', 'chunk-0009']


def test_include_last_does_not_duplicate_a_boundary_chunk():
    manifests = {Path(f'chunk-{c:04d}'): complete(c) for c in range(10)}
    assert names(pick_checkpoints(manifests, 5, True)) == ['chunk-0004', 'chunk-0009']


def test_partial_checkpoints_are_never_picked_or_taken_as_last():
    manifests = {Path('chunk-0000'): complete(0), Path('chunk-0001'): complete(1),
                 Path('partial-0002-250'): partial(2, 250), Path('partial-0004-450'): partial(4, 450)}
    assert names(pick_checkpoints(manifests, 5, True)) == ['chunk-0001']
    assert names(pick_checkpoints(manifests, 1, False)) == ['chunk-0000', 'chunk-0001']
    assert pick_checkpoints({Path('partial-0004-450'): partial(4, 450)}, 5, True) == []


def test_ordering_follows_manifest_chunk_not_folder_name():
    manifests = {Path('a-late'): complete(9), Path('z-early'): complete(4), Path('m-last'): complete(11)}
    assert names(pick_checkpoints(manifests, 5, True)) == ['z-early', 'a-late', 'm-last']
    assert names(pick_checkpoints({Path('a-late'): complete(11), Path('z-early'): complete(3)}, 100, True)) == ['a-late']


def test_every_chunks_must_be_positive():
    with pytest.raises(ValueError, match='every_chunks'):
        pick_checkpoints({Path('chunk-0000'): complete(0)}, 0, True)


def test_load_manifests_skips_incomplete_folders_and_lists_them():
    def verify(path):
        if path.name == 'chunk-0001':
            raise IncompleteCheckpointError('half downloaded')
        return complete(int(path.name[-4:]))

    manifests, skipped = load_manifests([Path('chunk-0000'), Path('chunk-0001'), Path('chunk-0002')], verify)
    assert names(manifests) == ['chunk-0000', 'chunk-0002']
    assert skipped == ['chunk-0001']


def test_load_manifests_lets_hash_mismatch_propagate():
    def verify(path):
        raise ValueError('Checkpoint hash mismatch: policy.zip')

    with pytest.raises(ValueError, match='hash mismatch'):
        load_manifests([Path('chunk-0000')], verify)


def test_evaluate_refuses_runs_without_domain_time_data(monkeypatch):
    import select_checkpoints
    monkeypatch.setattr(select_checkpoints, 'execute', lambda *a, **k: {'status': 'success'})
    monkeypatch.setattr(select_checkpoints, 'metrics', lambda run: {'domain_time_s': None})
    scenario = {'map': {}, 'start': (0, 0), 'goal': (1, 1)}
    monkeypatch.setattr(select_checkpoints, 'map_options', lambda m: {})
    with pytest.raises(ValueError, match='domain_time_s'):
        evaluate(Path('chunk-0000'), [scenario], True)


def test_evaluate_aggregates_rates_and_rejects_empty_split(monkeypatch):
    import select_checkpoints
    outcomes = iter([('success', 2.), ('collision', 4.), ('success', 0.), ('timeout', 2.)])
    monkeypatch.setattr(select_checkpoints, 'execute', lambda *a, **k: {'status': next(outcomes)[0]})
    monkeypatch.setattr(select_checkpoints, 'map_options', lambda m: {})
    domains = iter([2., 4., 0., 2.])
    monkeypatch.setattr(select_checkpoints, 'metrics', lambda run: {'domain_time_s': next(domains)})
    scenario = {'map': {}, 'start': (0, 0), 'goal': (1, 1)}
    result = evaluate(Path('chunk-0000'), [scenario]*4, True)
    assert result == {'success_rate': .5, 'collision_rate': .25, 'mean_domain_s': 2.}
    with pytest.raises(ValueError, match='empty'):
        evaluate(Path('chunk-0000'), [], True)


# ---- select(): identity, final chunk and portable paths ----

from hashlib import sha256  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import select_checkpoints  # noqa: E402

COMMIT = 'abc123'
CONFIG = {'chunks': 3, 'chunk_steps': 20480, 'envs': 8, 'filtered': True, 'observation': {}, 'budget_domain_s': 12.5,
          'corpus_sha256': {'train': 'train-sha'}, 'source_commit': COMMIT,
          'selection': {'split': 'select', 'every_chunks': 1, 'include_last': True, 'collision_max': .05}}


def write_checkpoint(run, chunk, **over):
    directory = Path(run)/f'chunk-{chunk:04d}'
    directory.mkdir(parents=True)
    files = {}
    for name in ('policy.zip', 'state.pt', 'rng.json'):
        data = f'{directory}{name}'.encode()
        (directory/name).write_bytes(data)
        files[name] = sha256(data).hexdigest()
    manifest = {'files': files, 'num_timesteps': (chunk+1)*10, 'variant': 'ppo', 'seed': 0, 'envs': 8,
                'chunk_steps': 20480, 'env_config': {'filtered': True, 'observation': {}},
                'scenario_manifest_sha256': 'train-sha', 'chunk': chunk, 'complete_chunk': True, 'budget': None,
                'source_commit': COMMIT, **over}
    (directory/'manifest.json').write_text(json.dumps(manifest))
    return directory


@pytest.fixture
def run(tmp_path):
    run = tmp_path/'results/train/ppo/seed0'
    for chunk in range(3):
        write_checkpoint(run, chunk)
    return run


def selected(run, root, **kwargs):
    return select_checkpoints.select(run, CONFIG, root=root, load=lambda *_: [{}],
                                     assess=lambda d, *_: {'success_rate': int(d.name[-4:])/10, 'collision_rate': 0., 'mean_domain_s': 1.},
                                     **kwargs)


def test_select_stores_the_path_relative_to_rebuild_whatever_the_cwd(run, tmp_path, monkeypatch):
    monkeypatch.chdir(run)  # a caller's cwd must not leak into the stored path
    best = selected(run, tmp_path)
    assert best['path'] == 'results/train/ppo/seed0/chunk-0002' and best['chunk'] == 2
    stored = json.loads((run/'selection.json').read_text())
    assert [c['path'] for c in stored['candidates']] == [f'results/train/ppo/seed0/chunk-000{c}' for c in range(3)]
    monkeypatch.chdir(tmp_path/'results')
    assert selected(Path('train/ppo/seed0'), tmp_path)['path'] == 'results/train/ppo/seed0/chunk-0002'


@pytest.mark.parametrize('field, over', [('seed', {'seed': 1}), ('envs', {'envs': 4}), ('chunk_steps', {'chunk_steps': 1}),
                                         ('env_config', {'env_config': {'filtered': False, 'observation': {}}}),
                                         ('scenario_manifest_sha256', {'scenario_manifest_sha256': 'x'}),
                                         ('variant', {'variant': 'ppo_lagrangian'})])
def test_select_refuses_a_run_directory_with_mixed_identity(run, tmp_path, field, over):
    write_checkpoint(run, 3, **over)
    with pytest.raises(ValueError, match=f'mixes.*{field}'):
        selected(run, tmp_path)
    assert not (run/'selection.json').exists()


def test_select_refuses_a_run_that_does_not_match_the_training_configuration(tmp_path):
    other = tmp_path/'results/train/ppo/seed0'
    for chunk in range(3):
        write_checkpoint(other, chunk, scenario_manifest_sha256='another-corpus')
    with pytest.raises(ValueError, match='frozen training configuration.*scenario_manifest_sha256'):
        selected(other, tmp_path)


def test_select_refuses_a_run_that_has_not_reached_the_final_chunk(tmp_path):
    short = tmp_path/'results/train/ppo/seed0'
    for chunk in range(2):
        write_checkpoint(short, chunk)
    with pytest.raises(ValueError, match='final chunk chunk-0002'):
        selected(short, tmp_path)


def test_select_checks_the_run_against_the_slot_it_sits_in(tmp_path):
    wrong = tmp_path/'results/train/ppo/seed4'  # the checkpoints say seed 0
    for chunk in range(3):
        write_checkpoint(wrong, chunk)
    with pytest.raises(ValueError, match='seed'):
        selected(wrong, tmp_path)
