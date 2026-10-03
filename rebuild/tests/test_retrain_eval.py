from hashlib import sha256
from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from retrain_eval import hierarchical, learned_variants, seed_matched  # noqa: E402
import freeze_retraining_protocol as freezer  # noqa: E402


def test_seed_matched_replicates_a_fixed_comparator():
    fixed = {'s1': 1., 's2': 0.}
    assert seed_matched(fixed, [0, 1]) == {0: fixed, 1: fixed}


def test_learned_variants_cover_every_seed_variant_and_filter_state():
    selections = {('ppo', 0): 'a', ('ppo', 1): 'b', ('ppo_lagrangian', 0): 'c', ('ppo_lagrangian', 1): 'd'}
    variants = learned_variants(selections)
    assert variants['ppo_s0_filtered'] == {'policy_name': 'ppo', 'learned_dir': 'a',
                                           'dynamics': 'marine', 'filtered': True}
    assert len(variants) == 8 and not variants['ppo_lagrangian_s1_unfiltered']['filtered']
    with pytest.raises(ValueError):
        learned_variants({('ppo', 0): 'a'})


# ---- hierarchical(): input shapes accepted by metrics.hierarchical_interval ----

def result_rows(seeds, scenarios=('h1', 'h2', 'h3')):
    """Complete rows for every arm of the comparison table; status depends on arm and scenario."""
    names = ['marine_mpc', 'marine_mpc_unfiltered', 'marine_sarl']
    names += [f'{v}_s{s}_{f}' for v in ('ppo', 'ppo_lagrangian') for s in seeds for f in ('filtered', 'unfiltered')]
    rows = []
    for n, name in enumerate(names):
        for k, hash_ in enumerate(scenarios):
            rows.append({'variant': name, 'scenario_hash': hash_, 'status': 'success' if (n+k) % 2 else 'timeout',
                         'ship_collision': (n+k) % 3 == 0, 'land_collision': False})
    return rows


def protocol_comparisons():
    return {'training': {'seeds': [0, 1, 2]}, 'comparisons': freezer.COMPARISONS}


def test_hierarchical_accepts_seed_by_scenario_tables_for_every_comparison():
    result = hierarchical(result_rows([0, 1, 2]), protocol_comparisons())
    labels = {label for _, _, label in freezer.COMPARISONS}
    assert set(result) == {f'{label}/{field}' for label in labels for field in ('success', 'any_collision')}
    assert all(r['training_seeds'] == 3 for r in result.values())


def test_hierarchical_difference_of_two_fixed_comparators_has_no_seed_spread():
    rows = result_rows([0, 1, 2])
    only = hierarchical(rows, {'training': {'seeds': [0, 1, 2]},
                               'comparisons': [['marine_mpc', 'marine_mpc_unfiltered', 'MPC_filter']]})
    expected = (sum(r['status'] == 'success' for r in rows if r['variant'] == 'marine_mpc') -
                sum(r['status'] == 'success' for r in rows if r['variant'] == 'marine_mpc_unfiltered'))/3
    assert only['MPC_filter/success']['difference'] == pytest.approx(expected)


def test_hierarchical_refuses_an_incomplete_arm():
    rows = [r for r in result_rows([0, 1, 2]) if not (r['variant'] == 'ppo_s1_filtered' and r['scenario_hash'] == 'h2')]
    with pytest.raises(ValueError):
        hierarchical(rows, protocol_comparisons())


# ---- freeze_retraining_protocol: pure parts and refusal paths ----

SEEDS = [0, 1, 2]
FILES = ('policy.zip', 'state.pt', 'rng.json')


COMMIT = 'c0ffee'*6+'c0ff'


def make_checkpoint(root, variant, seed, corrupt=False, name='chunk-0004', tag='', over=None):
    """A verifiable checkpoint of the run `variant`/`seed` whose manifest agrees with the `training` fixture."""
    run = root/'results/train'/variant/f'seed{seed}'
    directory = run/name
    directory.mkdir(parents=True)
    files = {}
    for file in FILES:
        payload = f'{tag or variant+str(seed)}{file}'.encode()
        (directory/file).write_bytes(payload)
        files[file] = sha256(payload).hexdigest()
    manifest = {'files': files, 'num_timesteps': 5, 'variant': variant, 'seed': seed, 'envs': 8, 'chunk_steps': 20480,
                'env_config': {'filtered': True, 'observation': {}, 'penalty': 0.},
                'scenario_manifest_sha256': 'train-sha', 'chunk': 4, 'complete_chunk': True,
                'budget': 12.5 if variant == 'ppo_lagrangian' else None, 'source_commit': COMMIT}
    (directory/'manifest.json').write_text(json.dumps({**manifest, **(over or {})}))
    if corrupt:
        (directory/'policy.zip').write_bytes(b'tampered')
    return run, directory, files


def write_selection(run, directory, files, path=None):
    (run/'selection.json').write_text(json.dumps({'chunk': 4, 'path': path or str(directory), 'files': files,
                                                  'success_rate': .8, 'collision_rate': 0., 'mean_domain_s': 1.,
                                                  'candidates': [], 'skipped_incomplete': []}))


@pytest.fixture
def training():
    return {'schema': 1, 'seeds': SEEDS, 'variants': ['ppo', 'ppo_lagrangian'], 'budget_domain_s': 12.5,
            'chunks': 5, 'chunk_steps': 20480, 'envs': 8, 'filtered': True, 'observation': {},
            'corpus_sha256': {'train': 'train-sha'}, 'source_commit': COMMIT}


@pytest.fixture
def trained(tmp_path):
    for variant in freezer.LEARNED:
        for seed in SEEDS:
            run, directory, files = make_checkpoint(tmp_path, variant, seed)
            # Task 11 records the path as it was given on the command line: absolute here, relative below.
            write_selection(run, directory, files, str(directory) if seed else f'results/train/{variant}/seed0/chunk-0004')
    return tmp_path


def test_selections_are_loaded_with_paths_relative_to_rebuild(trained, training):
    selections = freezer.load_selections(trained, training)
    assert set(selections) == {(v, s) for v in freezer.LEARNED for s in SEEDS}
    assert all(sel['path'] == f'results/train/{v}/seed{s}/chunk-0004' for (v, s), sel in selections.items())
    assert selections['ppo', 1]['files']['policy.zip'] == sha256(b'ppo1policy.zip').hexdigest()


def test_a_checkpoint_outside_rebuild_keeps_its_absolute_path(trained, training, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp('elsewhere')
    run, directory, files = make_checkpoint(elsewhere, 'ppo', 0)
    write_selection(trained/'results/train/ppo/seed0', directory, files)
    assert freezer.load_selections(trained, training)['ppo', 0]['path'] == str(directory)


@pytest.mark.parametrize('variant, seed', [('ppo', 0), ('ppo_lagrangian', 2)])
def test_freezer_refuses_a_missing_selection(trained, training, variant, seed):
    (trained/'results/train'/variant/f'seed{seed}'/'selection.json').unlink()
    with pytest.raises(ValueError, match=f'Missing selection.json for {variant}/seed{seed}'):
        freezer.load_selections(trained, training)


def test_freezer_refuses_a_selected_checkpoint_failing_verification(trained, training):
    (trained/'results/train/ppo/seed1/chunk-0004/policy.zip').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='ppo/seed1 failed verification'):
        freezer.load_selections(trained, training)


def test_freezer_refuses_an_incomplete_selected_checkpoint(trained, training):
    (trained/'results/train/ppo_lagrangian/seed0/chunk-0004/rng.json').unlink()
    with pytest.raises(ValueError, match='ppo_lagrangian/seed0 failed verification'):
        freezer.load_selections(trained, training)


def test_freezer_refuses_a_selection_whose_hashes_differ_from_the_checkpoint(trained, training):
    run = trained/'results/train/ppo/seed2'
    selection = json.loads((run/'selection.json').read_text())
    selection['files'] = {**selection['files'], 'state.pt': 'other'}
    (run/'selection.json').write_text(json.dumps(selection))
    with pytest.raises(ValueError, match='differs from its selection.json'):
        freezer.load_selections(trained, training)


def baseline_protocol(entries):
    return {'manifest': freezer.MANIFEST, 'manifest_sha256': 'manifest-hash', 'original_manifest': 'scenarios/splits.json',
            'original_manifest_sha256': 'original-hash', 'identities': {'model': {'rl_model.pth': 'm'}},
            'shared_limits_model': {e['scenario_hash']: 100 for e in entries}, 'inference': {'method': 'paired'}}


def build(trained, configured, **override):
    entries = [{'scenario_hash': 'h1'}, {'scenario_hash': 'h2'}]
    selections = freezer.load_selections(trained, configured)
    arguments = dict(baseline=baseline_protocol(entries), manifest={'splits': {'test': entries}},
                     manifest_sha256='manifest-hash', training=configured, selections=selections,
                     source_sha256={'pyproject.toml': 'p'}, lock_sha256='lock', model_hashes={'rl_model.pth': 'm'},
                     source_commit='abc', frozen_utc='2026-10-01T00:00:00+00:00')
    return freezer.build_protocol(**{**arguments, **override}), selections


def test_protocol_covers_every_arm_with_per_arm_identities(trained, training):
    protocol, selections = build(trained, training)
    assert len(protocol['variants']) == 3+2*3*2
    assert protocol['variants']['ppo_s1_unfiltered'] == {
        'policy_name': 'ppo', 'learned_dir': 'results/train/ppo/seed1/chunk-0004', 'dynamics': 'marine', 'filtered': False}
    learned = protocol['identities']['learned']
    assert set(learned) == {n for n, c in protocol['variants'].items() if c['policy_name'] == 'ppo'}
    assert learned['ppo_lagrangian_s2_filtered'] == learned['ppo_lagrangian_s2_unfiltered'] == \
        selections['ppo_lagrangian', 2]['files']
    assert protocol['targets'] == {'success_rate_minimum': .8, 'collision_rate_maximum': .05,
                                   'constraint_cost_budget': 12.5}
    assert protocol['comparisons'] == freezer.COMPARISONS and protocol['training']['seeds'] == SEEDS
    assert protocol['shared_limits_model'] == {'h1': 100, 'h2': 100}
    assert protocol['inference']['method'] == 'paired' and 'hierarchical_interval' in protocol['inference']['learned_arms']
    assert protocol['multiplicity'] == 'exploratory, unadjusted'
    json.dumps(protocol, allow_nan=False)


def test_protocol_refuses_an_incomplete_learned_variant_table(trained, training):
    selections = freezer.load_selections(trained, training)
    del selections['ppo_lagrangian', 1]
    with pytest.raises(ValueError, match='every seed'):
        build(trained, training, selections=selections)


def test_protocol_refuses_selections_for_other_seeds_or_too_few_seeds(trained, training):
    selections = freezer.load_selections(trained, training)
    with pytest.raises(ValueError, match='seeds'):
        build(trained, training, training={**training, 'seeds': [0, 1, 2, 3]})
    two = {**training, 'seeds': [0, 1]}
    selections = {k: v for k, v in selections.items() if k[1] < 2}
    with pytest.raises(ValueError, match='three'):
        build(trained, training, training=two, selections=selections)


def test_protocol_refuses_inconsistent_baseline_inputs(trained, training):
    with pytest.raises(ValueError, match='manifest'):
        build(trained, training, manifest_sha256='other')
    with pytest.raises(ValueError, match='SARL'):
        build(trained, training, model_hashes={'rl_model.pth': 'changed'})
    with pytest.raises(ValueError, match='budget'):
        build(trained, training, training={**training, 'budget_domain_s': float('nan')})


def test_main_explains_a_missing_training_configuration_before_touching_anything(tmp_path, monkeypatch):
    monkeypatch.setattr(freezer, 'OUTPUT', tmp_path/'protocol.json')
    monkeypatch.setattr(freezer, 'TRAINING_CONFIG', tmp_path/'training_config.json')
    monkeypatch.setattr(freezer, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['freeze_retraining_protocol.py'])
    with pytest.raises(SystemExit, match='training_config.json is missing'):
        freezer.main()
    assert not (tmp_path/'protocol.json').exists()


def test_main_never_overwrites_a_frozen_protocol(tmp_path, monkeypatch):
    (tmp_path/'protocol.json').write_text('{}')
    monkeypatch.setattr(freezer, 'OUTPUT', tmp_path/'protocol.json')
    monkeypatch.setattr(freezer, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['freeze_retraining_protocol.py'])
    with pytest.raises(SystemExit, match='already exists'):
        freezer.main()
    assert (tmp_path/'protocol.json').read_text() == '{}'


# ---- the selected checkpoints must belong to the frozen training configuration ----

def reselect(root, variant, seed, over):
    """Replace one run's checkpoint with one whose manifest deviates, and select it."""
    run = root/'results/train'/variant/f'seed{seed}'
    import shutil
    shutil.rmtree(run)
    run, directory, files = make_checkpoint(root, variant, seed, over=over)
    write_selection(run, directory, files, f'results/train/{variant}/seed{seed}/chunk-0004')


DEVIATIONS = [
    ('variant', {'variant': 'ppo_lagrangian'}), ('seed', {'seed': 2}),
    ('env_config.filtered', {'env_config': {'filtered': False, 'observation': {}}}),
    ('env_config.observation', {'env_config': {'filtered': True, 'observation': {'noise': 1}}}),
    ('chunk_steps', {'chunk_steps': 2048}), ('envs', {'envs': 4}),
    ('scenario_manifest_sha256', {'scenario_manifest_sha256': 'other'}),
    ('budget', {'budget': 3.}), ('complete_chunk', {'complete_chunk': False}),
    ('source_commit', {'source_commit': 'deadbeef'}),
]


@pytest.mark.parametrize('field, over', DEVIATIONS, ids=[d[0] for d in DEVIATIONS])
def test_freezer_refuses_a_checkpoint_that_deviates_from_the_training_configuration(trained, training, field, over):
    reselect(trained, 'ppo', 1, over)
    with pytest.raises(ValueError, match=f'frozen training configuration.*{field}'):
        freezer.load_selections(trained, training)


def test_freezer_refuses_a_lagrangian_run_without_the_frozen_budget(trained, training):
    reselect(trained, 'ppo_lagrangian', 0, {'budget': None})
    with pytest.raises(ValueError, match='budget'):
        freezer.load_selections(trained, training)


def test_freezer_refuses_a_run_that_did_not_reach_its_final_chunk(trained, training):
    import shutil
    shutil.rmtree(trained/'results/train/ppo/seed2')
    run, directory, files = make_checkpoint(trained, 'ppo', 2, name='chunk-0002', over={'chunk': 2})
    write_selection(run, directory, files, 'results/train/ppo/seed2/chunk-0002')
    with pytest.raises(ValueError, match='final chunk chunk-0004'):
        freezer.load_selections(trained, training)


def test_freezer_refuses_the_same_checkpoint_selected_twice(trained, training):
    reselect_dir = trained/'results/train/ppo/seed1'
    import shutil
    shutil.rmtree(reselect_dir)
    run, directory, files = make_checkpoint(trained, 'ppo', 1, tag='ppo0')  # same bytes as ppo/seed0
    write_selection(run, directory, files, 'results/train/ppo/seed1/chunk-0004')
    with pytest.raises(ValueError, match='same checkpoint'):
        freezer.load_selections(trained, training)


def git(repo, *args):
    import subprocess
    return subprocess.run(['git', '-C', str(repo), '-c', 'user.name=t', '-c', 'user.email=t@t', *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def test_source_must_be_unchanged_since_the_training_commit(tmp_path):
    from training_identity import check_source_unchanged
    git(tmp_path, 'init', '-q')
    (tmp_path/'src').mkdir()
    (tmp_path/'src/a.py').write_text('x = 1\n')
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-qm', 'trained')
    commit = git(tmp_path, 'rev-parse', 'HEAD')
    (tmp_path/'notes.txt').write_text('docs and evidence may change')
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-qm', 'evidence only')
    check_source_unchanged(commit, tmp_path)
    (tmp_path/'src/a.py').write_text('x = 2\n')
    git(tmp_path, 'commit', '-qam', 'edit src')
    with pytest.raises(ValueError, match='src/ differs'):
        check_source_unchanged(commit, tmp_path)
    with pytest.raises(ValueError, match='src/ differs'):
        check_source_unchanged('0'*40, tmp_path)  # an unknown commit is refused too


def test_training_config_freezer_requires_a_clean_tree_over_every_frozen_input(tmp_path):
    from training_identity import dirty_paths
    import freeze_training_config as config_freezer
    assert set(config_freezer.FROZEN_INPUTS) == {'src', 'tools', 'cloud', 'scenarios/training', 'pyproject.toml',
                                                 'uv.lock'}
    git(tmp_path, 'init', '-q')
    for name in ('src/a.py', 'tools/b.py', 'cloud/c.sh', 'scenarios/training/m.json', 'pyproject.toml', 'uv.lock'):
        (tmp_path/name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path/name).write_text('1')
    (tmp_path/'results').mkdir()
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-qm', 'base')
    assert dirty_paths(tmp_path, config_freezer.FROZEN_INPUTS) == []
    (tmp_path/'results/ignored.txt').write_text('outside the frozen inputs')
    assert dirty_paths(tmp_path, config_freezer.FROZEN_INPUTS) == []
    for name in ('tools/b.py', 'cloud/c.sh', 'uv.lock'):  # tracked edits
        (tmp_path/name).write_text('2')
    (tmp_path/'tools/new.py').write_text('untracked')
    dirty = ' '.join(dirty_paths(tmp_path, config_freezer.FROZEN_INPUTS))
    assert all(name in dirty for name in ('tools/b.py', 'cloud/c.sh', 'uv.lock', 'tools/new.py'))
