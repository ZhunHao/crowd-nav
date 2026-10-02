"""Future baseline reuse rejects stale identities and incomplete/mixed evidence."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import baseline_run as baseline
from baseline_evidence import archive
from shipnav.scenarios import scenario_hash


@pytest.fixture
def study(tmp_path, monkeypatch):
    root = tmp_path/'rebuild'
    root.mkdir()
    model = tmp_path/'model'
    model.mkdir()
    files = {'source.py': b'source', 'uv.lock': b'lock', 'splits.json': b'manifest',
             'original.json': b'original'}
    for name, raw in files.items():
        (root/name).write_bytes(raw)
    (model/'rl_model.pth').write_bytes(b'model')
    variants = {'a': {'policy_name': 'sarl'}, 'b': {'policy_name': 'orca'}}
    monkeypatch.setattr(baseline, 'ROOT', root)
    monkeypatch.setattr(baseline, 'VARIANTS', variants)
    scenario = {'schema': 1, 'family': 'fixture', 'seed': 7, 'map': {}, 'traffic': []}
    entry = dict(id='fixture', family='fixture', seed=7, traffic_count=0,
                 scenario_hash=scenario_hash(scenario))
    data = dict(source_sha256={n: sha256(files[n]).hexdigest() for n in ('source.py', 'uv.lock')},
                manifest='splits.json', manifest_sha256=sha256(files['splits.json']).hexdigest(),
                original_manifest='original.json', original_manifest_sha256=sha256(files['original.json']).hexdigest(),
                variants=variants, scenarios={'test': [entry]}, shared_limits_model={entry['scenario_hash']: 100},
                identities={'lock_sha256': sha256(b'lock').hexdigest(),
                            'model': {'rl_model.pth': sha256(b'model').hexdigest()}})
    identity = {'sha256': 'protocol-one', 'commit': 'frozen-commit'}
    output = tmp_path/'output'
    output.mkdir()
    return root, model, scenario, entry, data, identity, output


def retained_row(study, name='a'):
    root, model, scenario, entry, data, identity, output = study
    run = dict(status='success', scenario_hash=entry['scenario_hash'], variant=name,
               variant_config=data['variants'][name], settings={'limit': 100},
               model_hashes=baseline.expected_model_hashes(data, name, data['variants']),
               provenance={'uv_lock_sha256': data['identities']['lock_sha256']})
    def retain(filename, value):
        path = output/filename
        path.write_text(json.dumps(value))
        return archive(path)
    trace = retain(name+'.json', run)
    row = dict(scenario_hash=entry['scenario_hash'], scenario_id=entry['id'], family='fixture', seed=7,
               traffic_count=0, variant=name, status='success', protocol=identity, trace_hash=trace['raw_sha256'],
               trace_file=trace['file'], shared_limit_model=100, archive=trace,
               scenario_archive=retain(name+'-scenario.json', scenario),
               original_metrics_archive=retain(name+'-metrics.json', [dict(scenario_hash=entry['scenario_hash'],
                   variant=name, status='success', trace_hash=trace['raw_sha256'])]))
    return row


@pytest.mark.parametrize('resume', [False, True])
@pytest.mark.parametrize('changed', ['source.py', 'uv.lock', 'rl_model.pth', 'splits.json', 'original.json'])
def test_launch_and_resume_reject_changed_frozen_identity(study, changed, resume):
    root, model, scenario, entry, data, identity, output = study
    if resume:
        baseline.bind_output(output, identity, [entry])
        (output/'rows.jsonl').write_text(json.dumps(retained_row(study))+'\n')
    before = (output/'rows.jsonl').read_bytes() if resume else None
    (model/changed if changed == 'rl_model.pth' else root/changed).write_bytes(b'changed')
    with pytest.raises(ValueError, match='identity'):
        baseline.execute_matrix([entry], root, model, output, identity, data)
    assert ((output/'rows.jsonl').read_bytes() if resume else None) == before
    if not resume:
        assert not (output/'rows.jsonl').exists()


def test_directory_rejects_wrong_protocol_and_unbound_existing_rows(study):
    root, model, scenario, entry, data, identity, output = study
    baseline.bind_output(output, identity, [entry])
    with pytest.raises(ValueError, match='protocol'):
        baseline.bind_output(output, {'sha256': 'other', 'commit': 'other'}, [entry])
    (output/'run-identity.json').unlink()
    (output/'rows.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='binding'):
        baseline.bind_output(output, identity, [entry])


@pytest.mark.parametrize('change', ['protocol', 'scenario', 'variant', 'duplicate', 'model', 'lock'])
def test_resume_rejects_incompatible_journal_rows(study, change):
    root, model, scenario, entry, data, identity, output = study
    row = retained_row(study)
    if change == 'protocol':
        row['protocol'] = {'sha256': 'other', 'commit': 'other'}
    elif change == 'scenario':
        row['scenario_hash'] = 'other'
    elif change == 'variant':
        row['variant'] = 'other'
    elif change in ('model', 'lock'):
        from baseline_evidence import load_verified
        run = load_verified(output, row['archive'])
        if change == 'model':
            run['model_hashes'] = {'rl_model.pth': 'other'}
        else:
            run['provenance']['uv_lock_sha256'] = 'other'
        path = output/'changed.json'
        path.write_text(json.dumps(run))
        row['archive'] = archive(path)
        row['trace_hash'] = row['archive']['raw_sha256']
        row['trace_file'] = row['archive']['file']
    rows = [row, deepcopy(row)] if change == 'duplicate' else [row]
    with pytest.raises(ValueError):
        baseline.validate_rows(rows, [entry], output, identity, data)


def test_final_audit_requires_complete_unique_grid_and_partial_resume_is_valid(study):
    root, model, scenario, entry, data, identity, output = study
    rows = [retained_row(study)]
    baseline.validate_rows(rows, [entry], output, identity, data)
    with pytest.raises(ValueError, match='complete'):
        baseline.validate_rows(rows, [entry], output, identity, data, complete=True)
    rows.append(retained_row(study, 'b'))
    baseline.validate_rows(rows, [entry], output, identity, data, complete=True)
    with pytest.raises(ValueError, match='Duplicate'):
        baseline.validate_rows(rows+[deepcopy(rows[0])], [entry], output, identity, data, complete=True)


def test_partial_matrix_resumes_only_missing_validated_pair(study, monkeypatch):
    root, model, scenario, entry, data, identity, output = study
    path = root/'scenario.json'
    path.write_text(json.dumps(scenario))
    entry.update(archive(path))
    data['scenarios']['test'] = [entry]
    baseline.bind_output(output, identity, [entry])
    first = retained_row(study)
    (output/'rows.jsonl').write_text(json.dumps(first)+'\n')
    def missing_unit(scenario, entry, name, model, output):
        assert name == 'b', 'An already validated completed arm must not rerun'
        return retained_row(study, name)
    monkeypatch.setattr(baseline, 'run_unit', missing_unit)
    monkeypatch.setattr(baseline, 'summary', lambda rows: {'rows': len(rows)})
    result = baseline.execute_matrix([entry], root, model, output, identity, data)
    assert result == {'rows': 2, 'protocol': identity}
    rows = [json.loads(line) for line in (output/'rows.jsonl').read_text().splitlines()]
    assert rows[0] == first
    assert [(row['scenario_id'], row['variant']) for row in rows] == [('fixture', 'a'), ('fixture', 'b')]


def test_expected_model_hashes_are_per_variant():
    from baseline_run import expected_model_hashes
    data = {'identities': {'model': {'rl_model.pth': 'a'}, 'learned': {'ppo_s0': {'policy.zip': 'b'}}}}
    variants = {'marine_sarl': {'policy_name': 'sarl'}, 'marine_mpc': {'policy_name': 'mpc'},
                'ppo_s0': {'policy_name': 'ppo'}}
    assert expected_model_hashes(data, 'marine_sarl', variants) == {'rl_model.pth': 'a'}
    assert expected_model_hashes(data, 'marine_mpc', variants) == {}
    assert expected_model_hashes(data, 'ppo_s0', variants) == {'policy.zip': 'b'}


def test_non_sarl_arms_must_record_no_model_hashes(study):
    root, model, scenario, entry, data, identity, output = study
    baseline.validate_rows([retained_row(study, 'b')], [entry], output, identity, data)  # ORCA records {}
    from baseline_evidence import load_verified
    row = retained_row(study, 'b')
    run = load_verified(output, row['archive'])
    run['model_hashes'] = data['identities']['model']
    path = output/'sarl-hashes-on-orca.json'
    path.write_text(json.dumps(run))
    row['archive'] = archive(path)
    row['trace_hash'], row['trace_file'] = row['archive']['raw_sha256'], row['archive']['file']
    with pytest.raises(ValueError, match='identity'):
        baseline.validate_rows([row], [entry], output, identity, data)


def learned_study(study):
    """Add a selected-checkpoint arm to the study: a real manifest-verified directory under the fake ROOT."""
    root, model, scenario, entry, data, identity, output = study
    checkpoint = root/'results/train/ppo/seed0/chunk4'
    checkpoint.mkdir(parents=True)
    files = {}
    for name in ('policy.zip', 'state.pt', 'rng.json'):
        (checkpoint/name).write_bytes(name.encode())
        files[name] = sha256(name.encode()).hexdigest()
    (checkpoint/'manifest.json').write_text(json.dumps({'files': files, 'num_timesteps': 1}))
    variants = {**data['variants'], 'ppo_s0_filtered': {'policy_name': 'ppo', 'dynamics': 'marine', 'filtered': True,
                                                       'learned_dir': 'results/train/ppo/seed0/chunk4'}}
    data = {**data, 'variants': variants, 'identities': {**data['identities'], 'learned': {'ppo_s0_filtered': files}}}
    return checkpoint, files, variants, data


def test_learned_checkpoint_identity_is_verified(study):
    root, model, *_ = study
    checkpoint, files, variants, data = learned_study(study)
    baseline.verify_frozen_identities(data, model, variants=variants)
    (checkpoint/'policy.zip').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='identity'):
        baseline.verify_frozen_identities(data, model, variants=variants)


def test_learned_identity_rejects_stored_hash_drift_and_arm_mismatch(study):
    root, model, *_ = study
    checkpoint, files, variants, data = learned_study(study)
    drifted = {**data, 'identities': {**data['identities'],
                                      'learned': {'ppo_s0_filtered': {**files, 'rng.json': 'other'}}}}
    with pytest.raises(ValueError, match='identity'):
        baseline.verify_frozen_identities(drifted, model, variants=variants)
    missing = {**data, 'identities': {**data['identities'], 'learned': {}}}
    with pytest.raises(ValueError, match='identity'):
        baseline.verify_frozen_identities(missing, model, variants=variants)


def test_matrix_accepts_a_custom_variant_table(study, monkeypatch):
    root, model, scenario, entry, data, identity, output = study
    path = root/'scenario.json'
    path.write_text(json.dumps(scenario))
    entry.update(archive(path))
    checkpoint, files, variants, data = learned_study(study)
    data['scenarios']['test'] = [entry]
    seen = []

    def unit(scenario, entry, name, model, output, variants=None):
        seen.append((name, variants))
        return retained_row((root, model, scenario, entry, data, identity, output), name)

    monkeypatch.setattr(baseline, 'run_unit', unit)
    monkeypatch.setattr(baseline, 'summary', lambda rows, variants=None, pairs=None: {'rows': len(rows), 'pairs': pairs})
    result = baseline.execute_matrix([entry], root, model, output, identity, data, variants=variants, pairs=[])
    assert result == {'rows': 3, 'pairs': [], 'protocol': identity}
    assert sorted(name for name, _ in seen) == ['a', 'b', 'ppo_s0_filtered']
    assert all(table is variants for _, table in seen)


def test_run_unit_resolves_a_frozen_relative_learned_dir_without_changing_the_recorded_config(study, monkeypatch):
    root, model, scenario, entry, data, identity, output = study
    config = {'policy_name': 'ppo', 'learned_dir': 'results/train/ppo/seed0/chunk4', 'dynamics': 'marine',
              'filtered': True}
    entry = {**entry, 'reference_route': None}
    scenario = {**scenario, 'map': {'metadata': {}}, 'start': [0, 0], 'goal': [1, 1]}
    captured = {}

    class Stop(Exception):
        pass

    def fake_benchmark(scenarios, model_dir, out, variants, runner):
        captured['variants'] = variants
        captured['execute'] = []
        monkeypatch.setattr(baseline, 'execute', lambda *a, **k: captured['execute'].append(k) or {})
        runner({}, [0, 0], [1, 1], model_dir, scenario=scenarios[0], **variants['ppo_s0'])
        raise Stop

    monkeypatch.setattr(baseline, 'benchmark', fake_benchmark)
    with pytest.raises(Stop):
        baseline.run_unit(scenario, entry, 'ppo_s0', model, output, variants={'ppo_s0': config})
    assert captured['variants'] == {'ppo_s0': config}  # what benchmark records as variant_config is untouched
    [kwargs] = captured['execute']
    assert kwargs['learned_dir'] == str(root/'results/train/ppo/seed0/chunk4') and kwargs['policy_name'] == 'ppo'
