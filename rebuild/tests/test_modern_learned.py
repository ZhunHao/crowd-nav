import base64
import json
import pickle
import random
from hashlib import sha256
import zipfile

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from shipnav.maps import SeaMap
from shipnav.scenarios import make_scenario
from shipnav.service import execute
from shipnav.training.adapter import ModernLearned
from shipnav.training.checkpoints import save_checkpoint
from shipnav.training.env import EnvConfig, NavigationEnv
from shipnav.training.observation import ACTION_SCHEMA, SCHEMA, action_to_velocity


@pytest.fixture
def checkpoint(tmp_path):
    scene = make_scenario(SeaMap((0, 0, 24, 24)), (2, 2), (20, 2), 2, 4)
    model = PPO('MlpPolicy', NavigationEnv([scene], EnvConfig(filtered=False)), seed=1, n_steps=64, device='cpu')
    meta = {'observation_schema': SCHEMA, 'action_schema': ACTION_SCHEMA, 'dynamics': 'marine'}
    return scene, model, save_checkpoint(model, tmp_path, 'chunk-0000', meta)


def test_service_and_gym_issue_identical_actions(checkpoint):
    scene, model, path = checkpoint
    run = execute(scene['map'], scene['start'], scene['goal'], policy_name='ppo', learned_dir=str(path),
                  scenario=scene, dynamics='marine', filtered=False)
    env = NavigationEnv([scene], EnvConfig(filtered=False))
    observation, _ = env.reset(seed=0)
    for diagnostic in run['diagnostics'][:20]:
        expected = action_to_velocity(model.predict(observation, deterministic=True)[0], 1.)
        assert diagnostic['nominal'] == pytest.approx(expected, abs=1e-6)
        observation, *_ = env.step(model.predict(observation, deterministic=True)[0])
    assert run['model_hashes'] == json.loads((path/'manifest.json').read_text())['files']


def test_incompatible_or_tampered_checkpoints_are_rejected(checkpoint):
    _, _, path = checkpoint
    manifest = json.loads((path/'manifest.json').read_text())
    (path/'manifest.json').write_text(json.dumps({**manifest, 'observation_schema': 0}))
    with pytest.raises(ValueError, match='schema'):
        ModernLearned(path)
    (path/'manifest.json').write_text(json.dumps(manifest))
    (path/'policy.zip').write_bytes(b'not a model')
    with pytest.raises(ValueError, match='hash'):
        ModernLearned(path)


def test_ppo_requires_marine_dynamics(checkpoint):
    scene, _, path = checkpoint
    with pytest.raises(ValueError, match='marine'):
        execute(scene['map'], scene['start'], scene['goal'], policy_name='ppo', learned_dir=str(path),
                scenario=scene, dynamics='holonomic')


class _Marker:
    """Unpickling this object runs code: it writes a marker file."""
    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return exec, (f'open({self.path!r}, "w").write("pwned")',)


def _plant_pickle(path, marker):
    """Replace policy.zip's `data` entry with a cloudpickle-style payload and re-hash the manifest."""
    payload = base64.b64encode(pickle.dumps(_Marker(marker))).decode()
    data = json.dumps({'policy_class': {':type:': "<class 'type'>", ':serialized:': payload}})
    source = path/'policy.zip'
    entries = {}
    with zipfile.ZipFile(source) as archive:
        for name in archive.namelist():
            entries[name] = archive.read(name)
    entries['data'] = data.encode()
    with zipfile.ZipFile(source, 'w') as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    manifest = json.loads((path/'manifest.json').read_text())
    manifest['files']['policy.zip'] = sha256(source.read_bytes()).hexdigest()
    (path/'manifest.json').write_text(json.dumps(manifest))


def test_a_malicious_pickle_in_policy_zip_is_never_executed(checkpoint, tmp_path):
    scene, model, path = checkpoint
    marker = tmp_path/'pwned.txt'
    _plant_pickle(path, marker)
    try:  # the payload is real: SB3's own loader runs it (proves this test can fail)
        PPO.load(path/'policy.zip', device='cpu')
    except Exception:  # noqa: BLE001 - the planted data is not a valid model, only the side effect matters
        pass
    assert marker.exists()
    marker.unlink()
    learned = ModernLearned(path)  # passes verify_checkpoint (hash re-computed) but must not unpickle `data`
    assert not marker.exists()
    env = NavigationEnv([scene], EnvConfig(filtered=False))
    observation, _ = env.reset(seed=0)
    expected = action_to_velocity(model.predict(observation, deterministic=True)[0], env.packet['speed'])
    assert learned.act(env.packet) == pytest.approx(expected, abs=1e-6)
    assert not marker.exists()


def test_loading_leaves_the_global_rngs_alone(checkpoint):
    _, _, path = checkpoint
    before = (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state())
    ModernLearned(path)
    after = (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state())
    assert before[0] == after[0] and (before[1] == after[1]).all() and torch.equal(before[2], after[2])


def test_weights_for_another_network_are_rejected(checkpoint, tmp_path):
    scene, _, _ = checkpoint
    other = PPO('MlpPolicy', NavigationEnv([scene], EnvConfig(filtered=False)), seed=1, n_steps=64, device='cpu',
                policy_kwargs={'net_arch': [32]})
    meta = {'observation_schema': SCHEMA, 'action_schema': ACTION_SCHEMA, 'dynamics': 'marine'}
    with pytest.raises(ValueError, match='schema 1'):
        ModernLearned(save_checkpoint(other, tmp_path/'other', 'chunk-0000', meta))
