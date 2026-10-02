import json

import numpy as np
import pytest
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
