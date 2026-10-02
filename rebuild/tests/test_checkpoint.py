import json
import random

import numpy as np
import pytest
from stable_baselines3 import PPO
import torch

from shipnav.maps import SeaMap
from shipnav.scenarios import make_scenario
from shipnav.training.checkpoints import (latest_checkpoint, restore_rng, save_checkpoint,
                                          update_penalty, verify_checkpoint)
from shipnav.training.env import EnvConfig, NavigationEnv


def tiny_model():
    env = NavigationEnv([make_scenario(SeaMap((0, 0, 24, 24)), (2, 2), (20, 2), 1, 1)], EnvConfig(filtered=False))
    return PPO('MlpPolicy', env, seed=3, n_steps=64, batch_size=32, device='cpu')


def test_round_trip_preserves_deterministic_actions_and_rng(tmp_path):
    model = tiny_model()
    observations = np.random.default_rng(0).uniform(-1, 1, (5, 120)).astype(np.float32)
    expected = model.predict(observations, deterministic=True)[0]
    path = save_checkpoint(model, tmp_path, 'chunk-0000', {'penalty': 0.})
    next_torch, next_python = torch.rand(3), random.random()
    manifest = verify_checkpoint(path)
    loaded = PPO.load(path/'policy.zip', device='cpu')
    restore_rng(path)  # after PPO.load: _setup_model reseeds global RNGs
    assert np.array_equal(loaded.predict(observations, deterministic=True)[0], expected)
    assert torch.equal(torch.rand(3), next_torch) and random.random() == next_python
    assert set(manifest['files']) == {'policy.zip', 'state.pt', 'rng.json'}


def test_restore_rng_is_what_recovers_advanced_generators(tmp_path):
    """The round-trip above also passes without restore_rng (load reseeds identically to
    construction), so advance every generator before saving to make restore_rng load-bearing."""
    model = tiny_model()
    torch.rand(7), random.random(), np.random.rand(5)
    path = save_checkpoint(model, tmp_path, 'chunk-0000', {})
    expected = (torch.rand(3), random.random(), np.random.rand())
    PPO.load(path/'policy.zip', device='cpu')
    reseeded = (torch.rand(3), random.random(), np.random.rand())
    assert not torch.equal(reseeded[0], expected[0]) and reseeded[1] != expected[1]
    PPO.load(path/'policy.zip', device='cpu')
    restore_rng(path)
    restored = (torch.rand(3), random.random(), np.random.rand())
    assert torch.equal(restored[0], expected[0]) and restored[1:] == expected[1:]


def test_existing_names_are_never_overwritten_and_tampering_is_detected(tmp_path):
    model = tiny_model()
    path = save_checkpoint(model, tmp_path, 'chunk-0000', {})
    with pytest.raises(FileExistsError):
        save_checkpoint(model, tmp_path, 'chunk-0000', {})
    (path/'state.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash'):
        verify_checkpoint(path)


def test_latest_checkpoint_ignores_incomplete_folders(tmp_path):
    model = tiny_model()
    save_checkpoint(model, tmp_path, 'chunk-0000', {'chunk': 0})
    model.num_timesteps = 64
    save_checkpoint(model, tmp_path, 'chunk-0001', {'chunk': 1})
    (tmp_path/'chunk-0002').mkdir()  # no manifest: crashed upload or copy
    assert latest_checkpoint(tmp_path).name == 'chunk-0001'
    assert latest_checkpoint(tmp_path/'missing') is None


def test_penalty_rises_on_violation_and_stays_nonnegative():
    assert update_penalty(1., 2., 1.) > 1.
    assert update_penalty(0., 0., 1.) == 0.
    with pytest.raises(ValueError):
        update_penalty(0., float('nan'), 1.)
