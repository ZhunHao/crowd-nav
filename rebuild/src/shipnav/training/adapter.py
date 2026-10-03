"""Deterministic trained-PPO controller for the service and GUI.

It consumes the engine's decision packet and encodes it with the same function
as NavigationEnv, so a saved policy acts identically in training and deployment.

Security: SB3's `PPO.load` decodes the zip's `data` entry with cloudpickle, i.e. it runs arbitrary code from
whatever model folder it is given. A hash-verified manifest only proves the folder is unchanged, not that it
is safe. So this adapter never calls `PPO.load`: it reads only the weight tensors (`torch.load(...,
weights_only=True)` inside `load_from_zip_file(load_data=False)`), rebuilds the schema-1 actor-critic network
and loads the weights strictly. Loading never reseeds the global RNGs either.
"""
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.save_util import load_from_zip_file

from shipnav.training.checkpoints import verify_checkpoint
from shipnav.training.observation import ACTION_SCHEMA, SCHEMA, SIZE, action_to_velocity, encode


def _schema1_policy(device):
    """The network `PPO('MlpPolicy', NavigationEnv)` builds: PPO passes only use_sde=False to the policy
    constructor, so net_arch (pi/vf [64, 64]), Tanh activation and the other defaults are ActorCriticPolicy's."""
    observation = spaces.Box(-1., 1., (SIZE,), dtype=np.float32)
    action = spaces.Box(-1., 1., (2,), dtype=np.float32)
    with torch.random.fork_rng(devices=[]):  # weight initialisation must not disturb the caller's RNG
        policy = ActorCriticPolicy(observation, action, lambda _: 3e-4, use_sde=False)
    return policy.to(device)


class ModernLearned:
    def __init__(self, directory, device='cpu'):
        directory = Path(directory)
        manifest = verify_checkpoint(directory)
        found = (manifest.get('observation_schema'), manifest.get('action_schema'), manifest.get('dynamics'))
        if found != (SCHEMA, ACTION_SCHEMA, 'marine'):
            raise ValueError(f'Incompatible checkpoint manifest {found}: needs observation schema {SCHEMA}, '
                             f'action schema {ACTION_SCHEMA}, marine dynamics')
        _, params, _ = load_from_zip_file(directory/'policy.zip', load_data=False, device=device)
        policy = _schema1_policy(device)
        try:
            policy.load_state_dict(params['policy'], strict=True)
        except (KeyError, RuntimeError) as error:
            raise ValueError(f'Checkpoint weights do not match observation/action schema 1: {error}') from error
        policy.set_training_mode(False)
        self.policy, self.manifest, self.hashes = policy, manifest, dict(manifest['files'])

    def act(self, packet):
        action, _ = self.policy.predict(encode(packet), deterministic=True)
        return action_to_velocity(action, packet['speed'])
