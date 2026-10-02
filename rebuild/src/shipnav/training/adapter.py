"""Deterministic trained-PPO controller for the service and GUI.

It consumes the engine's decision packet and encodes it with the same function
as NavigationEnv, so a saved policy acts identically in training and deployment.
"""
from pathlib import Path

from shipnav.training.checkpoints import verify_checkpoint
from shipnav.training.observation import ACTION_SCHEMA, SCHEMA, SIZE, action_to_velocity, encode


class ModernLearned:
    def __init__(self, directory, device='cpu'):
        directory = Path(directory)
        manifest = verify_checkpoint(directory)
        found = (manifest.get('observation_schema'), manifest.get('action_schema'), manifest.get('dynamics'))
        if found != (SCHEMA, ACTION_SCHEMA, 'marine'):
            raise ValueError(f'Incompatible checkpoint manifest {found}: needs observation schema {SCHEMA}, '
                             f'action schema {ACTION_SCHEMA}, marine dynamics')
        from stable_baselines3 import PPO
        self.model = PPO.load(directory/'policy.zip', device=device)
        if self.model.observation_space.shape != (SIZE,) or self.model.action_space.shape != (2,):
            raise ValueError('Checkpoint spaces do not match observation/action schema 1')
        self.manifest, self.hashes = manifest, dict(manifest['files'])

    def act(self, packet):
        action, _ = self.model.predict(encode(packet), deterministic=True)
        return action_to_velocity(action, packet['speed'])
