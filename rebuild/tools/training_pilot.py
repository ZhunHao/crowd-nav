"""Measure environment and learner throughput for EC2 sizing. Not a performance claim."""
from pathlib import Path
import argparse
import json
import os
import platform
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from shipnav.training.env import EnvConfig, NavigationEnv
from shipnav.training.scenarios import load_split


def env_rate(scenarios, filtered, steps=1000):
    env, rng = NavigationEnv(scenarios, EnvConfig(filtered=filtered)), np.random.default_rng(0)
    env.reset(seed=0)
    began = time.perf_counter()
    for _ in range(steps):
        *_, terminated, truncated, _ = env.step(rng.uniform(-1, 1, 2))
        if terminated or truncated:
            env.reset()
    return steps/(time.perf_counter()-began)


def learn_rate(scenarios, envs, device, steps):
    env = make_vec_env(lambda: NavigationEnv(scenarios, EnvConfig()), n_envs=envs, seed=0,
                       vec_env_cls=SubprocVecEnv if envs > 1 else DummyVecEnv)
    try:
        model = PPO('MlpPolicy', env, seed=0, device=device, n_steps=2048//envs, batch_size=64)
        began = time.perf_counter()
        model.learn(steps)
        return steps/(time.perf_counter()-began)
    finally:
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scenarios', type=Path, default=Path('scenarios/training'))
    p.add_argument('--devices', nargs='+', default=['cpu'])
    p.add_argument('--envs', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--steps', type=int, default=8192)
    p.add_argument('--env-only', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    scenarios = load_split(a.scenarios, 'train')[:500]
    result = {'host': platform.node(), 'platform': platform.platform(), 'cpus': os.cpu_count(),
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'cuda_available': torch.cuda.is_available(),
              'env_steps_per_s': {f: env_rate(scenarios, f) for f in (False, True)}, 'learn_steps_per_s': {}}
    if not a.env_only:
        for device in a.devices:
            for envs in a.envs:
                result['learn_steps_per_s'][f'{device}/envs={envs}'] = learn_rate(scenarios, envs, device, a.steps)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
