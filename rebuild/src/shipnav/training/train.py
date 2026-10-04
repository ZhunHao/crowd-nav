"""Bounded-chunk PPO / Lagrangian-penalty training with full-state checkpoints."""
from multiprocessing import parent_process
from pathlib import Path
from statistics import mean
from threading import Event
import argparse
import json
import os
import signal
import time

import gymnasium
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from shipnav.provenance import provenance
from shipnav.training.checkpoints import (DEFAULT_CHUNK_STEPS, IDENTITY, latest_checkpoint, restore_rng,
                                          save_checkpoint, update_penalty, verify_checkpoint)
from shipnav.training.env import EnvConfig, NavigationEnv
from shipnav.training.observation import ACTION_SCHEMA, SCHEMA
from shipnav.training.scenarios import load_split

STOP = Event()
ROLLOUT, BATCH, CHECKPOINT_GRACE_S = 2048, 64, 120.
ROOT = Path(__file__).resolve().parents[4]  # repository root, as service._ROOT


class EpisodeStats(BaseCallback):
    """Collects completed-episode costs; stops at the next env step on SIGTERM or deadline.

    `trained` is num_timesteps at the end of the last completed rollout. SB3 calls on_rollout_end only after a
    rollout is fully collected, and train() follows it, so `trained` counts exactly the steps the policy was
    updated on. An interrupted rollout is discarded by SB3, so the trainer rolls num_timesteps back to it."""
    def __init__(self, deadline, start=0):
        super().__init__()
        self.deadline, self.costs, self.statuses, self.interrupted = deadline, [], [], False
        self.trained = start

    def _on_rollout_end(self):
        self.trained = self.model.num_timesteps

    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode_cost' in info:
                self.costs.append(info['episode_cost'])
                self.statuses.append(info['status'])
        self.interrupted = STOP.is_set() or time.monotonic() >= self.deadline
        return not self.interrupted


def make_env(scenarios, config):
    """Environment factory. In a SubprocVecEnv worker, ignore SIGTERM so a group-wide signal (docker stop,
    systemd control-group kill) lets the trainer save a partial checkpoint before it closes the workers."""
    if parent_process() is not None:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    return NavigationEnv(scenarios, config)


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenarios', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--variant', choices=('ppo', 'ppo_lagrangian'), required=True)
    p.add_argument('--budget', type=float)
    p.add_argument('--chunks', type=int, required=True)
    p.add_argument('--chunk-steps', type=int, default=DEFAULT_CHUNK_STEPS)
    p.add_argument('--envs', type=int, default=8)
    p.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    p.add_argument('--max-seconds', type=float, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--source-commit', default=os.environ.get('SHIPNAV_SOURCE_COMMIT') or None,
                   help='Revision recorded in each manifest; the EC2 host has no .git, so the runbook passes it')
    p.add_argument('--no-filter', action='store_true')
    p.add_argument('--observation', type=json.loads, default={})
    a = p.parse_args(argv)
    if a.variant == 'ppo_lagrangian' and a.budget is None:
        p.error('--budget (physical domain seconds per episode) is required for ppo_lagrangian')
    if min(a.chunks, a.chunk_steps, a.envs) < 1 or a.max_seconds <= 0:
        p.error('chunks, chunk-steps, envs and max-seconds must be positive')
    rollout = min(ROLLOUT, a.chunk_steps)
    if rollout % a.envs or a.chunk_steps % rollout:
        p.error('chunk-steps must be a multiple of the rollout, and the rollout divisible by envs')
    a.n_steps = rollout//a.envs
    return a


def identity(a, config):
    manifest = json.loads((a.scenarios/'manifest.json').read_text())
    return {'variant': a.variant, 'seed': a.seed, 'envs': a.envs, 'chunk_steps': a.chunk_steps,
            'env_config': config.to_dict(), 'scenario_manifest_sha256': manifest['splits']['train']['sha256']}


def write_status(output, state, **fields):
    status = {'state': state, **fields}
    (output/'status.json').write_text(json.dumps(status, indent=2, allow_nan=False)+'\n')
    return status


def request_stop(signum, frame):
    """Ignore duplicate TERM before touching Event's non-reentrant lock."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    STOP.set()


def main(argv=None):
    a = parse(argv)
    previous = signal.signal(signal.SIGTERM, request_stop)
    deadline = time.monotonic()+a.max_seconds-CHECKPOINT_GRACE_S
    a.output.mkdir(parents=True, exist_ok=True)
    last = latest_checkpoint(a.output) if a.resume else None
    if not a.resume and latest_checkpoint(a.output):
        raise SystemExit('Output already has checkpoints; use --resume or a new directory')
    resumed = verify_checkpoint(last) if last else None
    if resumed and not resumed['complete_chunk'] and resumed['chunk'] >= a.chunks:
        raise SystemExit('Latest checkpoint is a partial chunk beyond --chunks; raise --chunks or use a new directory')
    penalty = resumed['penalty_next'] if resumed else 0.
    config = EnvConfig(observation=a.observation, filtered=not a.no_filter, penalty=penalty)
    ident = identity(a, EnvConfig(observation=a.observation, filtered=not a.no_filter))
    if resumed and {k: resumed[k] for k in IDENTITY} != ident:
        raise SystemExit('Resume configuration differs from the checkpoint')
    scenarios = load_split(a.scenarios, 'train')
    env = make_vec_env(lambda: make_env(scenarios, config), n_envs=a.envs, seed=a.seed,
                       vec_env_cls=SubprocVecEnv if a.envs > 1 else DummyVecEnv)
    try:
        if last:
            # PPO.load unpickles policy.zip: acceptable only here, for our own output directory (see checkpoints.py)
            model = PPO.load(last/'policy.zip', env=env, device=a.device)
            restore_rng(last)
            env.seed(a.seed*1_000_003+model.num_timesteps)
            first = resumed['chunk']+(1 if resumed['complete_chunk'] else 0)  # a partial chunk is re-entered
        else:
            model = PPO('MlpPolicy', env, seed=a.seed, device=a.device, n_steps=a.n_steps, batch_size=BATCH)
            first = 0
        state = 'complete'
        for chunk in range(first, a.chunks):
            began, target = time.monotonic(), (chunk+1)*a.chunk_steps
            stats = EpisodeStats(deadline, model.num_timesteps)
            if not chunk*a.chunk_steps <= model.num_timesteps < target:
                raise RuntimeError(f'chunk {chunk} cannot start at num_timesteps={model.num_timesteps}')
            if STOP.is_set() or time.monotonic() >= deadline:
                stats.interrupted = True  # stop before learning: save a zero-step partial checkpoint
            else:
                env.env_method('set_penalty', penalty)
                model.learn(target-model.num_timesteps, reset_num_timesteps=False, callback=stats)
            done = not stats.interrupted
            if done and model.num_timesteps != target:
                raise RuntimeError(f'chunk {chunk} ended at {model.num_timesteps}, not {target}')
            used = penalty
            if done:  # one multiplier update per logical chunk, from this segment's completed episodes
                if a.variant == 'ppo_lagrangian' and stats.costs:
                    penalty = update_penalty(penalty, mean(stats.costs), a.budget)
            else:  # drop the interrupted rollout so recorded steps equal trained steps
                model.num_timesteps = stats.trained
            name = f'chunk-{chunk:04d}' if done else f'partial-{chunk:04d}-{model.num_timesteps}'
            if done or not (a.output/name).exists():  # a repeat zero-progress stop re-states the same partial
                prov = provenance(ident, ROOT)
                save_checkpoint(model, a.output, name, {
                    **ident, 'source_commit': a.source_commit or prov['git_revision'], 'chunk': chunk, 'complete_chunk': done, 'penalty_used': used, 'penalty_next': penalty,
                    'budget': a.budget, 'observation_schema': SCHEMA, 'action_schema': ACTION_SCHEMA,
                    'dynamics': 'marine', 'device': a.device, 'resume_mode': 'restart-equivalent',
                    'versions': {'torch': torch.__version__, 'stable_baselines3': stable_baselines3.__version__,
                                 'gymnasium': gymnasium.__version__},
                    'provenance': prov})
            with (a.output/'curve.jsonl').open('a') as stream:
                stream.write(json.dumps({'chunk': chunk, 'complete_chunk': done, 'num_timesteps': model.num_timesteps,
                    'episodes': len(stats.costs), 'mean_episode_cost': mean(stats.costs) if stats.costs else None,
                    'success_rate': stats.statuses.count('success')/len(stats.statuses) if stats.statuses else None,
                    'collision_rate': stats.statuses.count('collision')/len(stats.statuses) if stats.statuses else None,
                    'penalty_used': used, 'penalty_next': penalty, 'wall_s': time.monotonic()-began})+'\n')
            if not done:
                state = 'stopped' if STOP.is_set() else 'deadline'
                break
        return write_status(a.output, state, num_timesteps=model.num_timesteps)
    finally:
        env.close()
        signal.signal(signal.SIGTERM, previous)


def run(argv=None):
    """Process exit code: 0 when every chunk completed, 3 when stopped (signal or deadline) with a partial checkpoint."""
    return 0 if main(argv)['state'] == 'complete' else 3


if __name__ == '__main__':
    raise SystemExit(run())
