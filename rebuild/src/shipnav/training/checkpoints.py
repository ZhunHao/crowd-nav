"""Checkpoints: policy/optimizer (SB3 zip), RNG state and a manifest written last.

RNG state avoids pickle: Python/NumPy generators go to rng.json and torch
generator tensors to state.pt, loaded with `weights_only=True`. SB3's
policy.zip still deserialises objects (cloudpickle) when loaded with `PPO.load`, which a manifest hash
does not make safe: `PPO.load` is limited to `train.py --resume` on the trainer's own output directory.
Everything else (service, GUI, CLI) loads only the weights through `shipnav.training.adapter.ModernLearned`.
"""
from hashlib import sha256
from math import isfinite
from pathlib import Path
import json
import os
import random
import tempfile

import numpy as np
import torch

FILES = ('policy.zip', 'state.pt', 'rng.json')
IDENTITY = ('variant', 'seed', 'envs', 'chunk_steps', 'env_config', 'scenario_manifest_sha256')  # one run's settings
DEFAULT_CHUNK_STEPS = 20480  # the trainer's --chunk-steps default; cloud/resume_target.py compares against it


def _rng_json():
    version, internal, gauss = random.getstate()
    name, key, pos, has_gauss, cached = np.random.get_state()
    return {'python': [version, list(internal), gauss],
            'numpy': [name, key.tolist(), int(pos), int(has_gauss), float(cached)]}


def _fsync(path):
    """Flush a file's data, or a directory's entries, to stable storage."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_checkpoint(model, root, name, metadata):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    final = root/name
    if final.exists():
        raise FileExistsError(final)
    staging = Path(tempfile.mkdtemp(prefix=f'.{name}-', dir=root))
    model.save(staging/'policy.zip')
    torch.save({'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
               staging/'state.pt')
    (staging/'rng.json').write_text(json.dumps(_rng_json()))
    files = {n: sha256((staging/n).read_bytes()).hexdigest() for n in FILES}
    manifest = {**metadata, 'num_timesteps': model.num_timesteps, 'files': files}
    (staging/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+'\n')
    for name in (*FILES, 'manifest.json'):  # durable before the rename publishes the folder, even across a power loss
        _fsync(staging/name)
    _fsync(staging)
    os.rename(staging, final)
    _fsync(root)
    return final


class IncompleteCheckpointError(ValueError):
    """The folder lacks a usable manifest or one of its payload files (e.g. a half-synced copy)."""


def _read_manifest(directory):
    try:
        manifest = json.loads((directory/'manifest.json').read_text())
    except (OSError, ValueError) as error:  # JSONDecodeError and UnicodeDecodeError are ValueErrors
        raise IncompleteCheckpointError(f'Checkpoint incomplete: unreadable manifest in {directory}: {error}') from error
    files = manifest.get('files') if isinstance(manifest, dict) else None
    if (not isinstance(files, dict) or not all(isinstance(files.get(n), str) for n in FILES)
            or not isinstance(manifest.get('num_timesteps'), int)):
        raise IncompleteCheckpointError(f'Checkpoint incomplete: malformed manifest in {directory}')
    return manifest


def verify_checkpoint(directory):
    """Return the manifest. Raise IncompleteCheckpointError (a ValueError) for a missing/malformed
    manifest or missing payload file, and a plain ValueError mentioning 'hash' for corruption."""
    directory = Path(directory)
    manifest = _read_manifest(directory)
    missing = [n for n in FILES if not (directory/n).is_file()]
    if missing:
        raise IncompleteCheckpointError(f'Checkpoint incomplete: missing {", ".join(missing)} in {directory}')
    for name in FILES:
        if sha256((directory/name).read_bytes()).hexdigest() != manifest['files'][name]:
            raise ValueError(f'Checkpoint hash mismatch: {name}')
    return manifest


def restore_rng(directory):
    """Call after PPO.load, whose model setup reseeds the global RNGs."""
    directory = Path(directory)
    state = torch.load(directory/'state.pt', weights_only=True)
    rng = json.loads((directory/'rng.json').read_text())
    version, internal, gauss = rng['python']
    random.setstate((version, tuple(internal), gauss))
    name, key, pos, has_gauss, cached = rng['numpy']
    np.random.set_state((name, np.asarray(key, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state


def latest_checkpoint(root):
    root = Path(root)
    if not root.is_dir():
        return None
    complete = []
    for path in root.iterdir():
        if not path.is_dir() or path.name.startswith('.'):
            continue
        try:
            manifest = verify_checkpoint(path)
        except IncompleteCheckpointError:
            continue  # never a checkpoint; a hash mismatch (corruption) still propagates
        complete.append((manifest['num_timesteps'], path.name, path))
    return max(complete)[2] if complete else None


def update_penalty(current, mean_episode_cost, budget, rate=.05):
    if not all(isfinite(x) for x in (current, mean_episode_cost, budget, rate)):
        raise ValueError('Penalty update inputs must be finite')
    return max(0., current+rate*(mean_episode_cost-budget))
