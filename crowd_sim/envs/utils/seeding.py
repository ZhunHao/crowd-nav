"""RNG seeding utility — single entry point for deterministic runs (R1 / NF3)."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed python-random, numpy, and torch (CPU + CUDA) with a single int.

    Called from ``test.py`` and ``train.py`` so a given ``--seed`` produces
    bit-identical rollouts on the same hardware. See slide 13 task 1.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
