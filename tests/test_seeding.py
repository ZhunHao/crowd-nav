import random

import numpy as np
import pytest
import torch


@pytest.mark.unit
def test_seed_everything_makes_rng_deterministic():
    from crowd_sim.envs.utils.seeding import seed_everything

    seed_everything(42)
    r1, n1, t1 = random.random(), np.random.rand(), torch.rand(1).item()

    seed_everything(42)
    r2, n2, t2 = random.random(), np.random.rand(), torch.rand(1).item()

    assert r1 == r2
    assert n1 == n2
    assert t1 == t2


@pytest.mark.unit
def test_seed_everything_rejects_non_int():
    from crowd_sim.envs.utils.seeding import seed_everything

    with pytest.raises(TypeError):
        seed_everything("42")  # type: ignore[arg-type]
