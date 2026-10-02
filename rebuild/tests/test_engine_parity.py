from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from engine_parity import CASES, FIXTURE, NEEDS_CHECKPOINT, SARL, digest  # noqa: E402

EXPECTED = json.loads(FIXTURE.read_text())


@pytest.mark.parametrize('name', sorted(set(CASES)-NEEDS_CHECKPOINT))
def test_episode_physics_unchanged(name):
    assert digest(name) == EXPECTED[name]


@pytest.mark.integration
@pytest.mark.parametrize('name', sorted(NEEDS_CHECKPOINT))
def test_sarl_episode_physics_unchanged(name):
    if not (SARL/'rl_model.pth').is_file():
        pytest.skip('Supplied SARL checkpoint not present')
    assert digest(name) == EXPECTED[name]
