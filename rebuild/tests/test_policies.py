from pathlib import Path
from math import hypot, isfinite
import pytest
from shipnav.policies import Direct, Learned, Reciprocal

MODEL = Path(__file__).resolve().parents[2] / 'CrowdNav-20250813-DIP/crowd_nav/data/output_trained'


def test_direct_stops_exactly_and_respects_speed():
    direct = Direct()
    assert direct((2, 2), (0, 0), (2, 2), [], .5, 1, .25) == (0, 0)
    assert hypot(*direct((2, 2), (0, 0), (8, 8), [], .5, 1, .25)) == pytest.approx(1)


@pytest.mark.integration
@pytest.mark.parametrize('kind', ['sarl', 'orca'])
def test_adapters_handle_empty_and_nonempty_neighbours(kind):
    policy = Learned(MODEL) if kind == 'sarl' else Reciprocal()
    assert policy((2, 2), (0, 0), (4, 2), [], .5, 1, .25) == (1, 0)
    velocity = policy((2, 2), (0, 0), (8, 2), [((5, 3), (0, -.3), .6)], .5, 1, .25)
    assert all(isfinite(v) for v in velocity)
    assert hypot(*velocity) <= 1.000001
