import numpy as np
import pytest

from shipnav.dynamics import Vessel
from shipnav.maps import SeaMap
from shipnav.training.observation import SIZE, action_to_velocity, encode, ray_lengths


def packet(**overrides):
    base = {'position': (5., 5.), 'velocity': (.5, 0.), 'goal': (100., 5.), 'final': True,
            'observed': [{'id': 0, 'position': (8., 5.), 'velocity': (-1., 0.), 'radius': .6,
                          'age': 0., 'margin': 0.}],
            'radius': .5, 'speed': 1., 'vessel': Vessel(5., 5., 0., .5), 'sea': SeaMap((0, 0, 24, 24))}
    return {**base, **overrides}


def test_encoding_is_bounded_and_sized():
    values = encode(packet())
    assert values.shape == (SIZE,) and values.dtype == np.float32
    assert np.all(values >= -1) and np.all(values <= 1)
    assert values[2] == 1.  # far goal clipped
    assert values[15] == 1. and values[23] == 0.  # first target mask on, second off


def test_targets_sorted_nearest_first_and_capped_at_twelve():
    far = [{'id': i, 'position': (5.+i+2, 5.), 'velocity': (0., 0.), 'radius': .6, 'age': 0., 'margin': 0.}
           for i in range(15)]
    values = encode(packet(observed=list(reversed(far))))
    assert values[8] == pytest.approx(2/20)
    assert all(values[8+8*k+7] == 1. for k in range(12))


def test_rays_measure_land_and_edges():
    sea = SeaMap((0, 0, 24, 24), ((10, 0, 12, 24),))
    rays = ray_lengths(sea, (5., 5.), .5)
    assert rays[0] == pytest.approx(4.5, abs=.05)   # east, to the land face minus radius
    assert rays[8] == pytest.approx(4.5, abs=.05)   # west, to the map edge minus radius


def test_holonomic_packets_are_rejected():
    with pytest.raises(ValueError, match='marine'):
        encode(packet(vessel=None))


def test_action_transform_clips_to_the_unit_disk_and_rejects_bad_input():
    assert action_to_velocity([1., 1.], 1.) == pytest.approx((2**-.5, 2**-.5))
    assert action_to_velocity([.3, .4], 2.) == pytest.approx((.6, .8))
    assert action_to_velocity([0., 0.], 1.) == (0., 0.)
    with pytest.raises(ValueError):
        action_to_velocity([np.nan, 0.], 1.)


def test_golden_vector_for_one_fixed_packet():
    """Literal schema-1 values (the rays also agree with the analytic wall distances to the bisection tolerance)."""
    values = encode(packet())
    ego = [1., 0., 1., .5, 0., 0., 1., 1.]  # goal direction, goal range, velocity/speed, sin/cos heading, final
    target = [.15, 0., -.5, 0., .3, 0., 0., 1.]  # offset/20, velocity/2, radius/2, age/5, margin/5, mask
    rays = [.924805, 1., 1., 1., .924805, .587891, .317383, .243164,
            .224609, .243164, .317383, .243164, .224609, .243164, .317383, .587891]
    assert values.shape == (120,)
    assert values[:8] == pytest.approx(ego, abs=1e-6)
    assert values[8:16] == pytest.approx(target, abs=1e-6)
    assert not values[16:104].any()  # eleven empty target slots
    assert values[104:] == pytest.approx(rays, abs=1e-6)


@pytest.mark.parametrize('overrides', [{'goal': (float('nan'), 5.)}, {'position': (float('nan'), 5.)},
                                       {'goal': (float('inf'), 5.)}])
def test_a_non_finite_feature_is_an_error_not_an_observation(overrides):
    with pytest.raises(ValueError, match='finite'):
        encode(packet(**overrides))
