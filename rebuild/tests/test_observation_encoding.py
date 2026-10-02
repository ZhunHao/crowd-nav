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
