"""Orientation handling for the CommonOcean collision oracle; runs without
commonocean installed (the helper is pure Python)."""
from math import atan2, pi

import pytest

from shipnav.adapters.commonocean import trajectory_orientations
from shipnav.maps import SeaMap
from shipnav.policies import Direct
from shipnav.simulation import run_episode


def test_holonomic_result_orientation_is_derived_from_displacement():
    result = run_episode(SeaMap((0, 0, 24, 24)), [(2, 2), (6, 6)], [], Direct())
    assert 'heading' not in result['diagnostics'][0]
    orientations = trajectory_orientations(result)
    assert len(orientations) == len(result['frames'])
    assert all(o == pytest.approx(pi/4) for o in orientations)


def test_marine_result_uses_recorded_headings():
    result = run_episode(SeaMap((0, 0, 24, 24)), [(2, 2), (2, 10)], [], Direct(), dynamics='marine', limit=2)
    orientations = trajectory_orientations(result)
    assert orientations[1:] == [d['heading'] for d in result['diagnostics']]
    assert orientations[0] == orientations[1]


def test_stationary_steps_keep_the_previous_orientation():
    result = {'frames': [{'t': 0., 'position': [0., 0.]}, {'t': 1., 'position': [1., 1.]},
                         {'t': 2., 'position': [1., 1.]}],
              'diagnostics': [{}, {}]}
    assert trajectory_orientations(result) == [atan2(1, 1)]*3
    assert trajectory_orientations({'frames': [{'t': 0., 'position': [0., 0.]}], 'diagnostics': []}) == [0.]
