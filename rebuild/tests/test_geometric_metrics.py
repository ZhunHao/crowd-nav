from math import pi
import pytest
from shipnav.maps import SeaMap
from shipnav.metrics import point_segment_distance, land_clearance, hierarchical_interval, geometry_metrics
from shipnav.service import execute


def test_geometric_metrics_have_known_values():
    assert point_segment_distance((3, 4), (0, 0), (10, 0)) == 4
    assert point_segment_distance((3, 4), (0, 0), (0, 0)) == 5
    assert land_clearance(SeaMap((0, 0, 10, 10)), (2, 5), .5) == 1.5
    assert land_clearance(SeaMap((0, 0, 10, 10), ((4, 4, 6, 6),)), (5, 5), .5) == -1.5


def test_hierarchical_interval_uses_both_seed_levels():
    a = {0: {'x': 1, 'y': 1}, 1: {'x': 1, 'y': 1}}
    b = {0: {'x': 0, 'y': 0}, 1: {'x': 0, 'y': 0}}
    assert hierarchical_interval(a, b)['low'] == 1
    with pytest.raises(ValueError):
        hierarchical_interval(a, {0: b[0]})
    with pytest.raises(ValueError):
        hierarchical_interval(a, {0: b[0], 1: {'x': 0}})
    with pytest.raises(ValueError):
        hierarchical_interval(a, {0: b[0], 1: {'x': 0, 'y': None}})


def test_marine_run_starting_off_east_has_no_false_dynamics_violation():
    run = execute(SeaMap((0, 0, 24, 24)).to_dict(), (2, 2), (2, 20),
                  policy_name='direct', count=0, dynamics='marine')
    result = geometry_metrics(run)
    assert run['status'] == 'success' and result['dynamics_violations'] == 0
    assert result['cross_track_max_m'] < 1e-6
    assert geometry_metrics({'status': 'error', 'error': 'x'}) == {}


def test_geometry_converts_length_and_exposure_but_checks_dynamics_in_model_frame():
    sea = SeaMap((0, 0, 10, 10)).to_dict()
    sea['metadata']['model_scale'] = {'length_m': 10., 'speed_mps': 5.}
    run = {'status': 'timeout', 'map': sea, 'route': [(2, 2), (2, 8)], 'settings': {'radius': .5},
           'frames': [{'t': 0, 'position': (2, 2)}, {'t': .25, 'position': (3, 3)},
                      {'t': .35, 'position': (4, 4)}],
           'diagnostics': [{'clearance': .05, 'heading': pi/2, 'speed': .05},
                           {'clearance': .2, 'heading': pi/2+.05, 'speed': .07}]}
    result = geometry_metrics(run)
    assert result['cross_track_mean_m'] == 10
    assert result['cross_track_max_m'] == 20
    assert result['sampled_land_clearance_m'] == 15
    assert result['domain_time_s'] == .5
    assert result['dynamics_violations'] == 1


def test_heading_wrap_does_not_create_false_violation():
    run = {'map': SeaMap((-10, -10, 10, 10)).to_dict(), 'route': [(0, 0), (-2, .01)],
           'settings': {'radius': .5},
           'frames': [{'t': 0, 'position': (0, 0)}, {'t': .25, 'position': (-.01, 0)}],
           'diagnostics': [{'heading': -pi+.01, 'speed': .05}]}
    assert geometry_metrics(run)['dynamics_violations'] == 0


def test_marine_no_global_goals_initializes_from_navigation_leg_on_bent_planned_route():
    sea = SeaMap((0, 0, 24, 24), ((10, 6, 14, 18),))
    run = execute(sea.to_dict(), (2, 12), (22, 12), policy_name='direct', count=0,
                  dynamics='marine', global_goals=False, limit=.25)
    assert len(run['route']) > 2
    assert run['goals'] == [(2, 12), (22, 12)]
    assert geometry_metrics(run)['dynamics_violations'] == 0


def test_hierarchical_interval_resamples_training_seeds_instead_of_pooling_rollouts():
    a = {0: {str(i): 0 for i in range(20)}, 1: {str(i): 10 for i in range(20)}}
    b = {key: {str(i): 0 for i in range(20)} for key in a}
    result = hierarchical_interval(a, b)
    assert result['difference'] == 5
    assert result['low'] == 0 and result['high'] == 10
    assert result['training_seeds'] == 2
