import json
from math import dist
from random import Random
import pytest
from shapely import affinity
from shapely.geometry import LineString, Point, box
from shipnav.maps import SeaMap
from shipnav.scale import PROFILES, Profile, check_grid, scaled, to_model, units
from shipnav.scenarios import load_traffic
from shipnav.service import execute

HARBOUR = SeaMap.load('maps/harbour.json')


def enlarged(sea, k):
    return SeaMap(tuple(v*k for v in sea.bounds), [affinity.scale(p, k, k, origin=(0, 0)) for p in sea.land],
                  sea.to_dict()['metadata'])


def test_scaling_is_explicit_and_not_repeated():
    assert scaled(HARBOUR, PROFILES['model']) is HARBOUR
    model = to_model(enlarged(HARBOUR, 10), PROFILES['harbour_craft'])
    with pytest.raises(ValueError):
        to_model(model, PROFILES['harbour_craft'])
    with pytest.raises(ValueError):
        check_grid(SeaMap.load('maps/singapore-ubin.json'), PROFILES['model'])
    check_grid(SeaMap.load('maps/singapore-ubin.json'), PROFILES['harbour_craft'])


def test_tiled_index_answers_exactly_like_the_merged_coastline():
    sea = to_model(SeaMap.load('maps/singapore-ubin.json'), PROFILES['harbour_craft'])
    land = sea.land
    x0, y0, x1, y1 = sea.bounds
    rng = Random(0)
    for _ in range(300):
        a = (rng.uniform(x0, x1), rng.uniform(y0, y1))
        b = (a[0]+rng.uniform(-30, 30), a[1]+rng.uniform(-30, 30))
        segment = LineString([a, b])
        inside = all(x0+1.5 < x < x1-1.5 and y0+1.5 < y < y1-1.5 for x, y in (a, b))
        assert sea.clear(a, b, 1.5) == (inside and not any(p.dwithin(segment, 1.5) for p in land))
        edge = min(min(x-x0, x1-x, y-y0, y1-y) for x, y in (a, b))
        expected = 0. if edge <= 0 else min([edge] + [p.distance(segment) for p in land])
        assert sea.minimum_clearance(a, b) == pytest.approx(expected, abs=1e-9)


def test_corridor_traffic_meets_the_reference_route_on_time_for_every_planner():
    sea = to_model(SeaMap.load('maps/singapore-ubin.json'), PROFILES['harbour_craft'])
    start, goal = (-444.98, 138.39), (445.17, 303.81)
    options = dict(policy_name='direct', seed=0, count=5, placement='corridor', limit=1.,
                   **PROFILES['harbour_craft'].service_options())
    a = execute(sea.to_dict(), start, goal, planner='astar_smooth', **options)
    b = execute(sea.to_dict(), start, goal, planner='theta', **options)
    assert a['traffic_definitions'] == b['traffic_definitions'] and len(a['traffic_definitions']) == 5
    line = LineString(a['route'])
    for ship in load_traffic(a['scenario']):
        # Some waypoint lies on the reference route and is reached exactly when a 1 m/s
        # ego following it arrives there.
        assert any(line.distance(Point(p)) < 1e-6 and abs(t-line.project(Point(p))) < 1e-6
                   for t, p in ship.waypoints)
        assert sea.clear(ship.start, ship.goal, ship.radius)


def test_cli_runs_a_real_map_in_metres(tmp_path):
    import subprocess, sys
    out = tmp_path/'ubin.json'
    done = subprocess.run([sys.executable, '-m', 'shipnav.service', '--map', 'maps/singapore-ubin.json',
                           '--policy', 'direct', '--count', '0', '--start', '-2000', '500', '--goal', '-2500', '500',
                           '--output', str(out)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    result = json.loads(out.read_text())
    assert result['status'] == 'success' and result['settings']['placement'] == 'corridor'
    assert units(result['map']) == (10., 2.)
    assert result['route'][0] == [-200., 50.]


def test_open_water_shortcut_bounds_reach_by_the_fastest_candidate():
    from shipnav import safety
    calls = []
    real = safety.local_sea
    safety.local_sea = lambda sea, p, reach, radius: calls.append(reach) or real(sea, p, reach, radius)
    try:
        safety.choose(SeaMap((0, 0, 24, 24)), (12, 12), (1+1e-6, 0), [], .5, 1., .25)
    finally:
        safety.local_sea = real
    assert calls == [pytest.approx((1+1e-6)*.25*12, abs=0)]


def test_similarity_preserves_the_model_trace():
    profile = Profile('ten', 5., 1., 2., 10.)
    model = to_model(enlarged(HARBOUR, 10), profile)
    assert model.bounds == HARBOUR.bounds
    assert units(model.to_dict()) == (10., 10.)
    options = dict(policy_name='direct', seed=3, count=4, filtered=True, **profile.service_options())
    a = execute(HARBOUR.to_dict(), (2, 2), (22, 22), **options)
    b = execute(model.to_dict(), (2, 2), (22, 22), **options)
    assert a['status'] == b['status']
    assert a['traffic_definitions'] == b['traffic_definitions']
    assert a['route'] == b['route']
    for f, g in zip(a['frames'], b['frames'], strict=True):
        assert f['t'] == g['t']
        assert f['position'] == pytest.approx(g['position'], abs=1e-9)


@pytest.mark.parametrize('values', [(0, 1, 0, 1), (.5, 0, 0, 1), (.5, 1, -1, 1),
                                    (.5, 1, 0, 0), (.5, float('nan'), 0, 1)])
def test_profile_rejects_invalid_physical_parameters(values):
    with pytest.raises(ValueError):
        Profile('invalid', *values)


def test_default_profile_and_recorded_planner_options():
    from shipnav.scale import default_profile, map_options
    assert default_profile(HARBOUR) == PROFILES['model']
    assert map_options(HARBOUR.to_dict()) == {}
    sea = SeaMap.load('maps/singapore-ubin.json')
    profile = default_profile(sea)
    assert profile == PROFILES['harbour_craft']
    model = to_model(sea, profile)
    assert units(model.to_dict()) == (10., 2.)
    assert map_options(model.to_dict()) == {'resolution': 5., 'clearance': 1.5}


def test_long_scaled_route_gets_enough_time_and_explicit_limit_is_respected():
    sea = to_model(SeaMap((0, 0, 1000, 1000)), PROFILES['harbour_craft'])
    options = dict(policy_name='direct', count=0, **PROFILES['harbour_craft'].service_options())
    run = execute(sea.to_dict(), (2, 2), (80, 80), **options)
    assert run['status'] == 'success' and run['elapsed'] > 100
    assert run['settings']['limit'] == pytest.approx(3*dist((2, 2), (80, 80)))
    limited = execute(sea.to_dict(), (2, 2), (80, 80), limit=1., **options)
    assert limited['status'] == 'timeout' and limited['elapsed'] == 1.


def test_long_synthetic_route_keeps_the_legacy_automatic_timeout():
    run = execute(SeaMap((0, 0, 100, 100)).to_dict(), (2, 2), (80, 80),
                  policy_name='direct', count=0)
    assert run['status'] == 'timeout' and run['elapsed'] == 100.
    assert len(run['frames']) == 401 and run['settings']['limit'] == 100.


def test_grid_preflight_rounds_fractional_dimensions_like_the_planner():
    from shipnav.planning import PlanningLimit, plan
    sea = SeaMap((0, 0, 500.1, 499.8), ((240, 100, 260, 400),))
    # The route requires a grid; fractional-area arithmetic undercounts its cells.
    with pytest.raises(PlanningLimit, match='Grid has 250500 cells'):
        plan(sea, (100, 250), (400, 250))
    with pytest.raises(ValueError, match='250,500 cells'):
        check_grid(sea, PROFILES['model'])
    check_grid(SeaMap((0, 0, 500, 499.8)), PROFILES['model'])


def test_a_ten_times_larger_harbour_runs_identically_in_model_units(tmp_path):
    import csv
    from shipnav.export import export_run

    profile = Profile('ten', 5., 1., 2., 10.)   # L = 10 m, time scale 10 s
    real = enlarged(HARBOUR, 10)
    model = to_model(real, profile)
    assert model.bounds == HARBOUR.bounds and units(model.to_dict()) == (10., 10.)
    options = dict(policy_name='direct', seed=3, count=4, filtered=True, **profile.service_options())
    a = execute(HARBOUR.to_dict(), (2, 2), (22, 22), **options)
    b = execute(model.to_dict(), (2, 2), (22, 22), **options)
    assert a['status'] == b['status'] and a['traffic_definitions'] == b['traffic_definitions']
    for f, g in zip(a['frames'], b['frames'], strict=True):
        assert f['t'] == g['t'] and f['position'] == pytest.approx(g['position'], abs=1e-9)
    export_run(b, tmp_path/'b.csv')
    with (tmp_path/'b.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[-1]['time_s']) == pytest.approx(10*b['elapsed'])
    assert float(rows[-1]['x_m']) == pytest.approx(10*b['frames'][-1]['position'][0])
    assert float(rows[0]['executed_vx']) == pytest.approx(b['diagnostics'][0]['executed'][0])


def test_video_speedup_keeps_terminal_frame_and_caps_fps():
    from shipnav.export import video_frames

    result = {'frames': [{}]*4191, 'settings': {'dt': .25}, 'elapsed': 1047.5,
              'map': {'metadata': {'model_scale': {'length_m': 10., 'speed_mps': 5.}}}}
    indices, fps = video_frames(result, speedup=18.)
    assert indices[0] == 0 and indices[-1] == 4190 and fps <= 30
    assert len(indices)/fps == pytest.approx(1047.5*2/18, rel=.01)
