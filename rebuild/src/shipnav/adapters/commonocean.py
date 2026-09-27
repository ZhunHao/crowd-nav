"""CommonOcean adapter: canonical-schema import and the upstream collision
oracle, written against the ACTUALLY INSTALLED commonocean-io /
commonocean-drivability-checker API (verified in
`migration/external/commonocean/`, see `decision.md` there for the
install/verification record).

Scope and honesty notes (see task D4 rulings):
  - `import_scenario` needs only `commonocean-io`, which installed cleanly
    (pure-Python wheel) in the disposable external environment and was used
    to read real upstream scenario files.
  - `check_collision`/`check_trajectory` need `commonocean_dc` (the
    drivability checker), whose native build failed in this environment
    (missing system Boost headers for its bundled C++ collision library;
    see `migration/external/commonocean/install.log` and `install2.log`).
    Both functions are still written here against the documented/tutorial
    API (verified against the official CommonOcean interface tutorial) so
    the code is honest and reviewable, but they raise ImportError at call
    time in any environment without `commonocean_dc` installed -- they are
    NOT stubbed or faked. `check_trajectory` reports the collision result as
    explicitly "not checked" (with a reason) rather than pretending success.
    Water-boundary and model-feasibility checks are not implemented at all
    and are always reported as "not checked", never as passed.
"""
import os

from shipnav.adapters.coordinates import conservative_radius


def _shape_dims(shape):
    return getattr(shape, 'length', None), getattr(shape, 'width', None)


def _obstacle_states(obstacle):
    """Return the full time-indexed state list for one obstacle: its initial
    state, plus every state of its trajectory prediction. Rejects (raises)
    obstacles whose prediction is not a fixed TrajectoryPrediction -- e.g. a
    SetBasedPrediction, whose occupancy is a time-varying set of shapes
    rather than a single time-indexed trajectory -- instead of silently
    dropping that geometry.
    """
    from commonocean.prediction.prediction import TrajectoryPrediction

    states = [obstacle.initial_state]
    prediction = getattr(obstacle, 'prediction', None)
    if prediction is None:
        return states
    if not isinstance(prediction, TrajectoryPrediction):
        raise NotImplementedError(
            f'Obstacle {obstacle.obstacle_id} uses {type(prediction).__name__}; '
            'only TrajectoryPrediction (a fixed, time-indexed trajectory) is '
            'supported by this adapter. Time-varying/set-based occupancy '
            'geometry is rejected, not silently dropped.')
    states.extend(prediction.trajectory.state_list)
    return states


def _export_static_obstacle(obstacle):
    shape = obstacle.obstacle_shape
    vertices = getattr(shape, 'vertices', None)
    entry = {'obstacle_id': obstacle.obstacle_id, 'obstacle_type': str(obstacle.obstacle_type.value)}
    if vertices is not None:
        polygon = [[float(x), float(y)] for x, y in vertices]
        xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
        entry.update(
            exact_polygon=polygon,
            conservative_rectangle_approximation={
                'label': 'axis-aligned bounding box of exact_polygon; conservative, not exact',
                'bounds_xyxy': [min(xs), min(ys), max(xs), max(ys)],
            },
        )
        return entry
    length, width = _shape_dims(shape)
    if length is None or width is None:
        raise NotImplementedError(
            f'Static obstacle {obstacle.obstacle_id} has shape '
            f'{type(shape).__name__} with neither vertices nor length/width; '
            'unsupported shape, not silently dropped.')
    entry.update(center=[float(v) for v in shape.center], length=float(length), width=float(width))
    return entry


def _export_waters(scenario):
    return [
        {
            'waters_id': w.waters_id,
            'waters_type': str(w.waters_type.value),
            'left_vertices': [[float(x), float(y)] for x, y in w.left_vertices],
            'right_vertices': [[float(x), float(y)] for x, y in w.right_vertices],
            'center_vertices': [[float(x), float(y)] for x, y in w.center_vertices],
        }
        for w in scenario._waters_network.waters
    ]


def import_scenario(path):
    """Import one CommonOcean XML scenario into ShipNav's canonical schema.

    CommonOcean scenario coordinates are already planar Cartesian (metres).
    Per task D4, `adapters.coordinates.north_east_to_xy` is a North-East (as
    used by e.g. NTNU) to East-North conversion and is deliberately NOT
    applied here: CommonOcean is not a North-East source, and its own CRS is
    frequently unverified (see `external_metadata.coordinate_transform`
    below, which records this honestly rather than guessing an EPSG code).
    """
    from commonocean.common.file_reader import CommonOceanFileReader

    scenario, planning_problem_set = CommonOceanFileReader(str(path)).open()
    if not planning_problem_set.planning_problem_dict:
        raise ValueError(f'{path}: scenario has no planning problem')
    planning_problem = next(iter(planning_problem_set.planning_problem_dict.values()))

    start = [float(v) for v in planning_problem.initial_state.position]
    goal_shape = planning_problem.goal.state_list[0].position
    goal_center = getattr(goal_shape, 'center', None)
    if goal_center is None:
        raise NotImplementedError(
            f'{path}: goal region shape {type(goal_shape).__name__} has no '
            'single centre point; only point/rectangle/circle goal regions '
            'are supported by this adapter.')
    goal = [float(v) for v in goal_center]

    traffic = []
    for obstacle in scenario.dynamic_obstacles:
        states = _obstacle_states(obstacle)
        length, width = _shape_dims(obstacle.obstacle_shape)
        if length is None or width is None or min(length, width) <= 0:
            raise NotImplementedError(
                f'Dynamic obstacle {obstacle.obstacle_id} has shape '
                f'{type(obstacle.obstacle_shape).__name__} without a positive '
                'length/width; only rectangular hull approximations are '
                'supported here.')
        traffic.append({
            'obstacle_id': obstacle.obstacle_id,
            'obstacle_type': str(obstacle.obstacle_type.value),
            'start': [float(v) for v in states[0].position],
            'goal': [float(v) for v in states[-1].position],
            'speed': float(getattr(states[0], 'velocity', 0.0)),
            'radius': conservative_radius(length, width),
        })

    location = getattr(scenario, 'location', None)
    reference_location = None
    if location is not None:
        reference_location = {'gps_latitude': location.gps_latitude, 'gps_longitude': location.gps_longitude}
        geo_transformation = getattr(location, 'geo_transformation', None)
        if geo_transformation is not None:
            raise NotImplementedError(
                'Scenario declares a geo_transformation; this adapter has only '
                'been verified against scenarios with no declared CRS and does '
                'not honestly know how to interpret it yet.')

    return {
        'schema': 1,
        'seed': 0,
        'map': {'type': 'FeatureCollection', 'coordinate_system': 'external-commonocean-cartesian-unverified-crs',
                'features': []},
        'start': start,
        'goal': goal,
        'traffic': traffic,
        'traffic_mode': 'external_replay',
        'family': 'external_commonocean',
        'split': 'external',
        'external_metadata': {
            'source': 'commonocean-io==2025.1',
            'benchmark_id': str(scenario.scenario_id),
            'scenario_version': scenario.scenario_id.scenario_version,
            'dt_seconds': float(scenario.dt),
            'file': os.path.abspath(str(path)),
            'reference_location': reference_location,
            'original_waters_polygons': _export_waters(scenario),
            'original_static_obstacles': [_export_static_obstacle(o) for o in scenario.static_obstacles],
            'actor_shape_dynamics': traffic,
            'control_limits': {
                'note': (
                    'Not embedded per-actor in the CommonOcean scenario XML: '
                    'no obstacle-to-vessel-model mapping exists in this file. '
                    'The separately installed commonocean-vessel-models==1.0.0 '
                    'package provides generic (non-scenario-specific) vessel '
                    'parameter sets; example verified-by-import values from '
                    'vesselmodels.parameters_vessel_1: v_max=16.8 m/s, '
                    'a_max=0.24 m/s^2, w_max=0.03 rad/s. These are cited for '
                    'reference only, not attributed to any actor in this scenario.'
                ),
            },
            'coordinate_transform': (
                'None applied. CommonOcean scenario coordinates are already '
                'planar Cartesian (metres). adapters.coordinates.north_east_to_xy '
                'is a North-East-to-East-North transform for sources such as '
                'NTNU and is not used here. This scenario file declares no '
                'EPSG/CRS (location.geo_transformation is None); only an '
                'approximate WGS84 reference location is given, if present.'
            ),
        },
    }


def check_collision(original_path, result, length, width):
    """Independent upstream collision oracle for a resampled ego trajectory,
    per the task brief's verified starting point (matches the official
    CommonOcean interface tutorial). Requires `commonocean_dc`
    (commonocean-drivability-checker) to be installed; raises ImportError
    otherwise rather than faking a result.
    """
    import numpy as np
    from commonocean.common.file_reader import CommonOceanFileReader
    from commonocean.scenario.state import GeneralState
    from commonocean.scenario.trajectory import Trajectory
    from commonocean.prediction.prediction import TrajectoryPrediction
    from commonroad.geometry.shape import Rectangle
    from commonocean_dc.collision.collision_detection.pycrcc_collision_dispatch import (
        create_collision_checker, create_collision_object)

    scenario, _ = CommonOceanFileReader(str(original_path)).open()
    dt = float(scenario.dt)
    frames = result['frames']
    diagnostics = result['diagnostics']
    states = []
    for k, frame in enumerate(frames):
        if abs(frame['t'] - k * dt) > 1e-6:
            raise ValueError('Resample to the upstream timestep before checking')
        orientation = 0. if k == 0 else diagnostics[k - 1]['heading']
        states.append(GeneralState(time_step=k, position=np.asarray(frame['position']), orientation=orientation))
    predicted = TrajectoryPrediction(Trajectory(0, states), Rectangle(length=length, width=width))
    return {'discrete_collision': bool(create_collision_checker(scenario).collide(create_collision_object(predicted))),
            'continuous_collision_checked': False}


def check_trajectory(original_path, result, length, width):
    """Combine the independent upstream checks for one resampled ego
    trajectory. Only the collision component is implemented (per
    `check_collision`); water-boundary and model-feasibility are named
    explicitly and reported as not checked, never as passed. Full COLREGs
    legality is outside the meaning of this or any feasibility checker.
    """
    report = {
        'collision': {'checked': False, 'result': None, 'reason': None},
        'water_boundary': {'checked': False, 'result': None,
                            'reason': 'water-boundary feasibility is not implemented by this adapter'},
        'model_feasibility': {'checked': False, 'result': None,
                               'reason': 'kinematic/model feasibility is not implemented by this adapter'},
    }
    try:
        report['collision']['result'] = check_collision(original_path, result, length, width)
        report['collision']['checked'] = True
    except ImportError as exc:
        report['collision']['reason'] = f'commonocean_dc unavailable: {exc}'
    return report
