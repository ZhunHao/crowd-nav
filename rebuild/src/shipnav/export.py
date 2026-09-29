"""Draw and export a recorded `execute()` result. Never re-runs a policy.

Results are stored in model units. Maps scaled by `shipnav.scale.to_model` carry
`model_scale` metadata; drawing, CSV and video convert to metres and seconds here.
"""
from math import ceil
from pathlib import Path
import csv
import json

from matplotlib.animation import FFMpegWriter, writers
from matplotlib.figure import Figure
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FuncFormatter
from shapely.geometry import shape
from shapely.geometry.polygon import orient

from shipnav.scale import units


CSV_FIELDS = ['time_s', 'x_m', 'y_m', 'vx_mps', 'vy_mps', 'goal_index',
              'nominal_vx', 'nominal_vy', 'executed_vx', 'executed_vy', 'actual_vx', 'actual_vy',
              'override', 'no_feasible_action', 'decision_ms', 'deadline_miss',
              'max_target_age_s', 'ship_collision', 'land_collision', 'clearance_m']


def land_polygons(map_data: dict):
    """Yield every land Polygon of a schema-1 map, splitting MultiPolygons."""
    for feature in map_data['features']:
        geometry = shape(feature['geometry'])
        yield from getattr(geometry, 'geoms', [geometry])


def _patch(polygon, **style):
    """Create a compound path with consistent winding for polygon holes."""
    polygon = orient(polygon, sign=1.0)
    vertices, codes = [], []
    for ring in (polygon.exterior, *polygon.interiors):
        points = list(ring.coords)
        vertices += points
        codes += [MplPath.MOVETO] + [MplPath.LINETO]*(len(points)-2) + [MplPath.CLOSEPOLY]
    return PathPatch(MplPath(vertices, codes), **style)


def draw(ax, result: dict, frame_index: int = -1) -> None:
    """Draw a stored trace at one frame, including only that frame's diagnostics."""
    ax.clear()
    x0, y0, x1, y1 = result['map']['bounds']
    length_scale, time_scale = units(result['map'])
    ax.set(xlim=(x0, x1), ylim=(y0, y1), xlabel='x (m)', ylabel='y (m)', aspect='equal')
    metres = FuncFormatter(lambda v, _: f'{v*length_scale:g}')
    ax.xaxis.set_major_formatter(metres)
    ax.yaxis.set_major_formatter(metres)
    ax.set_facecolor('#dceef5')
    for polygon in land_polygons(result['map']):
        ax.add_patch(_patch(polygon, facecolor='#9b987f', edgecolor='none'))
    route = result['route']
    if route:
        ax.plot(*zip(*route), '--', color='#555555', label='Global route')
    goals = result['goals']
    if goals:
        ax.scatter(*zip(*goals), marker='*', color='#cf4b35', label='Goals', zorder=3)
    frames = result['frames']
    if frames:
        i = len(frames)-1 if frame_index < 0 else min(frame_index, len(frames)-1)
        frame = frames[i]
        ax.plot(*zip(*(f['position'] for f in frames[:i+1])), color='#17659c', label='Travelled')
        ax.add_patch(Circle(frame['position'], result['settings']['radius'], color='#17659c'))
        for position, ship in zip(frame['traffic'], result['traffic_definitions']):
            ax.add_patch(Circle(position, ship['radius'], color='#df9a36'))
        # True-size discs vanish on kilometre maps; fixed-size markers keep vessels visible.
        ax.scatter(*frame['position'], s=30, color='#17659c', edgecolors='white', zorder=4, label='Own ship')
        if frame['traffic']:
            ax.scatter(*zip(*frame['traffic']), s=30, color='#df9a36', edgecolors='white', zorder=4, label='Traffic')
        title = f"{result['status']} — {frame['t']*time_scale:.1f} s"
        diagnostics = result.get('diagnostics', [])
        # Diagnostic k is the decision from frame k to k+1: terminal frame has none,
        # and an earlier frame never shows a later tick's prediction.
        if i < len(diagnostics):
            d = diagnostics[i]
            for key, color in (('nominal', '#b05090'), ('executed', '#248060')):
                ax.arrow(*frame['position'], *d[key], color=color, width=.025, label=key.capitalize())
            for target in d['predictions']:
                ax.plot(*zip(*target['points']), ':', color='#e5a02e')
            if d['path']:
                ax.plot(*zip(*d['path']), '-', color='#248060', alpha=.5)
            infeasible = 'n/a' if d['no_feasible_action'] is None else d['no_feasible_action']
            title += f" | override={d['override']} | no feasible action={infeasible}"
        ax.set_title(title)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='upper left')


def csv_rows(result: dict):
    """Yield a row per frame; action columns describe the step leaving that frame.

    The terminal frame has no action, so its action columns stay empty.
    """
    diagnostics = result.get('diagnostics', [])
    length_scale, time_scale = units(result['map'])
    velocity_scale = length_scale/time_scale
    for i, frame in enumerate(result['frames']):
        row = {'time_s': frame['t']*time_scale,
               'x_m': frame['position'][0]*length_scale, 'y_m': frame['position'][1]*length_scale,
               'vx_mps': frame['velocity'][0]*velocity_scale, 'vy_mps': frame['velocity'][1]*velocity_scale,
               'goal_index': frame['goal_index']}
        if i < len(diagnostics):
            diagnostic = diagnostics[i]
            ages = [ship['age'] for ship in diagnostic['observed']]
            row.update(nominal_vx=diagnostic['nominal'][0]*velocity_scale,
                       nominal_vy=diagnostic['nominal'][1]*velocity_scale,
                       executed_vx=diagnostic['executed'][0]*velocity_scale,
                       executed_vy=diagnostic['executed'][1]*velocity_scale,
                       actual_vx=diagnostic['actual_velocity'][0]*velocity_scale,
                       actual_vy=diagnostic['actual_velocity'][1]*velocity_scale,
                       override=diagnostic['override'], no_feasible_action=diagnostic['no_feasible_action'],
                       decision_ms=diagnostic['decision_ms'],
                       deadline_miss=diagnostic['decision_ms'] > 1000*result['settings']['dt']*time_scale,
                       max_target_age_s=max(ages)*time_scale if ages else None,
                       ship_collision=diagnostic['ship_collision'], land_collision=diagnostic['land_collision'],
                       clearance_m=None if diagnostic['clearance'] is None else diagnostic['clearance']*length_scale)
        yield row


def video_frames(result: dict, speedup: float = 1., max_fps: float = 30.):
    """Return frame indices and FPS for simulated time / speedup, capped at max_fps."""
    if speedup <= 0:
        raise ValueError('Speed-up must be positive')
    n = len(result['frames'])
    if n == 0:
        raise ValueError('No frames to export')
    _, time_scale = units(result['map'])
    rate = speedup/(result['settings']['dt']*time_scale)
    stride = max(1, ceil(rate/max_fps - 1e-9))
    indices = list(range(0, n, stride))
    if indices[-1] != n-1:
        indices.append(n-1)
    return indices, rate/stride


def default_speedup(result: dict, target_s: float = 120.) -> float:
    """Use 1 for short runs; longer runs target about `target_s` seconds of video."""
    _, time_scale = units(result['map'])
    return float(max(1, ceil(result['elapsed']*time_scale/target_s)))


def export_run(result: dict, path: Path, speedup: float | None = None) -> None:
    """Export a recorded run as JSON, CSV, PNG overview or MP4 replay."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in ('.json', '.csv', '.png', '.mp4'):
        raise ValueError('Choose JSON, CSV, PNG or MP4')
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == '.json':
        path.write_text(json.dumps(result, indent=2, allow_nan=False))
        return
    if suffix == '.csv':
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(csv_rows(result))
        return
    figure = Figure(figsize=(7, 7))
    ax = figure.subplots()
    if suffix == '.png':
        draw(ax, result)
        figure.savefig(path, dpi=150)
        return
    if not writers.is_available('ffmpeg'):
        raise RuntimeError('FFmpeg is required to export MP4')
    if not result['frames']:
        raise ValueError('No frames to export')
    indices, fps = video_frames(result, default_speedup(result) if speedup is None else speedup)
    writer = FFMpegWriter(fps=fps, codec='libx264')
    with writer.saving(figure, str(path), dpi=100):
        for index in indices:
            draw(ax, result, index)
            writer.grab_frame()
