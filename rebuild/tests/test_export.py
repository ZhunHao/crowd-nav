import csv
import json
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon, mapping

from shipnav.maps import SeaMap
from shipnav.service import execute
from shipnav.export import draw, export_run, land_polygons


def test_exports_use_the_recorded_run(tmp_path):
    result = execute(SeaMap((0, 0, 10, 10)).to_dict(), (2, 2), (4, 2), policy_name='direct', count=0)
    export_run(result, tmp_path/'run.json')
    export_run(result, tmp_path/'run.csv')
    export_run(result, tmp_path/'run.png')
    loaded = json.loads((tmp_path/'run.json').read_text())
    assert loaded['frames'] == result['frames']
    with (tmp_path/'run.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result['frames'])
    assert float(rows[-1]['time_s']) == result['elapsed']
    assert (tmp_path/'run.png').read_bytes().startswith(b'\x89PNG')
    with pytest.raises(ValueError):
        export_run(result, tmp_path/'run.exe')


def test_polygon_land_and_diagnostics_come_from_the_stored_trace(tmp_path):
    sea = SeaMap.load('maps/harbour.json')
    result = execute(sea.to_dict(), (2, 2), (22, 22), policy_name='direct', count=3, seed=1, filtered=True)
    stored = json.loads(json.dumps(result))  # no policy or map object survives this
    export_run(stored, tmp_path/'run.png')
    export_run(stored, tmp_path/'run.csv')
    with (tmp_path/'run.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(stored['frames']) == len(stored['diagnostics'])+1
    first, d = rows[0], stored['diagnostics'][0]
    assert float(first['executed_vx']) == d['executed'][0]
    assert first['override'] == str(d['override'])
    assert rows[-1]['executed_vx'] == ''  # terminal frame: no invented action
    assert stored == json.loads(json.dumps(result))  # export did not mutate the trace


@pytest.mark.video
@pytest.mark.skipif(shutil.which('ffprobe') is None, reason='FFmpeg not installed')
def test_mp4_keeps_every_frame_at_one_over_dt(tmp_path):
    result = execute(SeaMap((0, 0, 10, 10)).to_dict(), (2, 2), (4, 2), policy_name='direct', count=0)
    export_run(result, tmp_path/'run.mp4')
    probe = json.loads(subprocess.run(
        ['ffprobe', '-v', 'error', '-count_frames', '-show_entries',
         'stream=nb_read_frames,r_frame_rate', '-of', 'json', str(tmp_path/'run.mp4')],
        check=True, capture_output=True, text=True).stdout)['streams'][0]
    assert probe['r_frame_rate'] == '4/1'
    assert int(probe['nb_read_frames']) == len(result['frames'])


def test_polygon_render_preserves_holes_and_multipolygon_components():
    # Feed raw same-winding rings directly; SeaMap canonicalization must not normalize this fixture.
    shell = [(1, 1), (9, 1), (9, 9), (1, 9), (1, 1)]
    hole = [(3, 3), (7, 3), (7, 7), (3, 7), (3, 3)]
    polygon = Polygon(shell, [hole])
    island = Polygon([(11, 1), (14, 1), (14, 4), (11, 4), (11, 1)])
    map_data = {'bounds': [0, 0, 16, 10], 'metadata': {},
                'features': [{'geometry': mapping(MultiPolygon([polygon, island]))}]}
    assert len(list(land_polygons(map_data))) == 2
    result = {'map': map_data, 'route': [], 'goals': [], 'frames': [], 'settings': {}}
    figure = Figure(figsize=(4, 2.5), dpi=100)
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    draw(ax, result)
    figure.canvas.draw()
    image = np.asarray(figure.canvas.buffer_rgba())

    def pixel_at(point):
        x, y = ax.transData.transform(point).astype(int)
        return image[image.shape[0]-1-y, x, :3].tolist()

    land = pixel_at((2, 2))
    water_hole = pixel_at((5, 5))
    second_island = pixel_at((12, 2))
    assert water_hole == pixel_at((.5, 5))
    assert land != water_hole
    assert second_island == land


@pytest.mark.parametrize('scaled,latency,expected', [(True, 300., False), (True, 501., True),
                                                     (False, 300., True), (False, 250., False)])
def test_csv_deadline_uses_physical_control_period(scaled, latency, expected):
    from shipnav.export import csv_rows
    result = execute(SeaMap((0, 0, 10, 10)).to_dict(), (2, 2), (4, 2), policy_name='direct', count=0)
    if scaled:
        result['map']['metadata']['model_scale'] = {'length_m': 10., 'speed_mps': 5.}
    result['diagnostics'][0]['decision_ms'] = latency
    result['diagnostics'][0]['deadline_miss'] = latency > 250.
    original = json.loads(json.dumps(result))
    rows = list(csv_rows(result))
    assert rows[0]['deadline_miss'] is expected
    assert 'deadline_miss' not in rows[-1]
    assert json.loads(json.dumps(result)) == original


def test_recorded_outgoing_overlays_follow_the_selected_frame_and_terminal_has_none():
    from matplotlib.patches import FancyArrow
    stored = json.loads(json.dumps({
        'map': {'bounds': [0, 0, 10, 10], 'metadata': {}, 'features': []},
        'route': [[2, 2], [6, 2]], 'goals': [], 'settings': {'radius': .5},
        'status': 'timeout', 'traffic_definitions': [],
        'frames': [{'position': [2, 2], 'traffic': [], 't': 0},
                   {'position': [3, 2], 'traffic': [], 't': .25},
                   {'position': [4, 2], 'traffic': [], 't': .5}],
        'diagnostics': [
            {'nominal': [2, 0], 'executed': [0, 1], 'override': False,
             'no_feasible_action': None, 'predictions': [{'points': [[7, 7], [8, 7]]}],
             'path': [[2, 2], [2, 3]]},
            {'nominal': [-1, 0], 'executed': [0, -1], 'override': True,
             'no_feasible_action': False, 'predictions': [{'points': [[7, 6], [8, 6]]}],
             'path': [[3, 2], [3, 1]]}]}))
    original = json.loads(json.dumps(stored))
    figure = Figure()
    ax = figure.subplots()
    for index, arrow_tips, prediction, path, title in [
        (0, [[4.1125, 2], [2, 3.1125]], [[7, 7], [8, 7]], [[2, 2], [2, 3]], 'no feasible action=n/a'),
        (1, [[1.8875, 2], [3, .8875]], [[7, 6], [8, 6]], [[3, 2], [3, 1]], 'no feasible action=False')]:
        draw(ax, stored, index)
        arrows = {p.get_label(): p for p in ax.patches if isinstance(p, FancyArrow)}
        assert set(arrows) == {'Nominal', 'Executed'}
        # FancyArrow's first vertex is the rendered arrowhead tip, including its default head length.
        np.testing.assert_allclose(arrows['Nominal'].get_xy()[0], arrow_tips[0])
        np.testing.assert_allclose(arrows['Executed'].get_xy()[0], arrow_tips[1])
        plotted = [np.column_stack((line.get_xdata(), line.get_ydata())).tolist() for line in ax.lines]
        assert plotted[-2:] == [prediction, path]
        assert title in ax.get_title()
    draw(ax, stored, 2)
    assert not any(isinstance(p, FancyArrow) for p in ax.patches)
    assert len(ax.lines) == 2  # global route and travelled positions only
    assert 'override=' not in ax.get_title() and 'no feasible action=' not in ax.get_title()
    assert stored == original
