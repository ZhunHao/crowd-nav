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
from shapely.geometry import MultiPolygon, Polygon

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
    # Reverse the exterior/hole orientation to prove rendering normalizes rings.
    shell = [(1, 1), (9, 1), (9, 9), (1, 9), (1, 1)]
    hole = [(3, 3), (7, 3), (7, 7), (3, 7), (3, 3)]
    polygon = Polygon(shell[::-1], [hole])
    island = Polygon([(11, 1), (14, 1), (14, 4), (11, 4), (11, 1)])
    map_data = SeaMap((0, 0, 16, 10), [MultiPolygon([polygon, island])]).to_dict()
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
    assert land != water_hole
    assert second_island == land
