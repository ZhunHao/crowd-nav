import json
import math

import pytest
from shapely.geometry import Polygon, box

from shipnav.maps import LocalFrame, SeaMap


def test_swept_clearance_and_strict_boundary():
    sea = SeaMap((0, 0, 10, 10), [box(4, 4, 6, 6)])
    assert sea.clear((2, 2), (8, 2), .5)
    assert not sea.clear((2, 5), (8, 5), .5)
    assert not sea.clear((3.5, 3), (3.5, 7), .5)
    assert not sea.clear((.5, 2), (2, 2), .5)
    assert sea.minimum_clearance((2, 2), (8, 2)) == pytest.approx(2.)


def test_holes_and_concave_coasts_are_not_bounding_boxes():
    island = Polygon([(2, 2), (8, 2), (8, 8), (2, 8)],
                     holes=[[(3, 3), (7, 3), (7, 7), (3, 7)]])
    sea = SeaMap((0, 0, 10, 10), [island])
    assert sea.clear((4, 4), (6, 6), .5)
    assert not sea.clear((1, 5), (5, 5), .1)
    concave = Polygon([(2, 2), (8, 2), (8, 4), (4, 4), (4, 8), (2, 8)])
    assert SeaMap((0, 0, 10, 10), [concave]).clear((5, 5), (8, 8), .5)


def test_unknown_depth_never_becomes_navigable():
    sea = SeaMap((0, 0, 10, 10))
    assert sea.clear((2, 2), (8, 8), .5)
    assert not sea.clear((2, 2), (8, 8), .5, mode='depth')
    with pytest.raises(ValueError):
        sea.clear((2, 2), (8, 8), .5, mode='typo')


@pytest.mark.parametrize('value', [-1, math.nan, math.inf])
def test_invalid_clearance_rejected(value):
    with pytest.raises(ValueError):
        SeaMap((0, 0, 10, 10)).clear((2, 2), (3, 3), value)


def test_invalid_geometry_and_coordinates_rejected():
    with pytest.raises(ValueError):
        SeaMap((0, 0, 10, 10), [Polygon([(2, 2), (8, 8), (2, 8), (8, 2)])])
    with pytest.raises(ValueError):
        SeaMap((0, 0, math.inf, 10))
    with pytest.raises(ValueError):
        SeaMap((0, 0, 10, 10), [box(-1, 2, 3, 4)])
    assert not SeaMap((0, 0, 10, 10)).clear((math.nan, 2), (3, 3), 0)


def test_roundtrip_and_metadata_cannot_mutate_map(tmp_path):
    metadata = {'layers': {'depth': {'status': 'unavailable'}}}
    sea = SeaMap((0, 0, 10, 10), [(4, 4, 6, 6)], metadata=metadata)
    metadata['layers']['depth']['status'] = 'available'
    path = tmp_path / 'map.json'
    sea.save(path)
    loaded = SeaMap.load(path)
    assert loaded.to_dict() == sea.to_dict()
    assert not loaded.clear((2, 5), (8, 5), .5)
    assert loaded.to_dict()['metadata']['layers']['depth']['status'] == 'unavailable'
    data = json.loads(path.read_text())
    data['schema'] = 999
    with pytest.raises(ValueError):
        SeaMap.from_dict(data)


def test_projection_roundtrip_axis_and_metric_scale():
    frame = LocalFrame(103.95, 1.4)
    assert frame.project(103.95, 1.4) == pytest.approx((0, 0), abs=1e-6)
    east = frame.project(103.951, 1.4)
    north = frame.project(103.95, 1.401)
    assert 110 < east[0] < 112 and abs(east[1]) < 1
    assert 110 < north[1] < 112 and abs(north[0]) < 1
    assert frame.unproject(*frame.project(103.98, 1.42)) == pytest.approx((103.98, 1.42), abs=1e-9)
    with pytest.raises(ValueError):
        frame.project(103, math.nan)
