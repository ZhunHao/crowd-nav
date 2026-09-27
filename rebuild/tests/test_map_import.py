import hashlib
import json
import pytest
from shapely.geometry import box, mapping
from shipnav.maps import SeaMap
from shipnav.map_import import freeze_source, import_land


def write_source(path, geometry):
    path.write_text(json.dumps({'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'properties': {}, 'geometry': mapping(geometry)}]}))


def test_real_file_projection_clipping_provenance_and_offline_rebuild(tmp_path):
    source = tmp_path / 'source.geojson'
    write_source(source, box(103.94, 1.39, 103.96, 1.41))
    frozen = tmp_path / 'frozen.geojson'
    freeze_source(source, frozen, (103.93, 1.38, 103.97, 1.42),
                  source_url='https://example.org/test.geojson', source_timestamp='2026-01-01')
    output = tmp_path / 'map.json'
    import_land(frozen, output)
    sea = SeaMap.load(output)
    assert not sea.clear((0, 0), (0, 0), 1)
    assert sea.clear((-1800, -1800), (-1800, 1800), 1)
    manifest = json.loads(output.with_suffix('.manifest.json').read_text())
    assert manifest['source_sha256'] == hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert manifest['map_sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert manifest['upstream']['source_timestamp'] == '2026-01-01'
    assert sea.to_dict()['metadata']['layers']['depth']['status'] == 'unavailable'
    original = output.read_bytes()
    import_land(frozen, output)
    assert output.read_bytes() == original


def test_invalid_source_not_silently_repaired(tmp_path):
    from shapely.geometry import Polygon
    source = tmp_path / 'source.geojson'
    write_source(source, Polygon([(103.94, 1.39), (103.96, 1.41), (103.94, 1.41), (103.96, 1.39)]))
    with pytest.raises(ValueError):
        freeze_source(source, tmp_path/'frozen.geojson', (103.93, 1.38, 103.97, 1.42), source_url='test')


def test_source_without_frozen_crop_metadata_rejected(tmp_path):
    source = tmp_path/'source.geojson'
    write_source(source, box(103.94, 1.39, 103.96, 1.41))
    with pytest.raises(ValueError):
        import_land(source, tmp_path/'map.json')


def test_no_land_is_not_silently_accepted_as_complete_map(tmp_path):
    source = tmp_path/'source.geojson'
    write_source(source, box(103.94, 1.39, 103.96, 1.41))
    with pytest.raises(ValueError):
        freeze_source(source, tmp_path/'frozen.geojson', (104, 1.38, 104.01, 1.42), source_url='test')


def test_nested_shapefile_zip_import(tmp_path):
    import numpy as np
    from pyogrio.raw import write
    from shapely import to_wkb
    import zipfile
    directory = tmp_path/'nested'
    directory.mkdir()
    write(directory/'land.shp', np.array([to_wkb(box(103.94,1.39,103.96,1.41))]),
          [], [], driver='ESRI Shapefile', crs='EPSG:4326', geometry_type='Polygon')
    archive=tmp_path/'land.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for p in directory.iterdir():
            z.write(p,'nested/'+p.name)
    freeze_source(archive,tmp_path/'frozen.geojson',(103.93,1.38,103.97,1.42),source_url='test')
    assert json.loads((tmp_path/'frozen.geojson').read_text())['features']


def test_projection_of_long_geographic_edges_never_erases_interior_land(tmp_path):
    from shipnav.maps import LocalFrame
    source=tmp_path/'source.geojson'
    write_source(source,box(8.5,60.,9.5,60.5))
    frozen=tmp_path/'frozen.geojson'
    freeze_source(source,frozen,(8.5,59.5,9.5,60.5),source_url='test')
    sea=import_land(frozen,tmp_path/'map.json')
    frame=LocalFrame(9,60)
    for lon in (8.6,8.8,9.,9.2,9.4):
        p=frame.project(lon,60.00001)
        assert not sea.clear(p,p,0)
