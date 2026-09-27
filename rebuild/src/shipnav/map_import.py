"""Freeze WGS84 coastline polygons, then rebuild an offline local metric map.

No network calls during map loading or planning. GDAL is supplied by Pyogrio's
wheel; this module does not require OSGeo bindings or a system GDAL installation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from math import isfinite, hypot, cos, pi
from pathlib import Path
import zipfile

import pyogrio
from pyogrio.raw import read
from pyproj import CRS
from shapely import from_wkb, normalize, union_all
from shapely.geometry import Polygon, MultiPolygon, box, mapping, shape

from shipnav.maps import LocalFrame, SeaMap, canonical_json


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def polygons(geometry):
    if isinstance(geometry, Polygon):
        return [] if geometry.is_empty else [geometry]
    if isinstance(geometry, MultiPolygon) or geometry.geom_type == 'GeometryCollection':
        return [p for g in geometry.geoms for p in polygons(g)]
    return []


def project_polygon(geometry, frame):
    """Approximate geographic straight edges with fine, curvature-checked chords.

    Split until span <= .001 degree and quarter-point projected deviation <=
    1 mm. Add a separate 1 cm conservative numerical envelope after projection.
    This envelope is not a bound on OSM's unknown positional accuracy.
    """
    def ring(coords):
        coords=list(coords)
        result=[frame.project(*coords[0][:2])]
        def edge(a,b,pa,pb,depth=0):
            samples=[]
            deviation=0.
            for t in (.25,.5,.75):
                point=(a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1]))
                projected=frame.project(*point)
                deviation=max(deviation,hypot(projected[0]-(pa[0]+t*(pb[0]-pa[0])),projected[1]-(pa[1]+t*(pb[1]-pa[1]))))
                samples.append((point,projected))
            if max(abs(b[0]-a[0]),abs(b[1]-a[1])) <= .001 and deviation <= .001:
                result.append(pb)
                return
            if depth >= 24:
                raise ValueError('Projection subdivision did not converge')
            mid,pmid=samples[1]
            edge(a,mid,pa,pmid,depth+1)
            edge(mid,b,pmid,pb,depth+1)
        for a,b in zip(coords,coords[1:]):
            edge(a,b,frame.project(*a[:2]),frame.project(*b[:2]))
        return result
    return Polygon(ring(geometry.exterior.coords),[ring(r.coords) for r in geometry.interiors])


def freeze_source(source, output, bbox, *, source_url, source_timestamp=None):
    """Clip a verified WGS84 polygon dataset into a small distributable snapshot."""
    bbox = tuple(float(v) for v in bbox)
    if len(bbox) != 4 or not all(isfinite(v) for v in bbox):
        raise ValueError('bbox requires four finite coordinates')
    w, s, e, n = bbox
    if not (-180 <= w < e < 180 and -80 < s < n < 84) or e-w > 1 or n-s > 1:
        raise ValueError('Use a local, non-antimeridian crop no larger than one degree')
    source = Path(source).resolve()
    dataset = str(source)
    if source.suffix.lower() == '.zip':
        with zipfile.ZipFile(source) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith('.shp')]
        if len(names) != 1:
            raise ValueError('Archive must contain exactly one shapefile')
        dataset = f'/vsizip/{source}/{names[0]}'
    info = pyogrio.read_info(dataset)
    if not info['crs'] or not CRS.from_user_input(info['crs']).equals(CRS.from_epsg(4326), ignore_axis_order=True):
        raise ValueError('Source must be WGS84 longitude/latitude')
    _, _, geometries, _ = read(dataset, bbox=bbox, columns=[])
    clip = box(*bbox)
    parts = []
    for wkb in geometries:
        geometry = from_wkb(wkb)
        if geometry is None or not isinstance(geometry, (Polygon, MultiPolygon)) or not geometry.is_valid:
            raise ValueError('Invalid source polygon; no automatic repair is permitted')
        parts.extend(polygons(geometry.intersection(clip)))
    if not parts:
        raise ValueError('No land found; verify source coverage and crop before declaring open water')
    merged = normalize(union_all(parts))
    parts = sorted(polygons(merged), key=lambda p: p.wkb_hex)
    data = {'type': 'FeatureCollection', 'shipnav_source_schema': 1, 'bbox': list(bbox),
        'provenance': {'source_url': source_url, 'source_sha256': sha256(source),
            'source_timestamp': source_timestamp, 'retrieved_at': datetime.now(timezone.utc).isoformat(),
            'source_crs': 'EPSG:4326', 'licence': 'ODbL-1.0',
            'attribution': '© OpenStreetMap contributors',
            'licence_url': 'https://www.openstreetmap.org/copyright',
            'processing': 'bbox clip and union of overlapping chunks; no simplification or topology repair'},
        'features': [{'type': 'Feature', 'properties': {'layer': 'land'}, 'geometry': mapping(p)} for p in parts]}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(data))
    return data


def import_land(source, output):
    """Project a frozen source snapshot; identical source + versions => identical bytes."""
    source, output = Path(source), Path(output)
    data = json.loads(source.read_text())
    if data.get('shipnav_source_schema') != 1 or not data.get('provenance') or not data.get('bbox'):
        raise ValueError('Input must be a frozen ShipNav WGS84 coastline snapshot')
    w, s, e, n = data['bbox']
    if not all(isfinite(v) for v in (w, s, e, n)) or not (-180 <= w < e < 180 and -80 < s < n < 84) or e-w > 1 or n-s > 1:
        raise ValueError('Invalid source crop')
    frame = LocalFrame((w+e)/2, (s+n)/2)
    # Conservative inset rectangle inside a densified projected source coverage.
    steps = [i/128 for i in range(129)]
    west = [frame.project(w, s+t*(n-s)) for t in steps]
    east = [frame.project(e, s+t*(n-s)) for t in steps]
    south = [frame.project(w+t*(e-w), s) for t in steps]
    north = [frame.project(w+t*(e-w), n) for t in steps]
    bounds = (max(x for x,y in west)+.1, max(y for x,y in south)+.1,
              min(x for x,y in east)-.1, min(y for x,y in north)-.1)
    coverage = Polygon(south + east[1:] + list(reversed(north))[1:] + list(reversed(west))[1:])
    region = box(*bounds)
    if not coverage.covers(region):
        raise ValueError('Projected rectangle extends beyond verified crop coverage')
    parts = []
    for feature in data['features']:
        geometry = shape(feature['geometry'])
        if not isinstance(geometry, (Polygon, MultiPolygon)) or not geometry.is_valid or not box(w,s,e,n).covers(geometry):
            raise ValueError('Invalid or out-of-coverage frozen geometry')
        for polygon in polygons(geometry):
            projected = project_polygon(polygon,frame)
            # Compensate for GEOS round-buffer chords so its envelope is >= 1 cm.
            projected = projected.buffer(.01/cos(pi/64),quad_segs=16)
            parts.extend(polygons(projected.intersection(region)))
    if not parts:
        raise ValueError('Projected crop has no land')
    metadata = {'frame': frame.to_dict(), 'geographic_bbox': [w,s,e,n],
        'source_sha256': sha256(source), 'upstream': data['provenance'],
        'layers': {'land': {'status': 'available', 'quality': 'OSM coastline; positional accuracy unquantified'},
                   **{name: {'status': 'unavailable'} for name in ('depth','restrictions','currents','water_level','fixed_obstacles')}},
        'navigability': 'coastline-only; water depth and restrictions unknown',
        'processing': {'projection': 'UTM with local origin', 'coverage_inset_m': .1, 'simplification_m': 0,
                       'edge_max_span_degrees': .001, 'edge_quarter_deviation_m': .001, 'numerical_land_envelope_m': .01,
                       'versions': {name: version(name) for name in ('shapely','pyproj','pyogrio')}}}
    sea = SeaMap(bounds, parts, metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    sea.save(output)
    manifest = {'schema': 1, 'source_file': source.name, 'source_sha256': sha256(source),
        'map_file': output.name, 'map_sha256': sha256(output), 'upstream': data['provenance'],
        'frame': frame.to_dict(), 'bounds_m': list(bounds), 'land_polygons': len(sea.land),
        'processing': metadata['processing']}
    output.with_suffix('.manifest.json').write_text(canonical_json(manifest))
    return sea


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    freeze = sub.add_parser('freeze', help='Freeze a WGS84 land-polygon dataset')
    freeze.add_argument('source', type=Path)
    freeze.add_argument('output', type=Path)
    freeze.add_argument('--bbox', nargs=4, type=float, required=True, metavar=('WEST','SOUTH','EAST','NORTH'))
    freeze.add_argument('--source-url', required=True)
    freeze.add_argument('--source-timestamp')
    build = sub.add_parser('build', help='Rebuild offline from a frozen WGS84 snapshot')
    build.add_argument('source', type=Path)
    build.add_argument('output', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'freeze':
            freeze_source(args.source, args.output, args.bbox, source_url=args.source_url, source_timestamp=args.source_timestamp)
        else:
            import_land(args.source, args.output)
    except (ValueError, OSError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
