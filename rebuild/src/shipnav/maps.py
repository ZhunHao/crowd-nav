"""Metric polygon maps. Vector distance is authoritative; unknown depth is blocked."""
from dataclasses import dataclass, field
import json
from math import ceil, floor, isfinite
from pathlib import Path

from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point, Polygon, MultiPolygon, box, mapping, shape
from shapely import get_coordinates, normalize, union_all
from shapely.strtree import STRtree


def canonical_json(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'


def valid_point(value):
    return len(value) == 2 and all(isfinite(v) for v in value)


@dataclass(frozen=True)
class LocalFrame:
    """A UTM projection translated to a geographic origin; API is always lon,lat."""
    lon: float
    lat: float
    _forward: Transformer = field(init=False, repr=False, compare=False)
    _reverse: Transformer = field(init=False, repr=False, compare=False)
    _origin: tuple = field(init=False, repr=False)
    epsg: int = field(init=False)

    def __post_init__(self):
        if not valid_point((self.lon, self.lat)) or not -180 <= self.lon < 180 or not -80 < self.lat < 84:
            raise ValueError('Origin must be finite WGS84 longitude/latitude within UTM coverage')
        zone = floor((self.lon + 180) / 6) + 1
        epsg = (32600 if self.lat >= 0 else 32700) + zone
        object.__setattr__(self, 'epsg', epsg)
        object.__setattr__(self, '_forward', Transformer.from_crs(4326, CRS.from_epsg(epsg), always_xy=True))
        object.__setattr__(self, '_reverse', Transformer.from_crs(CRS.from_epsg(epsg), 4326, always_xy=True))
        object.__setattr__(self, '_origin', self._forward.transform(self.lon, self.lat))

    def project(self, lon, lat):
        if not valid_point((lon, lat)) or not -180 <= lon <= 180 or not -80 < lat < 84:
            raise ValueError('Invalid WGS84 coordinate')
        x, y = self._forward.transform(lon, lat, errcheck=True)
        return x - self._origin[0], y - self._origin[1]

    def unproject(self, x, y):
        if not valid_point((x, y)):
            raise ValueError('Invalid local coordinate')
        return self._reverse.transform(x + self._origin[0], y + self._origin[1], errcheck=True)

    def to_dict(self):
        return {'origin_lonlat': [self.lon, self.lat], 'epsg': self.epsg, 'units': 'metres', 'axes': 'east,north'}


TILES_ACROSS = 48


def tile(polygons, size):
    """Split land into grid-aligned pieces for the spatial index only.

    `land` stays the canonical merged geometry. The union of the pieces equals it, so
    `clear` and `minimum_clearance` are unchanged, but each query now tests a few small
    pieces instead of a coastline with thousands of vertices."""
    pieces = []
    for polygon in polygons:
        a, b, c, d = polygon.bounds
        if c-a <= size and d-b <= size:
            pieces.append(polygon)
            continue
        for i in range(floor(a/size), ceil(c/size)):
            for j in range(floor(b/size), ceil(d/size)):
                part = polygon.intersection(box(i*size, j*size, (i+1)*size, (j+1)*size))
                pieces.extend(q for q in getattr(part, 'geoms', [part])
                              if isinstance(q, Polygon) and not q.is_empty)
    return tuple(pieces)


@dataclass(frozen=True, init=False)
class SeaMap:
    bounds: tuple
    land: tuple
    _metadata: str = field(repr=False)
    _tree: STRtree = field(repr=False, compare=False)
    _pieces: tuple = field(repr=False, compare=False)

    def __init__(self, bounds, land=(), metadata=None):
        bounds = tuple(float(v) for v in bounds)
        if len(bounds) != 4 or not all(isfinite(v) for v in bounds) or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError('Bounds require finite positive width and height')
        region = box(*bounds)
        polygons = []
        for geometry in land:
            if isinstance(geometry, (list, tuple)):
                if len(geometry) != 4 or not all(isfinite(v) for v in geometry) or geometry[0] >= geometry[2] or geometry[1] >= geometry[3]:
                    raise ValueError('Invalid rectangle')
                geometry = box(*geometry)
            if not isinstance(geometry, (Polygon, MultiPolygon)) or geometry.is_empty or not geometry.is_valid:
                raise ValueError('Land requires nonempty valid Polygon/MultiPolygon geometry')
            if not all(isfinite(v) for coordinate in get_coordinates(geometry) for v in coordinate):
                raise ValueError('Land contains nonfinite coordinates')
            if not region.covers(geometry):
                raise ValueError('Land must lie within crop bounds')
            polygons.extend(geometry.geoms if isinstance(geometry, MultiPolygon) else [geometry])
        # Canonicalize overlapping source chunks; never alter topology with buffer(0).
        merged = normalize(union_all(polygons))
        parts = [] if merged.is_empty else list(merged.geoms) if isinstance(merged, MultiPolygon) else [merged]
        parts.sort(key=lambda p: p.wkb_hex)
        object.__setattr__(self, 'bounds', bounds)
        object.__setattr__(self, 'land', tuple(parts))
        # About 48 tiles across the longer side (at least 1 unit): ~25 units on the scaled
        # Ubin map. Bounded tile count keeps loading fast at any coordinate scale.
        pieces = tile(parts, max(1., max(bounds[2]-bounds[0], bounds[3]-bounds[1])/TILES_ACROSS))
        object.__setattr__(self, '_pieces', pieces)
        object.__setattr__(self, '_tree', STRtree(pieces))
        object.__setattr__(self, '_metadata', canonical_json(metadata or {}))

    @staticmethod
    def _segment(a, b):
        return Point(a) if tuple(a) == tuple(b) else LineString([a, b])

    def minimum_clearance(self, a, b):
        """Minimum centre-path distance to land or crop edge, metres (0 if blocked)."""
        if not valid_point(a) or not valid_point(b):
            raise ValueError('Coordinates must be finite 2D points')
        x0, y0, x1, y1 = self.bounds
        edge = min(min(x-x0, x1-x, y-y0, y1-y) for x, y in (a, b))
        if edge <= 0:
            return 0.
        segment = self._segment(a, b)
        nearest = self._tree.nearest(segment)
        return min(edge, float(segment.distance(self._pieces[nearest]))) if nearest is not None else edge

    def clear(self, a, b, clearance=0., *, mode='coastline'):
        if not isfinite(clearance) or clearance < 0:
            raise ValueError('Clearance must be finite and nonnegative')
        if mode not in ('coastline', 'depth'):
            raise ValueError('Unknown navigability mode')
        # No depth adapter exists yet. Metadata alone never establishes coverage.
        if mode == 'depth':
            return False
        if not valid_point(a) or not valid_point(b):
            return False
        x0, y0, x1, y1 = self.bounds
        if any(not (x0+clearance < x < x1-clearance and y0+clearance < y < y1-clearance) for x, y in (a, b)):
            return False
        return len(self._tree.query(self._segment(a, b), predicate='dwithin', distance=clearance)) == 0

    def to_dict(self):
        # Local projected FeatureCollection, deliberately not advertised as RFC7946 WGS84.
        return {'type': 'FeatureCollection', 'schema': 1, 'bounds': list(self.bounds),
                'coordinate_system': 'local-east-north-metres', 'metadata': json.loads(self._metadata),
                'features': [{'type': 'Feature', 'properties': {'layer': 'land'}, 'geometry': mapping(p)} for p in self.land]}

    @classmethod
    def from_dict(cls, data):
        if data.get('schema') != 1 or data.get('type') != 'FeatureCollection' or data.get('coordinate_system') != 'local-east-north-metres':
            raise ValueError('Unsupported map schema or coordinate system')
        features = data['features']
        if any(f.get('type') != 'Feature' or f.get('properties', {}).get('layer') != 'land' for f in features):
            raise ValueError('Unsupported map layer')
        return cls(data['bounds'], [shape(f['geometry']) for f in features], data['metadata'])

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path):
        Path(path).write_text(canonical_json(self.to_dict()))
