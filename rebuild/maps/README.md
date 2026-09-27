# Real Singapore coastline maps

Two offline crops derived from OpenStreetMap coastline polygons:

| Map | Geographic extent (west, south, east, north) | Purpose |
| --- | --- | --- |
| `singapore-ubin.json` | 103.91, 1.36, 104.02, 1.435 | Pulau Ubin / Changi development |
| `singapore-southern-islands.json` | 103.82, 1.20, 103.885, 1.265 | Separate held-out geographic smoke tests |

**Coastline-only simulation.** Water colour means outside mapped land; depth, navigation restrictions, tides, currents and isolated fixed obstacles are unknown. These maps are not nautical charts. The 5 m demo radius and 10 m planning margin are experimental settings, not calibrated bounds on OSM accuracy. Enclose the actual vessel footprint with an appropriate radius before changing vessel dimensions.

## Provenance and licence

© OpenStreetMap contributors. The frozen WGS84 data and derived local map databases are available under the **Open Database Licence (ODbL) 1.0**. Retain attribution and licence information when sharing the map or preview. See [OSM copyright and licence](https://www.openstreetmap.org/copyright) and [ODbL legal text](https://opendatacommons.org/licenses/odbl/1-0/).

Source: [OSM land polygons](https://osmdata.openstreetmap.de/data/land-polygons.html), specifically `land-polygons-split-4326.zip`. Source data timestamp from the archive README: **2026-09-27T00:00:00Z**. Package HTTP Last-Modified: **2026-09-27T03:38:43Z**. These are separate from the local retrieval timestamp. Original archive hashes are embedded in the frozen source provenance.

`sources/*-wgs84.geojson` are ordinary WGS84 GeoJSON snapshots, clipped to each region and retaining upstream provenance. They are sufficient for an offline rebuild; the approximately 928 MB global archive is an ignored cache, not an application dependency.

`singapore-*.json` use ShipNav schema 1: a FeatureCollection-like structure in **local East-North metres**, with an explicit coordinate-system field. They are deliberately named `.json`, not advertised as RFC 7946 GeoJSON. Use the WGS84 snapshots with general GIS applications. `*.manifest.json` records source/output hashes, projection, dimensions and processing settings.

## Offline reproduction

Run from `rebuild/` after installing the `maps` extra:

```sh
UV_PROJECT_ENVIRONMENT=.venv-modern uv sync --locked --all-extras --python 3.14.7
.venv-modern/bin/python tools/install_rvo2.py
.venv-modern/bin/python -m shipnav.map_import build maps/sources/ubin-wgs84.geojson maps/singapore-ubin.json
.venv-modern/bin/python -m shipnav.map_import build maps/sources/southern-islands-wgs84.geojson maps/singapore-southern-islands.json
```

With the locked library versions these commands reproduce the map bytes and hashes. Raw source rings are preserved in the WGS84 snapshots. Projection adaptively subdivides long geographic edges (maximum .001 degree span and 1 mm quarter-point chord deviation), then adds a separate **1 cm numerical land envelope** before clipping. This prevents projected chords from erasing interior land. It does not resolve unknown source positional error. No coastline simplification or topology repair is performed by the importer; invalid source geometry fails explicitly. Upstream OSM polygon processing may itself repair coastline errors, as documented by the source provider.

The local metric rectangle is inset within projected source coverage. Polygon holes are preserved. Overlapping upstream split chunks are unioned. Queries use a spatial index and continuous vector distances; touching inflated land or the inset boundary is blocked. `SeaMap.clear(..., mode='depth')` returns false because no depth data adapter is implemented.

## Plan and view a route

```sh
.venv-modern/bin/python -m shipnav.map_demo route maps/singapore-ubin.json \
  --lonlat --start 103.925 1.41 --goal 104.005 1.425 \
  --radius 5 --margin 10 --resolution 50 --planner theta \
  --output results/maps/ubin-route
```

Outputs `ubin-route.json` and `ubin-route.png`. Without `--lonlat`, endpoints are local East-North metres. Route JSON includes the map hash, configuration, expansions, route length, runtime and continuously checked clearance. Invalid endpoints and compute-budget failures produce explicit non-success status and exit code 2.

Routes are geometric disk-centre paths. Abrupt waypoint turns are not a vessel-feasibility claim. Dynamics-aware tracking, oriented swept hulls and traffic avoidance belong to subsequent phases.

## Paired comparison

```sh
.venv-modern/bin/python -m shipnav.map_demo benchmark maps/benchmarks.json \
  --repeats 3 --output results/maps/benchmark.json
```

The frozen manifest contains endpoint pairs, map hashes, settings and development/test labels. Both A* plus smoothing and Basic Theta* use the same grid, endpoint connectors, clearance and compute budget; Theta* changes parent relaxation during search. Planner order alternates. Every accepted route is revalidated against the polygons.

Results preserve failures and report success-conditioned length alongside outcome counts, p50/p95/p99 wall latency, expansions and a separate Python-allocation memory probe. Timing excludes map loading; memory excludes native GEOS/GDAL allocations, so it is not process RSS. A small three-repeat smoke suite cannot establish statistically reliable tail latency or generalization. No state-of-the-art performance claim is made. No-path means failure on this graph, not proof of continuous-space infeasibility.

Synthetic regression tests cover disconnected water, boundary contact, holes, concavity, invalid topology, projection, unknown depth, deadlines and serialization. Actual depth-constrained routing and semantic chart layers remain explicitly unavailable.
