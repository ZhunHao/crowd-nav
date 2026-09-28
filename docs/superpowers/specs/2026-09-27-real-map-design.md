# Real geographic maps for ShipNav — research-grade draft for review

## Objective

Build an offline, reproducible real-coastline map and clearance-aware route demonstration in `rebuild/`, compatible with the accepted Python 3.14 stack. Initial proposed area: Pulau Ubin and the Changi coast, Singapore. The precise crop will be recorded in the map manifest once the source geometry is inspected.

This is a coastline/land-avoidance simulation map. Depth, tides, restricted areas, navigation aids, live AIS and navigability certification are outside this first increment.

## Quality target: evidence before a SOTA claim

The user requested state-of-the-art quality. Treat that as a target for data integrity, geometric correctness, scale and measured planning performance. Do not label the implementation state of the art merely because it uses recent packages or combines named algorithms. Demonstrate improvements against declared baselines on held-out geography; report failures and trade-offs.

Keep the first implementation focused on a map subsystem. Its interfaces must support richer maritime layers without requiring those unavailable datasets to be invented. Vessel dynamics, uncertainty-aware local control and traffic-rule reasoning remain separately validated later phases.

## Layered map model

Represent land, fixed obstacles, depth coverage, restricted regions and environmental fields as separate typed layers. Every layer carries provenance, timestamp, horizontal CRS, vertical datum where relevant, coverage and quality metadata. Optional layers are explicitly unavailable until populated from a verified source. Distinguish known water geometry from unknown depth and from navigable water for a specified vessel.

Provide two explicit query modes: coastline-only (no grounding claim) and depth-constrained (requires depth coverage and compatible datum metadata). Missing depth must produce unknown/blocked results in the latter. Future under-keel checks must account for draft, water level, uncertainty and an explicit clearance policy; do not combine unrelated vertical datums.

Follow the separation of concerns in IHO S-101 charts, S-102 bathymetry, S-104 water levels and S-111 currents. This is an internal data model informed by these standards, not a claim of S-100 format support or certification. Add actual file adapters only when test data and validation are available.

## Alternatives

1. Recommended: preprocessed OpenStreetMap land polygons, clipped to the selected region. This supplies usable land geometry without assembling raw coastline ways.
2. Regional raw OSM extraction: provides more feature types, but requires coastline assembly and additional topology validation.
3. Nautical-chart import: supplies richer maritime semantics, but brings source access, format and depth-datum work beyond the first map.

## Source and reproducibility

Use https://osmdata.openstreetmap.de/data/land-polygons.html as the coastline-derived land source. Record the exact download URL, retrieval time, available source timestamp, SHA256, source CRS, crop bounds, transform CRS, processing parameters and output hash. Display OpenStreetMap attribution and retain source licence information with distributable derived data.

Import is an explicit preprocessing command. Simulation and route planning load a saved local map and require no network access. Large original downloads live in an ignored cache; the small processed map and manifest are retained. Fail clearly on unavailable source data rather than substituting invented geometry.

## Geometry and interfaces

- Add immutable polygon-based map geometry with Polygon/MultiPolygon and holes, rectangular crop bounds, metadata, load/save and continuous segment-clearance checks.
- Use Shapely for geometry and PyProj for WGS84-to-local metric projection. Use a wheel-backed reader such as Pyogrio for the source shapefile if needed; avoid direct OSGeo/Fiona dependencies.
- Project the Singapore crop to WGS84 / UTM zone 48N and subtract a recorded origin to obtain local East-North metres. Preserve reversible coordinate conversion.
- Clip land polygons to the crop. Reject invalid/non-finite geometry and record any explicit repair. Preserve small islands and narrow passages; no unrecorded geometry simplification.
- A route segment is clear only if its swept vessel footprint, including planning margin, avoids land and remains inside the crop. Touching is blocked. Cover polygon holes, concavities and disconnected islands correctly.
- Retain simple rectangle construction as synthetic test fixtures, converted to polygon geometry.

Keep original projected vector geometry as the collision authority. Use spatial indexing and cached local distance/occupancy tiles to accelerate candidate searches. Raster resolution and interpolation errors must not create false-free segments: conservative occupancy and vector revalidation are required. Declare physical clearance, localization uncertainty and source-geometry uncertainty separately; an arbitrary planning margin is not a calibrated uncertainty bound.

The initial planner uses a conservative enclosing disk for the specified vessel dimensions. A later oriented-hull trajectory checker must cover translation and rotation between timestamps, using bounded subdivision or swept-volume bounds; checking sampled poses alone is insufficient.

## Planning and demonstration

Implement grid A* and line-of-sight smoothing against the same continuous geometry checks. Select grid resolution based on crop scale and a bounded cell budget. A coarse-grid failure is reported as such, not as proof that no continuous route exists.

Retain A* plus smoothing as the reference baseline and compare Basic Theta* under the same resolution, footprint and compute budget. Add tiled/coarse-to-fine search only after measuring a bottleneck; retain alternative coarse corridors or allow refinement so a coarse blocked cell does not silently erase a narrow passage. All accelerated outputs receive the same final geometric validation.

Geometric routes do not establish vessel feasibility. The later marine-planning phase should evaluate a state lattice built from the chosen vessel dynamics and a constrained local trajectory controller. Do not use car steering constraints as a substitute for ship dynamics. Keep this phase separate from map import and compare it against the simpler geometric-plus-tracker baseline.

Expose explicit vessel radius, planning margin, grid resolution and start/goal inputs. Select demonstration endpoints only after checking actual land geometry. Keep real distances; do not shrink kilometres to the previous 24-metre toy scale.

Deliver a headless command that loads the saved map, plans a route, saves route coordinates and diagnostic metadata, and renders a PNG with land, water, endpoints, route, metre axes, scale and attribution. Report coastline-only clearance honestly; this is not bathymetric grounding avoidance.

## Validation

- Projection round trips and known axis ordering.
- Polygon holes, concave coasts, islands, boundary contact and narrow channels.
- Swept segments that cross land even when endpoints lie in water.
- Map serialization, invalid geometry and metadata round trips.
- Deterministic routing, blocked endpoints and disconnected regions.
- Every demonstration route segment checked continuously at the configured clearance.
- Real map preview visually inspected against source geography.
- Existing modern-runtime tests remain passing; dependency lock updated without downgrading application pins.

## Benchmark and acceptance protocol

Freeze crop families, vessel configurations and endpoint pairs before planner comparisons. Hold out complete geographic crops, not just random endpoints on the training map. Include island detours, narrow channels, shore-adjacent endpoints, disconnected water and deliberately incomplete coverage.

For every planner/configuration record success, no-path, invalid-route, timeout, route length, continuously checked clearance, node expansions, peak memory and p50/p95/p99 latency with hardware and repetitions. Report route length only alongside failure rates; never discard failed cases. Compare quality under equal resource budgets, with paired runs and uncertainty intervals where appropriate.

Acceptance requires deterministic offline reconstruction from frozen input, reproducible map hashes, no invalid accepted routes in the geometric regression suite, and a visually inspected real-map route demonstration. Runtime and resolution defaults are selected from measured curves; no unmeasured millisecond target is promised. A SOTA claim requires a subsequent comparison to relevant external methods, not just beating an intentionally weak internal baseline.

## Research grounding

- IHO operational S-100 products (2026): https://iho.int/en/the-s-100-framework-is-now-operational and https://iho.int/en/enc-ecdis . Supports the separation of chart, depth, water-level and current layers; does not establish data availability for Singapore.
- Pivtoraiko, Knepper and Kelly, state lattices: https://publications.ri.cmu.edu/optimal-smooth-nonholonomic-mobile-robot-motion-planning-in-state-lattices . Established foundation for differential constraints, not a new maritime SOTA result.
- Nonlinear Model Predictive Control for Enhanced Navigation of Autonomous Surface Vessels: https://arxiv.org/abs/2403.19028 . Relevant future comparison for tracking, avoidance, environmental disturbances and anti-grounding.
- COLREGs Compliant Collision Avoidance and Grounding Prevention for Autonomous Marine Navigation (2026 preprint): https://arxiv.org/abs/2603.02484 . Research comparator involving uncertainty-aware collision avoidance; its reported results do not transfer automatically to this implementation.

## Delivery boundary

This increment delivers map import, a frozen geographic map, geometry queries, routing and a preview. It does not implement the later traffic simulator, GUI, RL training, CommonOcean integration or NTNU port. Existing checkpoint compatibility code and supplied historical assets remain unchanged.

## Approval status

Proposed design awaiting user review; no implementation has started. On approval, create an execution plan and implement inside `rebuild/`. Keep this document Git-ignored under the existing repository preference.
