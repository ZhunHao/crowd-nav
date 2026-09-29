"""Run real-world maps through the canonical model frame by geometric similarity.

Every controller, filter, prediction and dynamics setting in ShipNav is tuned for the
canonical ego: radius .5, speed 1, dt .25 (the supplied SARL's training units). A real
vessel profile with radius r metres and speed V m/s maps onto it with length scale
L = 2r and time scale L/V. Dividing map coordinates by L lets the unchanged service run
the vessel, and results convert back to metres, m/s and seconds for display and export.

Consequences recorded in the map metadata as `model_scale`, stated here once:
dt = .25 L/V seconds; traffic speed .3-1 model units = .3V-V m/s; marine limits
.2 V²/L m/s² and .35 V/L rad/s; the planner grid is `resolution_m`.
"""
from dataclasses import dataclass
from math import ceil, isfinite
from shapely import affinity
from shapely.geometry import box
from shipnav.maps import SeaMap

INTERACTIVE_AREA_M2 = 250_000  # the canonical 1 m planner's max_cells


@dataclass(frozen=True)
class Profile:
    name: str
    radius_m: float
    speed_mps: float
    margin_m: float
    resolution_m: float

    def __post_init__(self):
        if not all(isfinite(v) and v > 0 for v in (self.radius_m, self.speed_mps, self.resolution_m)) \
                or not isfinite(self.margin_m) or self.margin_m < 0:
            raise ValueError('Profile needs positive radius, speed, resolution and nonnegative margin')

    @property
    def length(self):
        return 2*self.radius_m

    @property
    def speed(self):
        return self.speed_mps

    def service_options(self):
        """Planner settings in model units for `execute(**options)`."""
        return {'resolution': self.resolution_m/self.length,
                'clearance': (self.radius_m+self.margin_m)/self.length}


# 'model' is the identity used by synthetic maps and every existing test. The harbour
# craft matches the offline map_demo route settings (5 m radius, 10 m margin, 50 m grid).
PROFILES = {'model': Profile('model', .5, 1., .2, 1.),
            'harbour_craft': Profile('harbour_craft', 5., 5., 10., 50.),
            'coastal_ship': Profile('coastal_ship', 25., 7.5, 25., 100.)}


def default_profile(sea: SeaMap) -> Profile:
    x0, y0, x1, y1 = sea.bounds
    return PROFILES['model'] if (x1-x0)*(y1-y0) <= INTERACTIVE_AREA_M2 else PROFILES['harbour_craft']


def to_model(sea: SeaMap, profile: Profile) -> SeaMap:
    """Scale a metric map into model units. Idempotent guard: refuses an already scaled map."""
    metadata = sea.to_dict()['metadata']
    if 'model_scale' in metadata:
        raise ValueError('Map is already in model units')
    L = profile.length
    bounds = tuple(v/L for v in sea.bounds)
    crop = box(*bounds)
    # Scaling can move land that touches the crop edge outside it by rounding; clip it back.
    land = [affinity.scale(p, 1/L, 1/L, origin=(0, 0)).intersection(crop) for p in sea.land]
    metadata = {**metadata, 'model_scale': {'profile': profile.name, 'length_m': L,
                                            'speed_mps': profile.speed_mps,
                                            'radius_m': profile.radius_m, 'margin_m': profile.margin_m,
                                            'resolution_m': profile.resolution_m}}
    return SeaMap(bounds, [p for p in land if not p.is_empty], metadata)


def units(map_data: dict) -> tuple[float, float]:
    """(metres per model unit, seconds per model second) for a map dict; (1, 1) if unscaled."""
    scale = map_data['metadata'].get('model_scale')
    if scale is None:
        return 1., 1.
    return scale['length_m'], scale['length_m']/scale['speed_mps']


def scaled(sea: SeaMap, profile: Profile) -> SeaMap:
    """`to_model`, except the canonical profile leaves synthetic maps byte-identical."""
    return sea if profile.name == 'model' else to_model(sea, profile)


def check_grid(sea: SeaMap, profile: Profile, max_cells: int = 250_000) -> None:
    """Reject a map/profile pair whose planning grid exceeds the planner's cell budget."""
    x0, y0, x1, y1 = sea.bounds
    cells = ceil((x1-x0)/profile.resolution_m)*ceil((y1-y0)/profile.resolution_m)
    if cells > max_cells:
        raise ValueError(f'{profile.name} plans on a {profile.resolution_m:g} m grid: {cells:,.0f} cells '
                         f'exceed {max_cells:,}. Choose a larger vessel profile')


def map_options(map_data: dict) -> dict:
    """Planner settings recorded in a scaled map (e.g. a frozen real-map scenario), else {}."""
    scale = map_data['metadata'].get('model_scale')
    if scale is None:
        return {}
    L = scale['length_m']
    return {'resolution': scale['resolution_m']/L, 'clearance': (scale['radius_m']+scale['margin_m'])/L}
