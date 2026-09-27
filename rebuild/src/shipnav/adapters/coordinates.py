"""Coordinate and hull-shape conversions for external maritime adapters.

ShipNav's rebuilt simulation uses metres, seconds, radians and Cartesian
East-North (x, y) coordinates with headings measured counter-clockwise from
East (the standard math convention). Several upstream maritime sources (e.g.
NTNU colav-simulator) instead report North-East positions with headings
measured clockwise from North. `north_east_to_xy` performs that explicit
conversion; it is NOT applied by default to sources (such as CommonOcean)
whose native coordinates are already Cartesian — those require their own
verified metadata rather than this transform.
"""
from math import pi


def north_east_to_xy(north, east, heading_from_north):
    """Convert a North-East position and a heading measured clockwise from
    North (radians) into ShipNav's East-North Cartesian (x, y) position and
    a heading measured counter-clockwise from East (radians, in (-pi, pi]).

    Callers holding a heading in degrees must convert with math.radians
    before calling; this function does not detect or convert degree inputs.
    """
    return east, north, (pi/2-heading_from_north+pi) % (2*pi)-pi


def conservative_radius(length, width):
    """Return the radius of a circle centred on the hull that conservatively
    bounds every corner of a length x width rectangular hull, independent of
    heading (the half-diagonal of the rectangle)."""
    from math import hypot
    if min(length, width) <= 0:
        raise ValueError('Positive hull dimensions required')
    return hypot(length, width)/2
