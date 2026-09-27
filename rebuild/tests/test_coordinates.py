from math import pi, radians, hypot

import pytest

from shipnav.adapters.coordinates import north_east_to_xy, conservative_radius


def test_north_east_heading_and_hull():
    assert north_east_to_xy(10, 20, 0) == pytest.approx((20, 10, pi/2))
    assert north_east_to_xy(10, 20, pi/2) == pytest.approx((20, 10, 0))
    assert conservative_radius(4, 3) == 2.5


def test_north_east_to_xy_round_trip_points():
    # East becomes x, North becomes y: round-tripping through the inverse
    # (x -> east, y -> north) must recover the original position exactly.
    for north, east in [(0, 0), (5.5, -3.25), (-12.0, 7.0), (100.0, 100.0)]:
        x, y, _ = north_east_to_xy(north, east, 0.0)
        assert (y, x) == pytest.approx((north, east))


def test_north_east_to_xy_heading_round_trip():
    # heading_from_north measured clockwise from North; the returned heading
    # is measured counter-clockwise from East (standard math convention).
    # Applying the transform and converting back must recover the original
    # heading modulo 2*pi.
    for heading in [0.0, pi/4, pi/2, pi, -pi/2, 3*pi/2, -0.1, 2*pi - 0.1]:
        _, _, math_heading = north_east_to_xy(0, 0, heading)
        recovered_heading_from_north = (pi/2 - math_heading) % (2*pi)
        expected = heading % (2*pi)
        assert recovered_heading_from_north == pytest.approx(expected)


def test_north_east_to_xy_requires_radians_not_degrees():
    """north_east_to_xy takes heading_from_north in radians. Callers holding
    a heading in degrees (as many upstream/NTNU-style sources report) must
    convert explicitly with math.radians before calling; passing raw degrees
    silently produces a wrong heading rather than raising."""
    heading_deg = 90.0
    wrong = north_east_to_xy(0, 0, heading_deg)[2]
    correct = north_east_to_xy(0, 0, radians(heading_deg))[2]
    assert wrong != pytest.approx(correct)
    assert correct == pytest.approx(0.0)


def test_north_east_to_xy_output_range_is_bounded():
    # The returned heading must stay within (-pi, pi] regardless of how far
    # out of range the input heading_from_north is (e.g. accumulated over
    # many simulation timesteps without wrapping).
    for heading in [10 * pi, -10 * pi, 4 * pi + 0.3, -4 * pi - 0.3]:
        _, _, math_heading = north_east_to_xy(0, 0, heading)
        assert -pi < math_heading <= pi


def test_conservative_radius_covers_hull_corners():
    # conservative_radius must be >= the distance from the hull centre to
    # every corner of the length x width rectangle, for several hull shapes
    # and independent of any heading (the returned circle is heading-agnostic
    # by construction, so it must bound corners at every rotation).
    for length, width in [(4, 3), (1, 1), (10, 2), (0.5, 0.5), (6, 6)]:
        r = conservative_radius(length, width)
        half_diag = hypot(length, width) / 2
        assert r == pytest.approx(half_diag)
        corner = (length/2, width/2)
        assert hypot(*corner) == pytest.approx(r)
        for _heading in [0.0, pi/6, pi/3, pi/2, pi]:
            # Rotating the hull does not change the distance from centre to
            # any corner, so the same conservative radius still bounds it.
            assert hypot(*corner) <= r + 1e-9


def test_conservative_radius_rejects_non_positive_dimensions():
    with pytest.raises(ValueError):
        conservative_radius(0, 3)
    with pytest.raises(ValueError):
        conservative_radius(4, 0)
    with pytest.raises(ValueError):
        conservative_radius(-1, 3)
