"""The calibrator's status line.

The interactive loop needs a window and a keyboard, so it is not tested here.
This is the one piece of it that carries a guarantee rather than pixels: a
four-point calibration must never present as verified, because it fits exactly
whatever the points are and a reported 0.000 would be a confident number
standing in for no evidence at all.
"""

import numpy as np
import pytest

from patron.cli import _calibration_status
from patron.floor import FloorMap

H_TRUE = np.array([[0.01, 0.0, 0.0], [0.0, 0.008, 0.0], [0.0, -0.0005, 1.0]])


def apply(point):
    x, y, w = H_TRUE @ np.array([point[0], point[1], 1.0])
    return (x / w, y / w)


def pairs_for(points):
    return tuple((p, apply(p)) for p in points)


CORNERS = [(100.0, 200.0), (900.0, 200.0), (900.0, 800.0), (100.0, 800.0)]


def test_too_few_points_asks_for_more():
    text, warn = _calibration_status(None, 2, "m")

    assert "2/4" in text
    assert warn is True


def test_four_points_never_present_as_verified():
    floor = FloorMap(pairs_for(CORNERS))

    text, warn = _calibration_status(floor, 4, "m")

    assert "UNVERIFIABLE" in text
    assert warn is True
    # The number that must not appear, because the fit is exact by construction.
    assert "0.000" not in text


def test_a_fifth_point_reports_a_real_error():
    floor = FloorMap(pairs_for(CORNERS + [(500.0, 500.0)]))

    text, warn = _calibration_status(floor, 5, "m")

    assert "UNVERIFIABLE" not in text
    assert "error" in text
    assert warn is False


def test_a_bad_fifth_point_reads_as_a_warning():
    sloppy = list(pairs_for(CORNERS + [(500.0, 500.0)]))
    image_point, floor_point = sloppy[4]
    sloppy[4] = (image_point, (floor_point[0] + 2.0, floor_point[1]))

    text, warn = _calibration_status(FloorMap(tuple(sloppy)), 5, "m")

    assert warn is True
    assert "error" in text


def test_degenerate_points_are_reported_not_silently_accepted():
    text, warn = _calibration_status(None, 5, "m")

    assert "degenerate" in text
    assert warn is True


def test_units_reach_the_status_line():
    floor = FloorMap(pairs_for(CORNERS + [(500.0, 500.0)]), units="tile")

    text, _ = _calibration_status(floor, 5, "tile")

    assert "tile" in text
