"""Ground-plane mapping: image pixels to floor coordinates.

The maths is exact, so these tests are exact. A homography is fully determined by
four correspondences, which means a correct implementation reproduces the source
transform on points it never saw, not merely on the ones it was fitted to.
"""

import json

import numpy as np
import pytest

from patron.floor import FloorMap

# A camera looking along an aisle. The last row is what makes it a perspective
# transform rather than an affine one: w shrinks as y grows, so equal pixel
# distances cover more floor further away. The horizon sits at y = 2000, where
# w reaches zero.
H_TRUE = np.array(
    [
        [0.010, 0.000, 0.0],
        [0.000, 0.008, 0.0],
        [0.000, -0.0005, 1.0],
    ]
)


def apply(matrix, point):
    x, y, w = matrix @ np.array([point[0], point[1], 1.0])
    return (x / w, y / w)


def correspondences_from(matrix, image_points):
    return tuple((p, apply(matrix, p)) for p in image_points)


CORNERS = [(100.0, 200.0), (900.0, 200.0), (900.0, 800.0), (100.0, 800.0)]
EXACT = correspondences_from(H_TRUE, CORNERS)


def test_projects_points_it_was_never_fitted_to():
    floor = FloorMap(EXACT)

    # Four points determine the transform, so an unseen fifth must land where
    # the source homography says it does. Fitting the corners is not evidence.
    for probe in [(500.0, 500.0), (250.0, 350.0), (880.0, 780.0), (110.0, 210.0)]:
        expected = apply(H_TRUE, probe)
        assert floor.project(probe) == pytest.approx(expected, abs=1e-6)


def test_the_calibration_points_themselves_round_trip():
    floor = FloorMap(EXACT)

    for image_point, floor_point in EXACT:
        assert floor.project(image_point) == pytest.approx(floor_point, abs=1e-6)


def test_four_points_cannot_verify_themselves():
    """The trap this API refuses to fall into.

    A homography has eight degrees of freedom; four correspondences supply
    exactly eight equations. The fit is therefore exact whatever the points are,
    including four clicked on a shelf edge instead of the floor. A reprojection
    error of 0.000 there would be a confident number standing in for no evidence.
    """
    honest = FloorMap(EXACT)

    nonsense = list(EXACT)
    image_point, floor_point = nonsense[2]
    nonsense[2] = (image_point, (floor_point[0] + 1.5, floor_point[1] - 1.2))
    wrong = FloorMap(tuple(nonsense))

    # Both fit exactly. Neither can be checked. Say so rather than report zero.
    assert honest.reprojection_error is None
    assert wrong.reprojection_error is None
    assert honest.is_verifiable is False
    assert wrong.is_verifiable is False


def test_a_fifth_point_is_what_makes_a_bad_click_visible():
    """Redundancy is the only thing that turns a residual into evidence."""
    probe = (500.0, 500.0)

    clean = FloorMap(correspondences_from(H_TRUE, CORNERS + [probe]))

    sloppy = list(correspondences_from(H_TRUE, CORNERS + [probe]))
    image_point, floor_point = sloppy[4]
    sloppy[4] = (image_point, (floor_point[0] + 1.5, floor_point[1] - 1.2))
    bad = FloorMap(tuple(sloppy))

    assert clean.is_verifiable and bad.is_verifiable
    assert clean.reprojection_error < 0.01
    assert bad.reprojection_error > 0.1


def test_points_above_the_horizon_project_to_nothing():
    floor = FloorMap(EXACT)

    # w = 1 - 0.0005y, so y = 2000 is the horizon and beyond it the ground plane
    # has no finite image. A number here would read as a shopper standing
    # somewhere impossible.
    assert floor.project((500.0, 2000.0)) is None
    assert floor.project((500.0, 2600.0)) is None
    assert floor.project((500.0, 1900.0)) is not None


def test_project_many_preserves_position_and_gaps():
    floor = FloorMap(EXACT)

    out = floor.project_many([(500.0, 500.0), (500.0, 2600.0), (250.0, 350.0)])

    assert len(out) == 3
    assert out[0] == pytest.approx(apply(H_TRUE, (500.0, 500.0)), abs=1e-6)
    assert out[1] is None  # above the horizon, and it stays in place as a gap
    assert out[2] == pytest.approx(apply(H_TRUE, (250.0, 350.0)), abs=1e-6)


def test_four_points_is_the_minimum():
    with pytest.raises(ValueError, match="at least 4 correspondences"):
        FloorMap(EXACT[:3])


def test_collinear_points_are_rejected_rather_than_solved():
    # Three points on a line leave the transform underdetermined. Solving it
    # anyway would produce a mapping that looks fine until someone measures
    # something with it.
    collinear = correspondences_from(
        H_TRUE, [(100.0, 300.0), (300.0, 300.0), (500.0, 300.0), (700.0, 300.0)]
    )

    with pytest.raises(ValueError):
        FloorMap(collinear)


def test_extra_correspondences_are_all_used():
    """More than four points should improve the fit, not be discarded.

    RANSAC would drop a bad click as an outlier, which is exactly the wrong
    behaviour here: a handful of deliberate clicks means a bad one is
    information, not noise.
    """
    noisy = list(correspondences_from(H_TRUE, CORNERS + [(500.0, 500.0)]))
    image_point, floor_point = noisy[4]
    noisy[4] = (image_point, (floor_point[0] + 0.8, floor_point[1]))

    floor = FloorMap(tuple(noisy))

    assert len(floor.correspondences) == 5
    # The bad fifth point drags the fit, which is the point: it is visible.
    assert floor.reprojection_error > 0.05
    # RANSAC would have discarded it and reported a clean fit instead.
    assert floor.project(CORNERS[0]) != pytest.approx(EXACT[0][1], abs=1e-9)


def test_round_trip_through_json(tmp_path):
    original = FloorMap(EXACT, units="m")
    path = tmp_path / "store.floor.json"
    original.save(path)

    loaded = FloorMap.load(path)

    assert loaded.units == "m"
    assert len(loaded.correspondences) == 4
    assert loaded.project((500.0, 500.0)) == pytest.approx(
        original.project((500.0, 500.0)), abs=1e-9
    )


def test_saved_calibration_keeps_the_points_not_just_the_matrix(tmp_path):
    """Nine opaque floats cannot be re-judged six months later."""
    path = tmp_path / "store.floor.json"
    FloorMap(EXACT).save(path)

    raw = json.loads(path.read_text(encoding="utf-8"))

    assert "correspondences" in raw
    assert len(raw["correspondences"]) == 4
    assert set(raw["correspondences"][0]) == {"image", "floor"}


def test_people_are_projected_from_their_feet_not_their_centre():
    from patron.types import Box, FrameResult, TrackedPerson

    floor = FloorMap(EXACT)
    box = Box(x1=460.0, y1=300.0, x2=540.0, y2=500.0)
    result = FrameResult(
        frame_index=0,
        timestamp_s=0.0,
        people=(TrackedPerson(track_id=7, box=box, confidence=0.9),),
    )

    positions = floor.project_people(result)

    # foot_point is bottom-centre: (500, 500), not the centre at (500, 400).
    assert positions[7] == pytest.approx(apply(H_TRUE, (500.0, 500.0)), abs=1e-6)
    assert positions[7] != pytest.approx(apply(H_TRUE, (500.0, 400.0)), abs=1e-6)


def test_a_shopper_above_the_horizon_is_omitted_not_guessed():
    from patron.types import Box, FrameResult, TrackedPerson

    floor = FloorMap(EXACT)
    grounded = TrackedPerson(
        track_id=1, box=Box(x1=460, y1=300, x2=540, y2=500), confidence=0.9
    )
    impossible = TrackedPerson(
        track_id=2, box=Box(x1=460, y1=2400, x2=540, y2=2600), confidence=0.9
    )
    result = FrameResult(
        frame_index=0, timestamp_s=0.0, people=(grounded, impossible)
    )

    positions = floor.project_people(result)

    assert 1 in positions
    # A shopper standing through a wall is harder to notice than a missing one.
    assert 2 not in positions


def test_real_units_survive_the_mapping():
    """A rectangle of known size on the floor must measure that size after
    projection, because everything downstream (distance, speed, floor area)
    depends on the units being real rather than arbitrary."""
    # Camera sees a 4m x 6m floor patch as a trapezoid.
    trapezoid = [(400.0, 300.0), (600.0, 300.0), (800.0, 900.0), (200.0, 900.0)]
    rectangle = [(0.0, 6.0), (4.0, 6.0), (4.0, 0.0), (0.0, 0.0)]
    floor = FloorMap(tuple(zip(trapezoid, rectangle)))

    corners = [floor.project(p) for p in trapezoid]
    assert all(c is not None for c in corners)

    width = np.hypot(corners[1][0] - corners[0][0], corners[1][1] - corners[0][1])
    depth = np.hypot(corners[3][0] - corners[0][0], corners[3][1] - corners[0][1])

    assert width == pytest.approx(4.0, abs=1e-6)
    assert depth == pytest.approx(6.0, abs=1e-6)
