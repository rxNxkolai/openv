"""Plan-view coordinate mapping.

Rendering itself is a demo surface and not worth pinning, but the floor-to-pixel
mapping underneath it is arithmetic that everything visual depends on, and an
inverted axis would look plausible while being wrong.
"""

import numpy as np

from openv.floorview import FloorView

VIEW = FloorView(extent=(0.0, 0.0, 10.0, 8.0), pixels_per_unit=20.0)


def test_canvas_size_follows_the_extent_and_scale():
    assert (VIEW.width, VIEW.height) == (200, 160)


def test_the_plan_is_not_upside_down():
    """Floor y increases away from the viewer; image y increases downward.

    Getting this backwards mirrors the store, which reads as plausible until
    someone compares the plan against the actual aisle they are standing in.
    """
    near = VIEW.to_pixels((5.0, 0.0))
    far = VIEW.to_pixels((5.0, 8.0))

    assert near[1] > far[1]
    assert near == (100, 160)
    assert far == (100, 0)


def test_x_runs_left_to_right():
    assert VIEW.to_pixels((0.0, 4.0))[0] == 0
    assert VIEW.to_pixels((10.0, 4.0))[0] == 200


def test_a_negative_origin_extent_still_maps_correctly():
    view = FloorView(extent=(-4.0, -2.0, 12.0, 14.0), pixels_per_unit=10.0)

    assert (view.width, view.height) == (160, 160)
    assert view.to_pixels((-4.0, 14.0)) == (0, 0)
    assert view.to_pixels((12.0, -2.0)) == (160, 160)
    assert view.to_pixels((0.0, 0.0)) == (40, 140)


def test_render_produces_a_canvas_of_the_declared_size():
    canvas = VIEW.render({1: (2.0, 3.0), 2: (7.5, 6.0)})

    assert canvas.shape == (160, 200, 3)
    assert canvas.dtype == np.uint8


def test_people_are_drawn_where_the_mapping_says():
    blank = VIEW.render({})
    marked = VIEW.render({1: (5.0, 4.0)})

    px, py = VIEW.to_pixels((5.0, 4.0))
    assert not np.array_equal(blank[py, px], marked[py, px])


def test_a_trail_shorter_than_two_points_does_not_crash():
    canvas = VIEW.render({1: (5.0, 4.0)}, trails={1: [(5.0, 4.0)]})

    assert canvas.shape == (160, 200, 3)
