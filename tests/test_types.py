"""Geometry that the whole spatial layer depends on."""

from patron.types import Box, FrameResult, TrackedPerson


def test_foot_point_is_bottom_center_not_box_center():
    # A shopper leaning over a low shelf: the box is wide and short, so the box
    # center sits well above where they actually stand. Zone membership must use
    # the foot point or they get attributed to the wrong aisle.
    box = Box(x1=100, y1=200, x2=300, y2=400)

    assert box.center == (200.0, 300.0)
    assert box.foot_point == (200.0, 400.0)


def test_box_dimensions():
    box = Box(x1=10, y1=20, x2=40, y2=80)

    assert box.width == 30
    assert box.height == 60


def test_frame_result_count_matches_people():
    people = tuple(
        TrackedPerson(track_id=i, box=Box(0, 0, 10, 10), confidence=0.9)
        for i in range(3)
    )
    result = FrameResult(frame_index=7, timestamp_s=0.25, people=people)

    assert result.count == 3
    assert result.frame_index == 7
