"""The visit state machine decides whether every dwell number is trustworthy."""

import pytest

from patron.events import VisitTracker
from patron.types import Box, FrameResult, TrackedPerson
from patron.zones import Zone, ZoneSet

FPS = 10.0

# A 100x100 box at the origin. Foot point is bottom-center of a person's box.
ZONES = ZoneSet(zones=(Zone(name="endcap", polygon=((0, 0), (100, 0), (100, 100), (0, 100))),))


def person(track_id: int, foot_x: float, foot_y: float) -> TrackedPerson:
    """A person whose foot point lands exactly on (foot_x, foot_y)."""
    return TrackedPerson(
        track_id=track_id,
        box=Box(x1=foot_x - 10, y1=foot_y - 40, x2=foot_x + 10, y2=foot_y),
        confidence=0.9,
    )


def feed(tracker: VisitTracker, positions, start_frame: int = 0):
    """Feed one frame per entry. `positions` is a list of (x, y) or None."""
    out = []
    for i, pos in enumerate(positions):
        frame_index = start_frame + i
        people = () if pos is None else (person(1, *pos),)
        out.extend(
            tracker.update(
                FrameResult(
                    frame_index=frame_index,
                    timestamp_s=frame_index / FPS,
                    people=people,
                )
            )
        )
    return out


def test_visit_opens_and_closes_with_correct_dwell():
    tracker = VisitTracker(ZONES, fps=FPS, min_frames_inside=3, min_frames_outside=5)

    # 10 frames inside (frames 0-9), then 6 frames outside (frames 10-15).
    completed = feed(tracker, [(50, 50)] * 10 + [(500, 500)] * 6)

    assert len(completed) == 1
    visit = completed[0]
    assert visit.zone == "endcap"
    assert visit.track_id == 1
    # Backdated to frame 0, and exit is the last frame actually inside (9).
    assert visit.entered_frame == 0
    assert visit.exited_frame == 9
    assert visit.dwell_s == pytest.approx(0.9)


def test_boundary_flicker_does_not_produce_spurious_visits():
    tracker = VisitTracker(ZONES, fps=FPS, min_frames_inside=3, min_frames_outside=5)

    # Alternating in/out, never 3 consecutive frames inside.
    completed = feed(tracker, [(50, 50), (500, 500)] * 12)

    assert completed == []


def test_brief_step_out_does_not_split_one_visit_in_two():
    tracker = VisitTracker(ZONES, fps=FPS, min_frames_inside=3, min_frames_outside=5)

    # Inside, a 2-frame blip outside (under the 5-frame debounce), inside again,
    # then a real exit. That is one shopper standing at one shelf, not two visits.
    completed = feed(
        tracker,
        [(50, 50)] * 6 + [(500, 500)] * 2 + [(50, 50)] * 6 + [(500, 500)] * 6,
    )

    assert len(completed) == 1
    assert completed[0].entered_frame == 0
    assert completed[0].exited_frame == 13


def test_shopper_who_leaves_frame_still_gets_a_closed_visit():
    tracker = VisitTracker(
        ZONES, fps=FPS, min_frames_inside=3, min_frames_outside=5, track_timeout_frames=10
    )

    # Inside for 8 frames, then the track vanishes entirely (occlusion, exit).
    completed = feed(tracker, [(50, 50)] * 8 + [None] * 20)

    assert len(completed) == 1
    assert completed[0].exited_frame == 7
    assert completed[0].dwell_s == pytest.approx(0.7)


def test_flush_closes_visits_still_open_at_end_of_stream():
    tracker = VisitTracker(ZONES, fps=FPS, min_frames_inside=3, min_frames_outside=5)

    assert feed(tracker, [(50, 50)] * 10) == []  # still standing there

    remaining = tracker.flush()
    assert len(remaining) == 1
    assert remaining[0].exited_frame == 9


def test_overlapping_zones_both_count():
    zones = ZoneSet(
        zones=(
            Zone(name="aisle-6", polygon=((0, 0), (200, 0), (200, 200), (0, 200))),
            Zone(name="endcap", polygon=((0, 0), (100, 0), (100, 100), (0, 100))),
        )
    )
    tracker = VisitTracker(zones, fps=FPS, min_frames_inside=3, min_frames_outside=5)

    completed = feed(tracker, [(50, 50)] * 8 + [(500, 500)] * 6)

    assert {v.zone for v in completed} == {"aisle-6", "endcap"}


def test_zone_membership_uses_foot_point_not_box_center():
    # Shopper stands just below the zone, leaning in. Their box center falls
    # inside the zone, their feet do not. They must not be counted.
    zones = ZoneSet(zones=(Zone(name="shelf", polygon=((0, 0), (100, 0), (100, 100), (0, 100))),))
    tracker = VisitTracker(zones, fps=FPS, min_frames_inside=1, min_frames_outside=1)

    leaning = TrackedPerson(
        track_id=1, box=Box(x1=40, y1=20, x2=60, y2=160), confidence=0.9
    )
    # The discriminating case: center is inside the zone, feet are outside.
    assert leaning.box.center == (50.0, 90.0)
    assert zones.zones[0].contains(leaning.box.center) is True
    assert leaning.box.foot_point == (50.0, 160.0)
    assert zones.zones[0].contains(leaning.box.foot_point) is False

    for i in range(5):
        tracker.update(FrameResult(frame_index=i, timestamp_s=i / FPS, people=(leaning,)))

    assert tracker.flush() == []
