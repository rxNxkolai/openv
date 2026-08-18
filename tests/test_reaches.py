"""Reach detection: a hand entering a shelf zone.

A reach is what separates a shopper who stood in front of a shelf from one who
engaged with it, so false reaches are worse than missed ones.
"""

import pytest

from openv.events import ReachTracker, VisitTracker
from openv.types import Box, FrameResult, Pose, TrackedPerson
from openv.zones import Zone, ZoneSet

FPS = 10.0

# Shelf face on the left, walkable floor on the right.
ZONES = ZoneSet(
    zones=(
        Zone(name="shelf-a", polygon=((0, 0), (100, 0), (100, 200), (0, 200)), kind="shelf"),
        Zone(name="aisle", polygon=((100, 0), (400, 0), (400, 200), (100, 200)), kind="floor"),
    )
)


def shopper(track_id: int = 1, foot_x: float = 200, foot_y: float = 150) -> TrackedPerson:
    return TrackedPerson(
        track_id=track_id,
        box=Box(x1=foot_x - 20, y1=foot_y - 80, x2=foot_x + 20, y2=foot_y),
        confidence=0.9,
    )


def pose_with_wrist(x: float, y: float, confidence: float = 0.9) -> Pose:
    """Left hand at (x, y), arm clearly extended (shoulders are 40px apart)."""
    return Pose(
        points={
            "left_wrist": (x, y, confidence),
            "right_wrist": (999, 999, 0.9),  # other hand well outside any zone
            "left_shoulder": (220, 80, 0.9),
            "right_shoulder": (180, 80, 0.9),
        }
    )


def run(tracker: ReachTracker, frames):
    """frames is a list of Pose or None, one per frame."""
    out = []
    for i, pose in enumerate(frames):
        person = shopper()
        poses = {} if pose is None else {1: pose}
        out.extend(
            tracker.update(
                FrameResult(frame_index=i, timestamp_s=i / FPS, people=(person,)),
                poses,
            )
        )
    return out


def test_hand_entering_shelf_produces_a_reach():
    tracker = ReachTracker(ZONES, fps=FPS, min_frames_inside=2, min_frames_outside=3)

    reaching = pose_with_wrist(50, 100)   # inside shelf-a
    withdrawn = pose_with_wrist(250, 100)  # back out in the aisle
    completed = run(tracker, [reaching] * 6 + [withdrawn] * 4)

    assert len(completed) == 1
    assert completed[0].zone == "shelf-a"
    assert completed[0].entered_frame == 0
    assert completed[0].exited_frame == 5
    assert completed[0].dwell_s == pytest.approx(0.5)


def test_standing_in_front_of_a_shelf_is_not_a_reach():
    # The shopper's feet are in the aisle and their hands stay down. Dwell yes,
    # reach no. This is the distinction the whole funnel rests on.
    tracker = ReachTracker(ZONES, fps=FPS, min_frames_inside=2, min_frames_outside=3)

    hands_down = pose_with_wrist(200, 190)
    assert run(tracker, [hands_down] * 20) == []
    assert tracker.flush() == []


def test_feet_inside_a_shelf_zone_never_count_as_a_reach():
    # A shelf zone is only ever tested against wrists. A foot point landing in one
    # (bad zone drawing, or a mirror) must not manufacture engagement.
    tracker = ReachTracker(ZONES, fps=FPS, min_frames_inside=2, min_frames_outside=3)

    standing_in_shelf = shopper(foot_x=50, foot_y=150)
    for i in range(10):
        tracker.update(
            FrameResult(frame_index=i, timestamp_s=i / FPS, people=(standing_in_shelf,)),
            {1: Pose(points={"left_wrist": (250, 100, 0.9)})},
        )
    assert tracker.flush() == []


def test_missing_pose_does_not_end_an_open_reach():
    # MediaPipe drops out for a frame or two under occlusion. Treating that as
    # "hand withdrawn" would chop one reach into several.
    tracker = ReachTracker(ZONES, fps=FPS, min_frames_inside=2, min_frames_outside=3)

    reaching = pose_with_wrist(50, 100)
    completed = run(tracker, [reaching] * 4 + [None] * 5 + [reaching] * 4)

    assert completed == []            # nothing closed during the dropout
    remaining = tracker.flush()
    assert len(remaining) == 1        # still one continuous reach
    assert remaining[0].entered_frame == 0


def test_low_confidence_wrist_is_ignored():
    tracker = ReachTracker(
        ZONES, fps=FPS, min_frames_inside=2, min_frames_outside=3, min_wrist_confidence=0.5
    )

    unsure = pose_with_wrist(50, 100, confidence=0.2)  # inside, but barely seen
    assert run(tracker, [unsure] * 10) == []
    assert tracker.flush() == []


def test_trackers_only_consume_their_own_zone_kind():
    reach = ReachTracker(ZONES, fps=FPS)
    visit = VisitTracker(ZONES, fps=FPS)

    assert reach.zones.names == ("shelf-a",)
    assert visit.zones.names == ("aisle",)


def test_hand_resting_near_the_body_is_not_a_reach():
    """The trolley case, and the reason arm extension exists.

    A shopper pushing a trolley down the aisle has both hands inside the shelf
    polygon from the camera's point of view, because a flat zone cannot tell a
    hand at the shelf from a hand in front of it. Their arms are not extended.
    """
    tracker = ReachTracker(ZONES, fps=FPS, min_frames_inside=2, min_frames_outside=3)

    hands_on_trolley = Pose(
        points={
            # Wrist is geometrically inside shelf-a, but only ~1.1 shoulder widths
            # from the shoulder: hands held in close, not reaching.
            "left_wrist": (55, 115, 0.9),
            "left_shoulder": (70, 80, 0.9),
            "right_shoulder": (30, 80, 0.9),
        }
    )
    assert ZONES.shelf.zones[0].contains((55, 115)) is True  # geometry says yes

    assert run(tracker, [hands_on_trolley] * 12) == []       # extension says no
    assert tracker.flush() == []


def test_arm_extension_gate_can_be_disabled():
    tracker = ReachTracker(
        ZONES, fps=FPS, min_frames_inside=2, min_frames_outside=3, min_arm_extension=0
    )
    close_hand = Pose(
        points={
            "left_wrist": (55, 115, 0.9),
            "left_shoulder": (70, 80, 0.9),
            "right_shoulder": (30, 80, 0.9),
        }
    )
    run(tracker, [close_hand] * 6)
    assert len(tracker.flush()) == 1


def test_missing_shoulders_do_not_block_a_reach():
    # No torso reference means no way to judge extension. Dropping the reach would
    # discard real engagement on missing evidence.
    tracker = ReachTracker(ZONES, fps=FPS, min_frames_inside=2, min_frames_outside=3)
    no_torso = Pose(points={"left_wrist": (50, 100, 0.9)})

    run(tracker, [no_torso] * 6)
    assert len(tracker.flush()) == 1


def test_either_hand_can_reach():
    tracker = ReachTracker(ZONES, fps=FPS, min_frames_inside=2, min_frames_outside=3)

    right_hand_in = Pose(
        points={
            "left_wrist": (300, 100, 0.9),
            "right_wrist": (50, 100, 0.9),  # inside shelf-a
        }
    )
    run(tracker, [right_hand_in] * 5)

    remaining = tracker.flush()
    assert len(remaining) == 1
    assert remaining[0].zone == "shelf-a"
