"""Reach detection against real bodies, not synthetic keypoints.

Every other reach test builds poses by hand, with shoulders a fixed distance
apart and always square to the camera. That is exactly the assumption that broke:
a shopper turned side-on projects almost no shoulder width, and the extension
ratio, which divides by that width, let a hand resting on a trolley handle
through as engagement.

The fixture holds real MediaPipe keypoints pulled from grocery-store.mp4 at every
frame where a wrist entered a shelf zone, across one genuine reach and three
trolley-push episodes. Labels were adjudicated by eye from the annotated frames.
"""

import json
from pathlib import Path

import pytest

from patron.events import ReachTracker
from patron.types import Box, Pose
from patron.zones import Zone, ZoneSet

FIXTURE = Path(__file__).parent / "fixtures" / "reach_poses.json"

# The predicate under test does not consult zones, but ReachTracker needs a shelf
# zone to exist before it will do anything at all.
ZONES = ZoneSet(
    zones=(Zone(name="shelf", polygon=((0, 0), (10, 0), (10, 10), (0, 10)), kind="shelf"),)
)


def load():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def tracker_for(data, **kwargs):
    return ReachTracker(
        ZONES, fps=30.0, frame_size=tuple(data["frame_size"]), **kwargs
    )


def classify(tracker, sample):
    """Run the real extension predicate over one recorded body."""
    box = Box(*sample["box"])
    if tracker._is_clipped(box):
        return None  # no usable evidence, neither reach nor not-reach
    pose = Pose(points={k: tuple(v) for k, v in sample["points"].items()})
    wrist = pose.get(f"{sample['side']}_wrist", tracker.min_wrist_confidence)
    if wrist is None:
        return None
    return tracker._is_extended(pose, sample["side"], wrist, box.height)


def test_fixture_covers_both_labels_and_several_episodes():
    data = load()
    samples = data["samples"]
    notes = {s["note"] for s in samples}

    assert len(samples) >= 60
    assert sum(1 for s in samples if s["reach"]) >= 10
    assert sum(1 for s in samples if not s["reach"]) >= 40
    # A single episode per label would let a threshold overfit to one shopper.
    assert len(notes) >= 4


def test_real_bodies_are_classified_correctly_at_the_default_threshold():
    data = load()
    tracker = tracker_for(data)

    wrong = [
        (s["frame"], s["reach"], s["note"])
        for s in data["samples"]
        if (verdict := classify(tracker, s)) is not None and verdict != s["reach"]
    ]

    assert wrong == [], f"misclassified real bodies: {wrong}"


def test_the_side_on_trolley_grip_is_rejected():
    """The regression this fixture exists for.

    A hand on a trolley handle scored 1.3 while the shopper faced the camera and
    4.1 once they turned side-on, purely because apparent shoulder width
    collapsed. Both must be rejected, and by the same rule.
    """
    data = load()
    tracker = tracker_for(data)

    side_on = [s for s in data["samples"] if "side-on" in s["note"]]
    assert side_on, "fixture no longer contains the side-on episode"

    passed = [s["frame"] for s in side_on if classify(tracker, s) is True]
    assert passed == [], f"trolley grip read as a reach at frames {passed}"


def test_the_genuine_reach_is_still_detected():
    """Rejecting false reaches is easy if you reject everything."""
    data = load()
    tracker = tracker_for(data)

    genuine = [s for s in data["samples"] if s["reach"]]
    assert [s["frame"] for s in genuine if classify(tracker, s) is not True] == []


def test_bodies_running_off_the_frame_are_skipped():
    data = load()
    tracker = tracker_for(data)

    skipped = [s["frame"] for s in data["samples"] if classify(tracker, s) is None]

    # A truncated box has the wrong height and a partly invisible torso, so both
    # inputs to the extension test are unreliable.
    assert skipped, "expected the fixture to contain at least one edge-clipped body"


def test_the_threshold_has_margin_on_real_bodies():
    """A threshold sitting flush against the data would be luck, not tuning."""
    data = load()
    tracker = tracker_for(data, min_arm_extension=0)  # measure, do not gate

    import math

    ratios = {True: [], False: []}
    for s in data["samples"]:
        box = Box(*s["box"])
        if tracker._is_clipped(box):
            continue
        pose = Pose(points={k: tuple(v) for k, v in s["points"].items()})
        left = pose.get("left_shoulder", 0.5)
        right = pose.get("right_shoulder", 0.5)
        wrist = pose.get(f"{s['side']}_wrist", 0.5)
        shoulder = pose.get(f"{s['side']}_shoulder", 0.5)
        if not (left and right and wrist and shoulder):
            continue
        from patron.events import SHOULDER_WIDTH_FLOOR

        width = max(math.dist(left, right), SHOULDER_WIDTH_FLOOR * box.height)
        ratios[s["reach"]].append(math.dist(wrist, shoulder) / width)

    assert min(ratios[True]) > max(ratios[False]), (
        f"reach {min(ratios[True]):.3f} does not clear "
        f"non-reach {max(ratios[False]):.3f}"
    )


@pytest.mark.parametrize("threshold", [0.0])
def test_disabling_the_gate_admits_everything(threshold):
    data = load()
    tracker = tracker_for(data, min_arm_extension=threshold)

    verdicts = {classify(tracker, s) for s in data["samples"]}
    assert verdicts <= {True, None}
