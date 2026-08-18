"""Reach detection against real bodies, not synthetic keypoints.

Every other reach test builds poses by hand, with shoulders a fixed distance
apart and always square to the camera. That is exactly the assumption that broke:
a shopper turned side-on projects almost no shoulder width, and the extension
ratio, which divides by that width, let a hand resting on a trolley handle
through as engagement.

Two fixtures, both real MediaPipe keypoints:

- `reach_poses` covers sensitivity and specificity on one shopper in a grocery
  aisle: one genuine reach, three trolley-push episodes, labels adjudicated by
  eye from the annotated frames.
- `walking_poses` covers specificity at scale from a high overhead concourse
  where nobody reaches for anything, so every sample is a false positive by
  construction across many bodies and every walking direction. It is trimmed to
  the hardest negatives, the ones nearest the decision boundary.

Together they bracket the threshold from both sides, on two camera geometries.
"""

import json
import math
from pathlib import Path

import pytest

from openv.events import SHOULDER_WIDTH_FLOOR, ReachTracker
from openv.types import Box, Pose
from openv.zones import Zone, ZoneSet

FIXTURES = Path(__file__).parent / "fixtures"

# The predicate under test does not consult zones, but ReachTracker needs a shelf
# zone to exist before it will do anything at all.
ZONES = ZoneSet(
    zones=(Zone(name="shelf", polygon=((0, 0), (10, 0), (10, 10), (0, 10)), kind="shelf"),)
)


def load(name):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def tracker_for(data, **kwargs):
    return ReachTracker(ZONES, fps=30.0, frame_size=tuple(data["frame_size"]), **kwargs)


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


def extension_ratio(sample):
    """Recompute the shipped ratio, so the margin tests measure the real thing."""
    def pt(name):
        v = sample["points"].get(name)
        return None if (v is None or v[2] < 0.5) else (v[0], v[1])

    left, right = pt("left_shoulder"), pt("right_shoulder")
    wrist, shoulder = pt(f"{sample['side']}_wrist"), pt(f"{sample['side']}_shoulder")
    height = sample["box"][3] - sample["box"][1]
    if not (left and right and wrist and shoulder) or height <= 0:
        return None
    width = max(math.dist(left, right), SHOULDER_WIDTH_FLOOR * height)
    return math.dist(wrist, shoulder) / width


# --------------------------------------------------------------------------
# The fixtures themselves
# --------------------------------------------------------------------------

def test_reach_fixture_covers_both_labels_and_several_episodes():
    samples = load("reach_poses")["samples"]

    assert len(samples) >= 60
    assert sum(1 for s in samples if s["reach"]) >= 10
    assert sum(1 for s in samples if not s["reach"]) >= 40
    # A single episode per label would let a threshold overfit to one shopper.
    assert len({s["note"] for s in samples}) >= 4


def test_walking_fixture_spans_many_distinct_bodies():
    data = load("walking_poses")
    samples = data["samples"]

    # The grocery fixture's negatives are all one person pushing one trolley.
    # This one exists to stop the threshold fitting that individual.
    assert len({s["track_id"] for s in samples}) >= 10
    assert all(s["reach"] is False for s in samples)
    assert data["observed_sample_count"] > len(samples), "expected a trimmed set"


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", ["reach_poses", "walking_poses"])
def test_real_bodies_are_classified_correctly_at_the_default_threshold(fixture):
    data = load(fixture)
    tracker = tracker_for(data)

    wrong = [
        (s["frame"], s["reach"], s["note"])
        for s in data["samples"]
        if (verdict := classify(tracker, s)) is not None and verdict != s["reach"]
    ]

    assert wrong == [], f"{fixture}: misclassified real bodies: {wrong[:10]}"


def test_the_side_on_trolley_grip_is_rejected():
    """The regression this fixture exists for.

    A hand on a trolley handle scored 1.3 while the shopper faced the camera and
    4.1 once they turned side-on, purely because apparent shoulder width
    collapsed. Both must be rejected, and by the same rule.
    """
    data = load("reach_poses")
    tracker = tracker_for(data)

    side_on = [s for s in data["samples"] if "side-on" in s["note"]]
    assert side_on, "fixture no longer contains the side-on episode"

    passed = [s["frame"] for s in side_on if classify(tracker, s) is True]
    assert passed == [], f"trolley grip read as a reach at frames {passed}"


def test_the_genuine_reach_is_still_detected():
    """Rejecting false reaches is easy if you reject everything."""
    data = load("reach_poses")
    tracker = tracker_for(data)

    genuine = [s for s in data["samples"] if s["reach"]]
    assert [s["frame"] for s in genuine if classify(tracker, s) is not True] == []


def test_nobody_walking_past_a_shelf_registers_a_reach():
    """Specificity at scale, on a camera geometry the threshold never saw."""
    data = load("walking_poses")
    tracker = tracker_for(data)

    fired = [(s["frame"], s["track_id"]) for s in data["samples"]
             if classify(tracker, s) is True]

    assert fired == [], f"walking read as reaching at {fired[:10]}"


def test_bodies_running_off_the_frame_are_skipped():
    data = load("reach_poses")
    tracker = tracker_for(data)

    skipped = [s["frame"] for s in data["samples"] if classify(tracker, s) is None]

    # A truncated box has the wrong height and a partly invisible torso, so both
    # inputs to the extension test are unreliable.
    assert skipped, "expected the fixture to contain at least one edge-clipped body"


# --------------------------------------------------------------------------
# Where the threshold actually sits
# --------------------------------------------------------------------------

def test_the_validated_reach_is_anatomically_implausible():
    """A standing warning, not a passing grade.

    A real arm is about 0.44 of standing height and shoulder breadth about 0.23.
    The one reach this threshold was tuned on measures well outside both, which
    is what a lower body hidden behind a trolley does: the box stops short of
    the floor, standing height is underestimated, and the floored denominator
    inflates the ratio.

    This test exists so the fact cannot quietly stop being true, in either
    direction. If footage arrives where a reach measures anatomically, this
    fails and the threshold should be revisited with it.
    """
    data = load("reach_poses")
    arm_fractions = []

    for sample in data["samples"]:
        if not sample["reach"]:
            continue
        def point(name):
            v = sample["points"].get(name)
            return None if (v is None or v[2] < 0.5) else (v[0], v[1])

        wrist = point(f"{sample['side']}_wrist")
        shoulder = point(f"{sample['side']}_shoulder")
        height = sample["box"][3] - sample["box"][1]
        if wrist and shoulder and height > 0:
            arm_fractions.append(math.dist(wrist, shoulder) / height)

    assert arm_fractions
    # Anatomy says about 0.44. Everything here is longer, so the box is short.
    assert min(arm_fractions) > 0.48, (
        "a reach now measures anatomically: revisit the 2.5 threshold, which "
        "was tuned when they did not"
    )


def test_the_threshold_sits_in_a_real_gap_across_both_clips():
    """A threshold flush against the data would be luck, not tuning.

    Pooling both fixtures: every non-reach must fall below the default and the
    genuine reach must clear it, with daylight on both sides.
    """
    from openv.events import DEFAULT_MIN_ARM_EXTENSION

    positives, negatives = [], []
    for fixture in ("reach_poses", "walking_poses"):
        data = load(fixture)
        tracker = tracker_for(data)
        for s in data["samples"]:
            if tracker._is_clipped(Box(*s["box"])):
                continue
            r = extension_ratio(s)
            if r is not None:
                (positives if s["reach"] else negatives).append(r)

    assert positives and negatives
    assert max(negatives) < DEFAULT_MIN_ARM_EXTENSION < min(positives), (
        f"threshold {DEFAULT_MIN_ARM_EXTENSION} is not inside the gap "
        f"[{max(negatives):.3f}, {min(positives):.3f}]"
    )


def test_disabling_the_gate_admits_everything():
    data = load("reach_poses")
    tracker = tracker_for(data, min_arm_extension=0)

    assert {classify(tracker, s) for s in data["samples"]} <= {True, None}
