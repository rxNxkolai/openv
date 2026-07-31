"""Occlusion tolerance must mean the same wall-clock time at every framerate.

A shopper hidden behind another for two seconds is the same event whether the
camera runs at 7fps or 60fps. If this drifts, distinct-shopper counts drift with
it, and the failure is silent: the tracker still works, it just fragments people
differently depending on the camera.
"""

import pytest

from patron.tracking import ALGORITHMS, PersonTracker


@pytest.mark.parametrize("fps", [7.5, 15.0, 30.0, 60.0])
@pytest.mark.parametrize("lost_seconds", [1.0, 2.0, 5.0])
def test_occlusion_tolerance_is_real_time_at_any_framerate(fps, lost_seconds):
    tracker = PersonTracker(fps=fps, algorithm="bytetrack", lost_seconds=lost_seconds)

    # The library stores its own rescaled frame count. Whatever it does
    # internally, the tolerance in seconds is what has to hold.
    frames = tracker._tracker.maximum_frames_without_update
    assert frames / fps == pytest.approx(lost_seconds, rel=0.15)


def test_library_default_would_be_one_second():
    # Guards the reasoning in tracking.py: `lost_track_buffer` is 30fps-equivalent
    # frames, not source frames, so the library default of 30 is one second of
    # real time rather than half a second on a 60fps camera.
    tracker = PersonTracker(fps=60.0, algorithm="bytetrack", lost_seconds=1.0)
    assert tracker._tracker.maximum_frames_without_update == 60


def test_every_algorithm_constructs():
    for name in ALGORITHMS:
        tracker = PersonTracker(fps=30.0, algorithm=name)
        assert tracker.algorithm == name


def test_unknown_algorithm_is_rejected():
    with pytest.raises(ValueError, match="algorithm must be one of"):
        PersonTracker(fps=30.0, algorithm="deepsort")
