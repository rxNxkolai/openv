"""Zone spans: the event layer.

Turns the per-frame `FrameResult` stream into durable records. A span is one
continuous stay of one body point inside one zone, so enter time, exit time and
dwell are all derivable from it, and a span is the thing worth storing.

Two kinds of span, same machinery, different body point:

- a **visit** is a shopper's foot point inside a floor zone
- a **reach** is a shopper's wrist inside a shelf zone

Two details decide whether the numbers are trustworthy, and they apply to both:

*Hysteresis.* A point sitting on a zone boundary flickers in and out every few
frames. Naively that produces dozens of one-frame spans and a meaningless dwell
distribution. A span only opens after `min_frames_inside` consecutive frames
inside, and only closes after `min_frames_outside` consecutive frames outside.

*Backdating.* Because of the above, by the time a span is confirmed the point has
already been in the zone for several frames. Timestamps are backdated to when the
streak actually began, otherwise every dwell is short by the debounce window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from patron.types import FrameResult, Pose
from patron.zones import ZoneSet

# Wrist-to-shoulder distance, in units of the shopper's own shoulder width, before
# a hand inside a shelf zone counts as engagement.
#
# Tuned against tests/fixtures/reach_poses.json, which holds real keypoints from
# one genuine reach and three trolley-push episodes. On that fixture the genuine
# reach measures 2.59 to 2.84 and the trolley pushes top out at 2.49. The midpoint
# is 2.54. Rounded down, because the project would rather miss a reach than invent
# one. That is a single reach episode by a single shopper, so treat this number as
# provisional until more footage exists.
DEFAULT_MIN_ARM_EXTENSION = 2.5

# Apparent shoulder width is the denominator of the extension ratio, and it
# collapses toward zero when a shopper turns side-on to the camera: the two
# shoulders project onto each other. That sends the ratio to infinity and lets a
# hand resting on a trolley handle through as a reach. Flooring the denominator at
# a fraction of the shopper's own height fixes it, because standing height barely
# moves with rotation. Adult shoulder breadth runs about 0.23 of stature, so this
# floor engages only once the torso is clearly foreshortened.
SHOULDER_WIDTH_FLOOR = 0.20

# A body running off the edge of the frame has a truncated box, so its height is
# wrong and part of its torso is simply not there. Both inputs to the extension
# test are unreliable, so those frames are treated as no evidence rather than as
# no reach.
FRAME_EDGE_MARGIN_PX = 8


def extension_ratio(
    pose: Pose,
    side: str,
    wrist: tuple[float, float],
    body_height: float,
    min_confidence: float = 0.5,
) -> float | None:
    """How far the wrist is from its shoulder, in floored shoulder widths.

    The number the reach gate compares against `min_arm_extension`, exposed so
    that anything measuring this population reads exactly what the detector
    reads. Two implementations of one measurement is how a threshold ends up
    tuned against a number the detector never sees.

    Returns None when there is no torso reference to measure against, which the
    caller must treat as no evidence rather than as a low ratio.
    """
    left = pose.get("left_shoulder", min_confidence)
    right = pose.get("right_shoulder", min_confidence)
    if left is None or right is None:
        return None

    # Floored, because apparent shoulder width collapses under body rotation and
    # an uncorrected ratio then passes anything. See SHOULDER_WIDTH_FLOOR.
    shoulder_width = max(
        math.dist(left, right), SHOULDER_WIDTH_FLOOR * max(body_height, 0.0)
    )
    if shoulder_width < 1e-6:
        return None

    shoulder = left if side == "left" else right
    return math.dist(wrist, shoulder) / shoulder_width


@dataclass(frozen=True)
class ZoneSpan:
    """One continuous presence of one shopper's body point in one zone.

    `track_id` is session-scoped and carries no identity. See CLAUDE.md.
    """

    track_id: int
    zone: str
    entered_frame: int
    entered_s: float
    exited_frame: int
    exited_s: float

    @property
    def dwell_s(self) -> float:
        return max(0.0, self.exited_s - self.entered_s)


# A visit and a reach are the same shape. The distinction is which body point
# produced it and which table it lands in.
ZoneVisit = ZoneSpan
ZoneReach = ZoneSpan


@dataclass
class _OpenSpan:
    entered_frame: int
    entered_s: float
    last_inside_frame: int
    last_inside_s: float


@dataclass
class _PairState:
    inside_streak: int = 0
    outside_streak: int = 0
    open_span: _OpenSpan | None = None


@dataclass
class _PresenceMachine:
    """Debounced in-zone bookkeeping for (track, zone) pairs.

    Deliberately knows nothing about bodies. Callers decide which point to test
    and hand in the resulting membership, which is what lets visits and reaches
    share one tested implementation.
    """

    zone_names: tuple[str, ...]
    fps: float
    min_frames_inside: int = 3
    min_frames_outside: int = 5
    track_timeout_frames: int = 45

    _state: dict[tuple[int, str], _PairState] = field(default_factory=dict, init=False)
    _last_seen: dict[int, tuple[int, float]] = field(default_factory=dict, init=False)

    def update(
        self,
        frame_index: int,
        timestamp_s: float,
        membership: dict[int, set[str]],
        visible: set[int],
    ) -> list[ZoneSpan]:
        """Advance one frame.

        `membership` holds only tracks we have evidence for this frame. A track
        that is visible but absent from it (a shopper whose pose could not be
        resolved, say) has its streaks left untouched rather than being counted as
        outside, because no evidence is not evidence of absence.
        """
        completed: list[ZoneSpan] = []

        for track_id in visible:
            self._last_seen[track_id] = (frame_index, timestamp_s)

        for track_id, inside_now in membership.items():
            for zone_name in self.zone_names:
                key = (track_id, zone_name)
                state = self._state.setdefault(key, _PairState())

                if zone_name in inside_now:
                    state.inside_streak += 1
                    state.outside_streak = 0

                    if state.open_span is None:
                        if state.inside_streak >= self.min_frames_inside:
                            offset = self.min_frames_inside - 1
                            state.open_span = _OpenSpan(
                                entered_frame=frame_index - offset,
                                entered_s=timestamp_s - offset / self.fps,
                                last_inside_frame=frame_index,
                                last_inside_s=timestamp_s,
                            )
                    else:
                        state.open_span.last_inside_frame = frame_index
                        state.open_span.last_inside_s = timestamp_s
                else:
                    state.outside_streak += 1
                    state.inside_streak = 0

                    if (
                        state.open_span is not None
                        and state.outside_streak >= self.min_frames_outside
                    ):
                        completed.append(self._close(track_id, zone_name, state.open_span))
                        state.open_span = None

        completed.extend(self._expire_lost_tracks(frame_index, visible))
        return completed

    def _expire_lost_tracks(self, frame_index: int, visible: set[int]) -> list[ZoneSpan]:
        """Close spans for shoppers who left the frame entirely.

        Without this, anyone who walks out of shot mid-span keeps it open forever
        and never appears in the numbers.
        """
        completed: list[ZoneSpan] = []
        lost = [
            track_id
            for track_id, (last_frame, _) in self._last_seen.items()
            if track_id not in visible
            and frame_index - last_frame > self.track_timeout_frames
        ]

        for track_id in lost:
            for zone_name in self.zone_names:
                state = self._state.get((track_id, zone_name))
                if state is not None and state.open_span is not None:
                    completed.append(self._close(track_id, zone_name, state.open_span))
                    state.open_span = None
                self._state.pop((track_id, zone_name), None)
            self._last_seen.pop(track_id, None)

        return completed

    def flush(self) -> list[ZoneSpan]:
        """Close every still-open span. Call once the stream ends."""
        completed: list[ZoneSpan] = []
        for (track_id, zone_name), state in self._state.items():
            if state.open_span is not None:
                completed.append(self._close(track_id, zone_name, state.open_span))
                state.open_span = None
        self._state.clear()
        self._last_seen.clear()
        return completed

    @staticmethod
    def _close(track_id: int, zone_name: str, span: _OpenSpan) -> ZoneSpan:
        # Exit is the last frame actually seen inside, not the frame the debounce
        # fired on, so dwell is not inflated by the outside streak.
        return ZoneSpan(
            track_id=track_id,
            zone=zone_name,
            entered_frame=span.entered_frame,
            entered_s=span.entered_s,
            exited_frame=span.last_inside_frame,
            exited_s=span.last_inside_s,
        )


class VisitTracker:
    """Shoppers standing in floor zones, tested on the foot point."""

    def __init__(
        self,
        zones: ZoneSet,
        fps: float,
        min_frames_inside: int = 3,
        min_frames_outside: int = 5,
        track_timeout_frames: int = 45,
    ) -> None:
        self.zones = zones.floor
        self._machine = _PresenceMachine(
            zone_names=self.zones.names,
            fps=fps,
            min_frames_inside=min_frames_inside,
            min_frames_outside=min_frames_outside,
            track_timeout_frames=track_timeout_frames,
        )

    def update(self, result: FrameResult) -> list[ZoneVisit]:
        membership: dict[int, set[str]] = {}
        visible: set[int] = set()
        for person in result.people:
            visible.add(person.track_id)
            membership[person.track_id] = set(
                self.zones.containing(person.box.foot_point)
            )
        return self._machine.update(
            result.frame_index, result.timestamp_s, membership, visible
        )

    def flush(self) -> list[ZoneVisit]:
        return self._machine.flush()


class ReachTracker:
    """Hands entering shelf zones, tested on wrists.

    A reach is the signal that separates a shopper who merely stood in front of a
    shelf from one who engaged with it, which is the difference between dwell and
    an actual funnel.
    """

    def __init__(
        self,
        zones: ZoneSet,
        fps: float,
        min_frames_inside: int = 2,
        min_frames_outside: int = 3,
        track_timeout_frames: int = 45,
        min_wrist_confidence: float = 0.5,
        min_arm_extension: float = DEFAULT_MIN_ARM_EXTENSION,
        frame_size: tuple[int, int] | None = None,
    ) -> None:
        self.zones = zones.shelf
        self.min_wrist_confidence = min_wrist_confidence
        self.min_arm_extension = min_arm_extension
        # Without the frame size there is no way to tell a body at the edge of
        # the view from one in the middle, so the check simply does not run.
        self.frame_size = frame_size
        # Reaches are shorter and sharper than visits, so the debounce is tighter.
        # A shopper picking something up may have a hand in the shelf for well
        # under a second.
        self._machine = _PresenceMachine(
            zone_names=self.zones.names,
            fps=fps,
            min_frames_inside=min_frames_inside,
            min_frames_outside=min_frames_outside,
            track_timeout_frames=track_timeout_frames,
        )

    def update(self, result: FrameResult, poses: dict[int, Pose]) -> list[ZoneReach]:
        membership: dict[int, set[str]] = {}
        visible: set[int] = set()

        for person in result.people:
            visible.add(person.track_id)
            pose = poses.get(person.track_id)
            if pose is None:
                # No resolvable pose this frame. Leave the streaks alone rather
                # than ending a reach on missing evidence.
                continue

            if self._is_clipped(person.box):
                # Same treatment as a missing pose: no usable evidence this
                # frame, so leave the streaks alone rather than ending a reach.
                continue

            inside: set[str] = set()
            for side in ("left", "right"):
                wrist = pose.get(f"{side}_wrist", self.min_wrist_confidence)
                if wrist is None or not self._is_extended(
                    pose, side, wrist, person.box.height
                ):
                    continue
                inside.update(self.zones.containing(wrist))
            membership[person.track_id] = inside

        return self._machine.update(
            result.frame_index, result.timestamp_s, membership, visible
        )

    def _is_clipped(self, box) -> bool:
        """Is this body running off the edge of the frame?"""
        if self.frame_size is None:
            return False
        width, height = self.frame_size
        m = FRAME_EDGE_MARGIN_PX
        return (
            box.x1 <= m or box.y1 <= m or box.x2 >= width - m or box.y2 >= height - m
        )

    def _is_extended(
        self, pose: Pose, side: str, wrist: tuple[float, float], body_height: float
    ) -> bool:
        """Is the arm actually reaching, or is the hand just resting near the body?

        A shelf zone is a flat polygon in image space, so it cannot tell a hand at
        the shelf face from a hand merely between the camera and the shelf. A
        shopper pushing a trolley down the aisle has both hands inside the shelf
        polygon from the camera's point of view, and counting that as engagement
        would inflate the funnel with people who never touched anything.

        Arm extension separates the two: the wrist-to-shoulder distance is measured
        in units of the shopper's own shoulder width, so it does not depend on how
        far away they are or how big they appear.

        It does, however, depend on which way they are facing, which is why the
        denominator is floored. A shopper side-on to the camera projects almost no
        shoulder width at all, and the raw ratio then passes a hand resting on a
        trolley handle. Measured on real footage: the same trolley grip scored 1.3
        while the shopper faced the camera and 4.1 once they turned side-on.
        """
        if self.min_arm_extension <= 0:
            return True

        ratio = extension_ratio(
            pose, side, wrist, body_height, self.min_wrist_confidence
        )
        # No torso reference, so no way to judge. Do not silently drop a possible
        # reach on missing evidence.
        return True if ratio is None else ratio >= self.min_arm_extension

    def flush(self) -> list[ZoneReach]:
        return self._machine.flush()
