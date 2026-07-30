"""Zone visits: the event layer.

Turns the per-frame `FrameResult` stream into durable visit records, one row per
(shopper, zone) stay. Enter time, exit time and dwell are all derivable from a
visit, so a visit is the thing worth storing.

Two details that decide whether the numbers are trustworthy:

*Hysteresis.* A shopper standing on a zone boundary flickers in and out every few
frames. Naively that produces dozens of one-frame visits and a meaningless dwell
distribution. A visit only opens after `min_frames_inside` consecutive frames
inside, and only closes after `min_frames_outside` consecutive frames outside.

*Backdating.* Because of the above, by the time a visit is confirmed the shopper
has already been in the zone for several frames. Timestamps are backdated to when
the streak actually began, otherwise every dwell is short by the debounce window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from patron.types import FrameResult
from patron.zones import ZoneSet


@dataclass(frozen=True)
class ZoneVisit:
    """One shopper's continuous stay in one zone.

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


@dataclass
class _OpenVisit:
    entered_frame: int
    entered_s: float
    last_inside_frame: int
    last_inside_s: float


@dataclass
class _PairState:
    inside_streak: int = 0
    outside_streak: int = 0
    open_visit: _OpenVisit | None = None


@dataclass
class VisitTracker:
    """Accumulates zone visits from a FrameResult stream.

    `min_frames_*` are in frames, so they are framerate dependent by design: the
    caller passes fps and the CLI converts sensible defaults from seconds.
    """

    zones: ZoneSet
    fps: float
    min_frames_inside: int = 3
    min_frames_outside: int = 5
    track_timeout_frames: int = 45

    _state: dict[tuple[int, str], _PairState] = field(default_factory=dict, init=False)
    _last_seen: dict[int, tuple[int, float]] = field(default_factory=dict, init=False)

    def update(self, result: FrameResult) -> list[ZoneVisit]:
        """Feed one frame. Returns any visits that completed on this frame."""
        completed: list[ZoneVisit] = []
        visible_ids: set[int] = set()

        for person in result.people:
            visible_ids.add(person.track_id)
            self._last_seen[person.track_id] = (result.frame_index, result.timestamp_s)

            inside_now = set(self.zones.containing(person.box.foot_point))

            for zone_name in self.zones.names:
                key = (person.track_id, zone_name)
                state = self._state.setdefault(key, _PairState())

                if zone_name in inside_now:
                    state.inside_streak += 1
                    state.outside_streak = 0

                    if state.open_visit is None:
                        if state.inside_streak >= self.min_frames_inside:
                            # Backdate to the first frame of the streak.
                            offset = self.min_frames_inside - 1
                            state.open_visit = _OpenVisit(
                                entered_frame=result.frame_index - offset,
                                entered_s=result.timestamp_s - offset / self.fps,
                                last_inside_frame=result.frame_index,
                                last_inside_s=result.timestamp_s,
                            )
                    else:
                        state.open_visit.last_inside_frame = result.frame_index
                        state.open_visit.last_inside_s = result.timestamp_s
                else:
                    state.outside_streak += 1
                    state.inside_streak = 0

                    if (
                        state.open_visit is not None
                        and state.outside_streak >= self.min_frames_outside
                    ):
                        completed.append(
                            self._close(person.track_id, zone_name, state.open_visit)
                        )
                        state.open_visit = None

        completed.extend(self._expire_lost_tracks(result.frame_index, visible_ids))
        return completed

    def _expire_lost_tracks(
        self, frame_index: int, visible_ids: set[int]
    ) -> list[ZoneVisit]:
        """Close visits for shoppers who left the frame entirely.

        Without this, anyone who walks out of shot while inside a zone keeps an
        open visit forever and never appears in the numbers.
        """
        completed: list[ZoneVisit] = []
        lost = [
            track_id
            for track_id, (last_frame, _) in self._last_seen.items()
            if track_id not in visible_ids
            and frame_index - last_frame > self.track_timeout_frames
        ]

        for track_id in lost:
            for zone_name in self.zones.names:
                state = self._state.get((track_id, zone_name))
                if state is not None and state.open_visit is not None:
                    completed.append(self._close(track_id, zone_name, state.open_visit))
                    state.open_visit = None
                self._state.pop((track_id, zone_name), None)
            self._last_seen.pop(track_id, None)

        return completed

    def flush(self) -> list[ZoneVisit]:
        """Close every still-open visit. Call once the stream ends."""
        completed: list[ZoneVisit] = []
        for (track_id, zone_name), state in self._state.items():
            if state.open_visit is not None:
                completed.append(self._close(track_id, zone_name, state.open_visit))
                state.open_visit = None
        self._state.clear()
        self._last_seen.clear()
        return completed

    @staticmethod
    def _close(track_id: int, zone_name: str, visit: _OpenVisit) -> ZoneVisit:
        # Exit is the last frame actually seen inside, not the frame the debounce
        # fired on, so dwell is not inflated by the outside streak.
        return ZoneVisit(
            track_id=track_id,
            zone=zone_name,
            entered_frame=visit.entered_frame,
            entered_s=visit.entered_s,
            exited_frame=visit.last_inside_frame,
            exited_s=visit.last_inside_s,
        )
