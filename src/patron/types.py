"""Normalized types that cross the pipeline seam.

Everything downstream of `pipeline.py` (zones, events, agent) consumes these and
nothing else. Detector-specific and tracker-specific shapes stop here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixel coordinates, top-left origin."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-center: where the person meets the floor.

        Zone membership is always computed on the floor plane, so this is the
        anchor for every spatial question, not `center`. A shopper leaning over a
        low shelf has a box center that drifts far off their actual position.
        """
        return ((self.x1 + self.x2) / 2.0, self.y2)


@dataclass(frozen=True)
class TrackedPerson:
    """One person in one frame, with an identity stable across frames.

    `track_id` is session-scoped by design. It is never persisted, never linked
    across camera sessions, and carries no biometric information. See CLAUDE.md.
    """

    track_id: int
    box: Box
    confidence: float


@dataclass(frozen=True)
class FrameResult:
    """The per-frame output of the pipeline."""

    frame_index: int
    timestamp_s: float
    people: tuple[TrackedPerson, ...]

    @property
    def count(self) -> int:
        return len(self.people)
