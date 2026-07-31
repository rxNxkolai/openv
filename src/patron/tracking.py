"""Multi-object tracking (Apache 2.0, via roboflow/trackers).

Assigns identities that are stable across frames within a single session. IDs are
deliberately session-scoped: nothing here persists, and no appearance embedding is
computed, so a person cannot be re-identified across sessions, days, or stores.
That is a privacy constraint, not an oversight. See CLAUDE.md constraint 2.

ByteTrack is the default (fast, motion-only). BoT-SORT is available for when the
camera moves or occlusion gets heavy, at a real speed cost.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import supervision as sv
from trackers import BoTSORTTracker, ByteTrackTracker, OCSORTTracker, SORTTracker

from patron.types import Box, TrackedPerson

ALGORITHMS = {
    "bytetrack": ByteTrackTracker,
    "botsort": BoTSORTTracker,
    "ocsort": OCSORTTracker,
    "sort": SORTTracker,
}


class PersonTracker:
    def __init__(
        self,
        fps: float,
        algorithm: str = "bytetrack",
        track_activation_threshold: float = 0.5,
        lost_seconds: float = 2.0,
    ) -> None:
        if algorithm not in ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {sorted(ALGORITHMS)}, got {algorithm!r}"
            )

        tracker_cls = ALGORITHMS[algorithm]

        # How long a shopper can be hidden before their identity is retired and a
        # fresh one is issued on reacquisition. The library default is 30 frames,
        # which is half a second at 60fps: shorter than an ordinary occlusion in a
        # crowd, so it fragments one shopper into several and inflates every
        # distinct-shopper count. Specified in seconds so the behaviour is the
        # same on a 7fps webcam and a 60fps camera.
        self.lost_seconds = lost_seconds
        lost_buffer = max(1, round(lost_seconds * (fps if fps > 0 else 30.0)))

        # Trackers do not share an init signature, so only pass what each accepts.
        wanted: dict[str, Any] = {
            "frame_rate": float(fps) if fps > 0 else 30.0,
            "track_activation_threshold": track_activation_threshold,
            "lost_track_buffer": lost_buffer,
        }
        accepted = set(inspect.signature(tracker_cls.__init__).parameters)
        kwargs = {k: v for k, v in wanted.items() if k in accepted}

        self.algorithm = algorithm
        self._tracker = tracker_cls(**kwargs)

    def update(
        self, detections: sv.Detections, frame: np.ndarray | None = None
    ) -> tuple[TrackedPerson, ...]:
        # BoT-SORT uses the frame for camera motion compensation. The others ignore it.
        tracked = self._tracker.update(detections, frame)

        if len(tracked) == 0 or tracked.tracker_id is None:
            return ()

        confidences = (
            tracked.confidence
            if tracked.confidence is not None
            else np.ones(len(tracked), dtype=np.float32)
        )

        people: list[TrackedPerson] = []
        for xyxy, confidence, tracker_id in zip(
            tracked.xyxy, confidences, tracked.tracker_id, strict=False
        ):
            if tracker_id is None or int(tracker_id) < 0:
                # Not yet a confirmed track. Drop it rather than emitting an
                # unstable identity downstream.
                continue

            x1, y1, x2, y2 = (float(v) for v in xyxy)
            people.append(
                TrackedPerson(
                    track_id=int(tracker_id),
                    box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=float(confidence),
                )
            )

        return tuple(people)
