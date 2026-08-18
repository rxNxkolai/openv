"""The pipeline seam.

`Pipeline.run()` yields a `FrameResult` stream and that is the contract every
downstream stage (zones, events, agent) builds on. `run_with_frames()` additionally
hands back the raw frame, which exists only for rendering and debugging. Raw frames
must never be persisted or leave the edge. See CLAUDE.md constraint 2.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from openv.detectors.base import PersonDetector
from openv.sources import VideoSource
from openv.tracking import PersonTracker
from openv.types import FrameResult


class Pipeline:
    def __init__(
        self,
        detector: PersonDetector,
        tracker: str = "bytetrack",
        pose: object | None = None,
    ) -> None:
        self._detector = detector
        self._tracker_algorithm = tracker
        # Pose is opt-in. It costs roughly 14ms per person per frame, and only
        # matters once shelf zones exist to reach into.
        self._pose = pose

    def run_with_frames(
        self, source: VideoSource, max_frames: int | None = None
    ) -> Iterator[tuple[np.ndarray, FrameResult]]:
        tracker = PersonTracker(
            fps=source.info.fps, algorithm=self._tracker_algorithm
        )

        for index, timestamp_s, frame in source.frames():
            if max_frames is not None and index >= max_frames:
                break

            detections = self._detector.detect(frame)
            people = tracker.update(detections, frame)
            poses = self._pose.estimate(frame, people) if self._pose is not None else {}

            yield frame, FrameResult(
                frame_index=index,
                timestamp_s=timestamp_s,
                people=people,
                poses=poses,
            )

    def run(
        self, source: VideoSource, max_frames: int | None = None
    ) -> Iterator[FrameResult]:
        for _frame, result in self.run_with_frames(source, max_frames=max_frames):
            yield result
