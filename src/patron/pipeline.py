"""The pipeline seam.

`Pipeline.run()` yields a `FrameResult` stream and that is the contract every
downstream stage (zones, events, agent) builds on. `run_with_frames()` additionally
hands back the raw frame, which exists only for rendering and debugging. Raw frames
must never be persisted or leave the edge. See CLAUDE.md constraint 2.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from patron.detectors.base import PersonDetector
from patron.sources import VideoSource
from patron.tracking import PersonTracker
from patron.types import FrameResult


class Pipeline:
    def __init__(self, detector: PersonDetector, tracker: str = "bytetrack") -> None:
        self._detector = detector
        self._tracker_algorithm = tracker

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

            yield frame, FrameResult(
                frame_index=index, timestamp_s=timestamp_s, people=people
            )

    def run(
        self, source: VideoSource, max_frames: int | None = None
    ) -> Iterator[FrameResult]:
        for _frame, result in self.run_with_frames(source, max_frames=max_frames):
            yield result
