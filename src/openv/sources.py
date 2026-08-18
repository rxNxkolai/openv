"""Frame sources: video files and live cameras.

A source yields `(frame_index, timestamp_s, frame_bgr)` and knows its own fps and
resolution so downstream code never has to guess. Raw frames stop at the pipeline,
they are never persisted. See CLAUDE.md constraint 2.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class SourceInfo:
    fps: float
    width: int
    height: int
    frame_count: int | None  # None for a live camera


class VideoSource:
    """Reads frames from a video file or a camera index.

    `spec` is either a path to a video file or `webcam:N` for camera index N.
    """

    def __init__(
        self, spec: str, width: int | None = None, height: int | None = None
    ) -> None:
        self._spec = spec
        self._is_live = spec.startswith("webcam:")

        if self._is_live:
            index_text = spec.split(":", 1)[1]
            try:
                target: int | str = int(index_text)
            except ValueError as exc:
                raise ValueError(
                    f"Bad webcam spec {spec!r}, expected something like 'webcam:0'"
                ) from exc
            # CAP_DSHOW avoids the multi-second MSMF open delay on Windows.
            self._capture = cv2.VideoCapture(target, cv2.CAP_DSHOW)
            if width and height:
                self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        else:
            path = Path(spec)
            if not path.exists():
                raise FileNotFoundError(f"No video at {path}")
            target = str(path)
            self._capture = cv2.VideoCapture(target)

        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open source {spec!r}")

        fps = self._capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps != fps:  # 0, negative, or NaN
            fps = 30.0

        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        raw_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = None if self._is_live or raw_count <= 0 else raw_count

        self.info = SourceInfo(
            fps=float(fps), width=width, height=height, frame_count=frame_count
        )

    @property
    def is_live(self) -> bool:
        return self._is_live

    @property
    def paced(self) -> bool:
        """True when the caller must throttle reads to keep real time.

        A camera delivers frames at its own rate; a file hands them over as fast
        as they can be decoded, which would make every dwell measurement
        meaningless against the wall clock.
        """
        return not self._is_live

    def read(self) -> np.ndarray | None:
        """One frame, or None at end of stream. For callers driving their own loop."""
        ok, frame = self._capture.read()
        return frame if ok else None

    def rewind(self) -> bool:
        """Seek a file back to the start. No-op for a camera."""
        if self._is_live:
            return False
        return bool(self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0))

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        index = 0
        try:
            while True:
                ok, frame = self._capture.read()
                if not ok:
                    break
                yield index, index / self.info.fps, frame
                index += 1
        finally:
            self.release()

    def release(self) -> None:
        if self._capture is not None and self._capture.isOpened():
            self._capture.release()

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()
