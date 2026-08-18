"""Detector interface.

Detectors return supervision `Detections` filtered to people only. Keeping the
interface this narrow is deliberate: the detector will get swapped (RF-DETR today,
something better in six months) and nothing downstream should have to care.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import supervision as sv


class PersonDetector(Protocol):
    """Detects people in a single BGR frame."""

    def detect(self, frame_bgr: np.ndarray) -> sv.Detections:
        """Return person detections for one frame, already confidence-filtered."""
        ...
