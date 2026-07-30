"""Pose estimation (MediaPipe, Apache 2.0).

Top-down: RF-DETR gives us the person boxes, and pose runs on each cropped person.
MediaPipe's pose model is single-person regardless of its `num_poses` option, so
cropping is not a workaround here, it is the intended way to use it when you
already have detections.

Only upper-body joints are kept. Patron needs to know whether a hand went toward a
shelf, and nothing else about the body, so storing a full skeleton would be
collecting more than the product uses. See CLAUDE.md constraint 2.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np

from patron.types import WRISTS, Pose, TrackedPerson

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "pose_landmarker_lite.task"

# MediaPipe landmark indices. Wrists are what reach detection runs on; shoulders
# and elbows give it an arm to reason about.
JOINTS: Mapping[str, int] = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}

__all__ = ["JOINTS", "WRISTS", "Pose", "PoseEstimator", "ensure_model"]


def ensure_model(path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Download the pose model on first use."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, path)  # noqa: S310 - fixed Google URL
    return path


class PoseEstimator:
    def __init__(
        self,
        model_path: str | Path | None = None,
        min_confidence: float = 0.3,
        padding: float = 0.12,
        min_box_px: int = 60,
    ) -> None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        path = ensure_model(Path(model_path) if model_path else DEFAULT_MODEL_PATH)
        self._landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(path)),
                num_poses=1,
                min_pose_detection_confidence=min_confidence,
                min_pose_presence_confidence=min_confidence,
            )
        )
        self._padding = padding
        self._min_box_px = min_box_px

    def estimate(
        self, frame_bgr: np.ndarray, people: Sequence[TrackedPerson]
    ) -> dict[int, Pose]:
        """Poses by track id. People too small to resolve are skipped, not guessed."""
        import mediapipe as mp

        if not people:
            return {}

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        out: dict[int, Pose] = {}

        for person in people:
            box = person.box
            if max(box.width, box.height) < self._min_box_px:
                # A 40px-tall shopper has no resolvable wrists. Returning nothing is
                # honest; returning noise would put false reaches in the funnel.
                continue

            pad = int(self._padding * max(box.width, box.height))
            x1 = max(0, int(box.x1) - pad)
            y1 = max(0, int(box.y1) - pad)
            x2 = min(width, int(box.x2) + pad)
            y2 = min(height, int(box.y2) + pad)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue

            crop = np.ascontiguousarray(frame_rgb[y1:y2, x1:x2])
            crop_h, crop_w = crop.shape[:2]

            result = self._landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
            )
            if not result.pose_landmarks:
                continue

            landmarks = result.pose_landmarks[0]
            points: dict[str, tuple[float, float, float]] = {}
            for name, index in JOINTS.items():
                lm = landmarks[index]
                # Landmarks are normalised to the crop, so they map back through the
                # crop origin. They can also fall outside 0..1 when MediaPipe
                # extrapolates an occluded joint.
                points[name] = (
                    x1 + lm.x * crop_w,
                    y1 + lm.y * crop_h,
                    float(min(lm.visibility, lm.presence)),
                )
            out[person.track_id] = Pose(points=points)

        return out
