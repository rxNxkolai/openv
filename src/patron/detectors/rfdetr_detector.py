"""RF-DETR person detector (Apache 2.0).

Only the Nano / Small / Medium / Large variants are used. The XLarge and 2XLarge
checkpoints ship under Roboflow's PML 1.0 licence, not Apache 2.0, so they are
deliberately not exposed here. See CLAUDE.md constraint 1.
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

# Apache 2.0 checkpoints only.
_VARIANTS = ("nano", "small", "medium", "large")


def _resolve_person_class_id() -> int:
    """Find the COCO class id for "person" by name rather than hardcoding it.

    RF-DETR uses the 91-class COCO indexing where person is 1, not the 80-class
    indexing where it is 0. Looking it up means a checkpoint change cannot
    silently start tracking bicycles.
    """
    from rfdetr.assets.coco_classes import COCO_CLASSES

    if isinstance(COCO_CLASSES, dict):
        pairs = COCO_CLASSES.items()
    else:
        pairs = enumerate(COCO_CLASSES)

    for class_id, name in pairs:
        if str(name).strip().lower() == "person":
            return int(class_id)

    raise RuntimeError(
        "Could not find a 'person' class in rfdetr COCO_CLASSES. "
        "The checkpoint's class map changed, fix _resolve_person_class_id."
    )


class RFDETRPersonDetector:
    """Detects people in a frame using RF-DETR."""

    def __init__(
        self,
        variant: str = "medium",
        confidence: float = 0.4,
        device: str | None = None,
        half: bool = True,
        resolution: int = 896,
        slice_size: int | None = None,
    ) -> None:
        if variant not in _VARIANTS:
            raise ValueError(
                f"variant must be one of {_VARIANTS} (Apache 2.0 checkpoints only), got {variant!r}"
            )

        import rfdetr

        class_name = f"RFDETR{variant.capitalize()}"
        model_cls = getattr(rfdetr, class_name, None)
        if model_cls is None:
            # Older rfdetr releases only shipped RFDETRBase / RFDETRLarge.
            model_cls = getattr(rfdetr, "RFDETRBase", None)
            if model_cls is None:
                raise RuntimeError(
                    f"rfdetr has neither {class_name} nor RFDETRBase. "
                    f"Available: {[n for n in dir(rfdetr) if n.startswith('RFDETR')]}"
                )

        if resolution % 32 != 0:
            raise ValueError(
                f"resolution must be divisible by 32, got {resolution}. "
                "RF-DETR windows the ViT patches in pairs (16 * 2)."
            )

        kwargs: dict[str, object] = {"resolution": resolution}
        if device is not None:
            kwargs["device"] = device

        # Detection resolution dominates accuracy on surveillance footage. At the
        # 576 default a mid-aisle shopper in a 4K frame scores ~0.42 and picks up a
        # false positive; at 896 the same shopper scores ~0.80 clean. The cost is
        # roughly 79ms -> 107ms per frame on an RTX 3070.
        self._model = model_cls(**kwargs)
        self.resolution = resolution
        self._confidence = confidence
        self._person_class_id = _resolve_person_class_id()

        if half and (device or "").startswith("cuda"):
            # FP16 Tensor Cores are a large speedup. compile=False on purpose:
            # torch.compile needs Triton, which is not reliable on Windows.
            import torch

            self._model.inference(compile=False, dtype=torch.float16)

        # Tiled inference. A shopper far from a wide-mounted camera can be ~20px
        # tall once the frame is downscaled to the detector input, which whole-frame
        # inference simply misses. Running overlapping crops at native scale fixes
        # it: on an overhead crowd this took detections from 4 to 126 at 0.86 median
        # confidence. It costs roughly 15x the compute, so it is opt-in and suits
        # wide or high camera mounts, not every feed.
        self._slicer = None
        if slice_size is not None:
            self._slicer = sv.InferenceSlicer(
                callback=self._detect_patch,
                slice_wh=slice_size,
                overlap_wh=int(slice_size * 0.2),
                thread_workers=1,
            )
        self.slice_size = slice_size

    @property
    def person_class_id(self) -> int:
        return self._person_class_id

    def _detect_patch(self, patch_rgb: np.ndarray) -> sv.Detections:
        # include_source_image=False matters here: RF-DETR otherwise attaches the
        # source frame as metadata, and merging tiles then fails on the conflict.
        detections = self._model.predict(
            patch_rgb, threshold=self._confidence, include_source_image=False
        )
        if len(detections) == 0:
            return detections
        return detections[detections.class_id == self._person_class_id]

    def detect(self, frame_bgr: np.ndarray) -> sv.Detections:
        # RF-DETR expects RGB. OpenCV hands us BGR. cvtColor is used rather than a
        # [:, :, ::-1] view because torch rejects arrays with negative strides.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._slicer is not None:
            return self._slicer(frame_rgb)

        return self._detect_patch(frame_rgb)
