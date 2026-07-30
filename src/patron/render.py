"""Frame annotation for demos and debugging.

Rendering is a development and sales-demo surface, not part of the product data
path. Nothing here writes to the event store.
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from patron.types import FrameResult
from patron.zones import ZoneSet

# Distinct, readable on both dark and bright shelving.
_ZONE_COLORS = [
    (255, 128, 0),
    (0, 200, 255),
    (255, 0, 200),
    (0, 255, 128),
    (200, 200, 0),
    (128, 0, 255),
]


class Renderer:
    def __init__(
        self,
        resolution_wh: tuple[int, int],
        draw_traces: bool = True,
        zones: ZoneSet | None = None,
    ) -> None:
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

        self._box = sv.BoxAnnotator(thickness=thickness)
        self._label = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=max(1, thickness - 1),
            text_position=sv.Position.TOP_LEFT,
        )
        self._trace = (
            sv.TraceAnnotator(
                thickness=thickness,
                trace_length=60,
                position=sv.Position.BOTTOM_CENTER,
            )
            if draw_traces
            else None
        )

        # HUD and foot markers must scale with the frame, otherwise they vanish on
        # 4K footage and swamp the image on a 480p camera.
        self._hud_scale = text_scale * 1.5
        self._hud_thickness = max(1, thickness)
        self._hud_origin = (int(12 * text_scale) + 8, int(40 * text_scale) + 12)
        self._foot_radius = max(3, thickness + 1)
        self._zones = zones
        self._zone_scale = text_scale
        self._zone_thickness = thickness

    def _draw_zones(self, canvas: np.ndarray, occupancy: dict[str, int]) -> None:
        assert self._zones is not None
        overlay = canvas.copy()

        for i, zone in enumerate(self._zones):
            color = _ZONE_COLORS[i % len(_ZONE_COLORS)]
            contour = zone.contour.astype(np.int32)
            cv2.fillPoly(overlay, [contour], color)
            # Shelf zones get a heavier outline: they are tested against wrists,
            # not feet, so it should be obvious at a glance which is which.
            cv2.polylines(
                canvas,
                [contour],
                True,
                color,
                self._zone_thickness * (2 if zone.is_shelf else 1),
            )

            count = occupancy.get(zone.name, 0)
            anchor = contour.reshape(-1, 2).min(axis=0)
            cv2.putText(
                canvas,
                f"{zone.name}: {count}",
                (int(anchor[0]) + 6, int(anchor[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                self._zone_scale,
                color,
                self._zone_thickness,
                cv2.LINE_AA,
            )

        cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, dst=canvas)

    def annotate(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        canvas = frame.copy()

        if self._zones is not None:
            # Floor zones count feet, shelf zones count hands. Occupancy has to be
            # gathered the same way each tracker measures it or the overlay would
            # disagree with the numbers.
            occupancy: dict[str, int] = {}
            for person in result.people:
                for name in self._zones.floor.containing(person.box.foot_point):
                    occupancy[name] = occupancy.get(name, 0) + 1
                pose = result.poses.get(person.track_id)
                if pose is not None:
                    for wrist in pose.wrists():
                        for name in self._zones.shelf.containing(wrist):
                            occupancy[name] = occupancy.get(name, 0) + 1
            self._draw_zones(canvas, occupancy)

        if result.people:
            detections = sv.Detections(
                xyxy=np.array(
                    [
                        [p.box.x1, p.box.y1, p.box.x2, p.box.y2]
                        for p in result.people
                    ],
                    dtype=np.float32,
                ),
                confidence=np.array(
                    [p.confidence for p in result.people], dtype=np.float32
                ),
                class_id=np.zeros(len(result.people), dtype=int),
                tracker_id=np.array([p.track_id for p in result.people], dtype=int),
            )
            labels = [f"#{p.track_id}" for p in result.people]

            if self._trace is not None:
                canvas = self._trace.annotate(canvas, detections)
            canvas = self._box.annotate(canvas, detections)
            canvas = self._label.annotate(canvas, detections, labels=labels)

            for person in result.people:
                # Foot point: what floor-zone membership is tested against.
                fx, fy = person.box.foot_point
                cv2.circle(
                    canvas, (int(fx), int(fy)), self._foot_radius, (0, 255, 255), -1
                )

                # Wrists: what shelf-zone membership is tested against. Drawing
                # the arm makes a missed or hallucinated joint obvious on sight.
                pose = result.poses.get(person.track_id)
                if pose is None:
                    continue
                for side in ("left", "right"):
                    wrist = pose.get(f"{side}_wrist")
                    if wrist is None:
                        continue
                    elbow = pose.get(f"{side}_elbow")
                    if elbow is not None:
                        cv2.line(
                            canvas,
                            (int(elbow[0]), int(elbow[1])),
                            (int(wrist[0]), int(wrist[1])),
                            (255, 255, 255),
                            max(1, self._foot_radius - 2),
                            cv2.LINE_AA,
                        )
                    cv2.circle(
                        canvas,
                        (int(wrist[0]), int(wrist[1])),
                        self._foot_radius + 1,
                        (60, 90, 255),
                        -1,
                    )

        cv2.putText(
            canvas,
            f"frame {result.frame_index}  |  people {result.count}",
            self._hud_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            self._hud_scale,
            (0, 255, 255),
            self._hud_thickness,
            cv2.LINE_AA,
        )
        return canvas
