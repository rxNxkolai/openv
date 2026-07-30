"""Live capture and inference loop.

Runs on its own thread and publishes two things: the latest annotated frame as
JPEG bytes, and a snapshot of the current numbers. HTTP handlers read those slots,
they never trigger inference themselves. That decoupling is what keeps the browser
smooth when inference is slower than the request rate, and stops N open tabs from
becoming N inference loops.

Raw frames live only inside this loop. Nothing writes them to disk.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from patron.events import VisitTracker, ZoneVisit
from patron.render import Renderer
from patron.tracking import PersonTracker
from patron.types import Box, FrameResult, TrackedPerson
from patron.zones import Zone, ZoneSet


@dataclass
class _ZoneRollup:
    visits: int = 0
    total_dwell: float = 0.0
    shoppers: set[int] = field(default_factory=set)

    @property
    def mean_dwell(self) -> float:
        return self.total_dwell / self.visits if self.visits else 0.0


class LiveEngine:
    def __init__(
        self,
        camera: int = 0,
        width: int = 1280,
        height: int = 960,
        zones_path: str | Path | None = None,
        conf: float = 0.4,
        resolution: int = 896,
        variant: str = "medium",
        device: str | None = None,
        jpeg_quality: int = 80,
    ) -> None:
        self.camera = camera
        self.width = width
        self.height = height
        self.zones_path = Path(zones_path) if zones_path else None
        self.jpeg_quality = jpeg_quality

        self._conf = conf
        self._resolution = resolution
        self._variant = variant
        self._device = device

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._jpeg: bytes | None = None
        self._error: str | None = None
        self._fps = 0.0
        self._fps_window: deque[float] = deque(maxlen=30)
        self._people_now = 0
        self._seen_ids: set[int] = set()
        self._occupancy: dict[str, int] = {}
        self._rollups: dict[str, _ZoneRollup] = {}
        self._recent: deque[dict[str, Any]] = deque(maxlen=12)
        self._started_at = time.time()

        self._zones = ZoneSet(zones=())
        if self.zones_path and self.zones_path.exists():
            self._zones = ZoneSet.load(self.zones_path)

        self._visit_tracker: VisitTracker | None = None
        self._renderer: Renderer | None = None

    # ---------------- lifecycle ----------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="patron-live", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    # ---------------- published state ----------------

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def stats(self) -> dict[str, Any]:
        with self._lock:
            zones = [
                {
                    "name": name,
                    "inside": self._occupancy.get(name, 0),
                    "visits": rollup.visits,
                    "shoppers": len(rollup.shoppers),
                    "mean_dwell": round(rollup.mean_dwell, 1),
                }
                for name, rollup in self._rollups.items()
            ]
            return {
                "running": self._thread is not None and self._thread.is_alive(),
                "error": self._error,
                "fps": round(self._fps, 1),
                "people_now": self._people_now,
                "shoppers_total": len(self._seen_ids),
                "uptime_s": int(time.time() - self._started_at),
                "zones": zones,
                "recent": list(self._recent),
            }

    def zones_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "width": self.width,
                "height": self.height,
                "zones": [
                    {"name": z.name, "kind": z.kind, "polygon": [list(p) for p in z.polygon]}
                    for z in self._zones
                ],
            }

    def set_zones(self, raw_zones: list[dict[str, Any]]) -> None:
        """Replace the zone set and reset the numbers.

        Counts collected against different boundaries are not comparable, so
        changing zones deliberately clears the rollups rather than mixing them.
        """
        zones = ZoneSet(
            zones=tuple(
                Zone(
                    name=z["name"],
                    polygon=tuple((float(x), float(y)) for x, y in z["polygon"]),
                    kind=z.get("kind", "shelf"),
                )
                for z in raw_zones
            )
        )
        with self._lock:
            self._zones = zones
            self._rollups = {name: _ZoneRollup() for name in zones.names}
            self._occupancy = {}
            self._recent.clear()
            self._visit_tracker = None  # rebuilt on the next frame
            self._renderer = None
        if self.zones_path:
            zones.save(self.zones_path)

    # ---------------- the loop ----------------

    def _run(self) -> None:
        cap = cv2.VideoCapture(self.camera, cv2.CAP_DSHOW)
        if not cap.isOpened():
            with self._lock:
                self._error = f"could not open camera {self.camera}"
            return

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        try:
            from patron.detectors import RFDETRPersonDetector

            detector = RFDETRPersonDetector(
                variant=self._variant,
                confidence=self._conf,
                device=self._device,
                resolution=self._resolution,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the browser
            with self._lock:
                self._error = f"detector failed to load: {exc}"
            cap.release()
            return

        # Camera fps is measured, not trusted, for the same reason the recorder
        # measures it: a wrong rate silently scales every dwell time.
        for _ in range(10):
            cap.read()
        warm_start = time.perf_counter()
        for _ in range(20):
            cap.read()
        capture_fps = max(1.0, 20 / (time.perf_counter() - warm_start))

        tracker = PersonTracker(fps=capture_fps, algorithm="bytetrack")
        frame_index = 0

        try:
            while not self._stop.is_set():
                loop_start = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    with self._lock:
                        self._error = "camera stopped delivering frames"
                    break

                detections = detector.detect(frame)
                people = tracker.update(detections, frame)
                result = FrameResult(
                    frame_index=frame_index,
                    timestamp_s=frame_index / capture_fps,
                    people=people,
                )

                self._consume(result, frame, capture_fps)
                frame_index += 1

                self._fps_window.append(time.perf_counter() - loop_start)
                if self._fps_window:
                    mean = sum(self._fps_window) / len(self._fps_window)
                    with self._lock:
                        self._fps = 1.0 / mean if mean > 0 else 0.0
        finally:
            cap.release()

    def _consume(self, result: FrameResult, frame: np.ndarray, fps: float) -> None:
        with self._lock:
            zones = self._zones
            if self._visit_tracker is None and len(zones):
                self._visit_tracker = VisitTracker(
                    zones=zones,
                    fps=fps,
                    min_frames_inside=max(1, round(0.2 * fps)),
                    min_frames_outside=max(1, round(0.5 * fps)),
                    track_timeout_frames=max(1, round(1.5 * fps)),
                )
                self._rollups = {name: _ZoneRollup() for name in zones.names}
            if self._renderer is None:
                self._renderer = Renderer(
                    resolution_wh=(self.width, self.height),
                    draw_traces=True,
                    zones=zones if len(zones) else None,
                )
            visit_tracker = self._visit_tracker
            renderer = self._renderer

        completed: list[ZoneVisit] = []
        if visit_tracker is not None:
            completed = visit_tracker.update(result)

        occupancy: dict[str, int] = {}
        for person in result.people:
            for name in zones.containing(person.box.foot_point):
                occupancy[name] = occupancy.get(name, 0) + 1

        canvas = renderer.annotate(frame, result)
        ok, buffer = cv2.imencode(
            ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )

        with self._lock:
            self._people_now = result.count
            self._seen_ids.update(p.track_id for p in result.people)
            self._occupancy = occupancy
            for visit in completed:
                rollup = self._rollups.setdefault(visit.zone, _ZoneRollup())
                rollup.visits += 1
                rollup.total_dwell += visit.dwell_s
                rollup.shoppers.add(visit.track_id)
                self._recent.appendleft(
                    {
                        "track_id": visit.track_id,
                        "zone": visit.zone,
                        "dwell": round(visit.dwell_s, 1),
                    }
                )
            if ok:
                self._jpeg = buffer.tobytes()


__all__ = ["LiveEngine", "Box", "TrackedPerson"]
