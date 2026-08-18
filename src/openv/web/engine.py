"""Live capture, inference, and persistence.

Runs on its own thread and publishes three things: the latest annotated frame as
JPEG bytes, a snapshot of the current numbers, and the ranked findings. HTTP
handlers read those slots, they never trigger inference themselves. That
decoupling keeps the browser smooth when inference is slower than the request
rate, and stops N open tabs from becoming N inference loops.

Everything the engine measures goes to the same event store the offline pipeline
writes, so a live session is analysable afterwards with exactly the same
`openv analyze`. A console that only kept numbers in memory would be a demo, not
an instrument.

The source may be a camera or a video file. File replay is paced to the source
framerate and loops, which makes the whole live path demonstrable and testable
without a camera in front of it.

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

from openv.events import (
    DEFAULT_MIN_ARM_EXTENSION,
    ReachTracker,
    VisitTracker,
    ZoneSpan,
)
from openv.render import Renderer
from openv.sources import VideoSource
from openv.tracking import PersonTracker
from openv.types import FrameResult
from openv.zones import FLOOR_KIND, Zone, ZoneSet

# The findings query is cheap, but recomputing it per frame would still be waste.
FINDINGS_INTERVAL_S = 3.0


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
        source: str = "webcam:0",
        width: int = 1280,
        height: int = 960,
        zones_path: str | Path | None = None,
        db_path: str | Path | None = None,
        conf: float = 0.4,
        resolution: int = 896,
        variant: str = "medium",
        device: str | None = None,
        pose: bool = False,
        loop: bool = True,
        jpeg_quality: int = 80,
        min_arm_extension: float = DEFAULT_MIN_ARM_EXTENSION,
        floor_path: str | Path | None = None,
        position_interval: float = 1.0,
    ) -> None:
        self.source_spec = source
        self.width = width
        self.height = height
        self.zones_path = Path(zones_path) if zones_path else None
        self.db_path = Path(db_path) if db_path else None
        self.pose_enabled = pose
        self.loop = loop
        self.jpeg_quality = jpeg_quality
        self.min_arm_extension = min_arm_extension

        # The plan view is opt-in per camera, because it needs a calibration and
        # a camera without one should still run rather than refuse to start.
        self._floor_map = None
        self._floor_recorder = None
        self._floor_extent: tuple[float, float, float, float] | None = None
        self._floor_now: dict[int, tuple[float, float]] = {}
        self._floor_trails: dict[int, list[tuple[float, float]]] = {}
        self._floor_seen: dict[int, float] = {}
        if floor_path is not None:
            from openv.floor import FloorMap, PositionRecorder

            self._floor_map = FloorMap.load(floor_path)
            self._floor_recorder = PositionRecorder(
                self._floor_map, min_interval_s=position_interval
            )
            # Framed on the calibrated area, padded, so the plan shows the floor
            # that was actually surveyed rather than an arbitrary window.
            xs = [c[1][0] for c in self._floor_map.correspondences]
            ys = [c[1][1] for c in self._floor_map.correspondences]
            pad = 0.35 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
            self._floor_extent = (
                min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad
            )

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
        self._reach_rollups: dict[str, _ZoneRollup] = {}
        self._recent: deque[dict[str, Any]] = deque(maxlen=12)
        self._findings: list[dict[str, Any]] = []
        self._started_at = time.time()
        self._session_id: int | None = None

        self._zones = ZoneSet(zones=())
        if self.zones_path and self.zones_path.exists():
            self._zones = ZoneSet.load(self.zones_path)

        self._visits: VisitTracker | None = None
        self._reaches: ReachTracker | None = None
        self._renderer: Renderer | None = None

    # ---------------- lifecycle ----------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="openv-live", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    # ---------------- published state ----------------

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def stats(self) -> dict[str, Any]:
        with self._lock:
            # Enumerate the *configured* zones, not just the ones that have
            # accumulated a span. A zone the operator just drew must appear
            # immediately at zero, otherwise there is no way to tell a zone that
            # registered from one that silently failed to.
            counts = {**self._rollups, **self._reach_rollups}
            zones = [
                {
                    "name": zone.name,
                    "kind": zone.kind,
                    "inside": self._occupancy.get(zone.name, 0),
                    "visits": counts[zone.name].visits if zone.name in counts else 0,
                    "shoppers": len(counts[zone.name].shoppers) if zone.name in counts else 0,
                    "mean_dwell": round(counts[zone.name].mean_dwell, 1)
                    if zone.name in counts
                    else 0.0,
                }
                for zone in self._zones
            ]
            return {
                "running": self._thread is not None and self._thread.is_alive(),
                "error": self._error,
                "source": self.source_spec,
                "pose": self.pose_enabled,
                "session_id": self._session_id,
                "fps": round(self._fps, 1),
                "people_now": self._people_now,
                "shoppers_total": len(self._seen_ids),
                "uptime_s": int(time.time() - self._started_at),
                "zones": zones,
                "recent": list(self._recent),
                "findings": list(self._findings),
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
                    kind=z.get("kind", FLOOR_KIND),
                )
                for z in raw_zones
            )
        )
        with self._lock:
            self._zones = zones
            self._rollups = {}
            self._reach_rollups = {}
            self._occupancy = {}
            self._recent.clear()
            self._findings = []
            self._visits = None  # rebuilt on the next frame
            self._reaches = None
            self._renderer = None
        if self.zones_path:
            zones.save(self.zones_path)

    # ---------------- the loop ----------------

    def _run(self) -> None:
        try:
            source = VideoSource(self.source_spec, width=self.width, height=self.height)
        except Exception as exc:  # noqa: BLE001 - surfaced to the browser
            with self._lock:
                self._error = f"could not open {self.source_spec}: {exc}"
            return

        self.width, self.height = source.info.width, source.info.height
        fps = source.info.fps

        try:
            from openv.detectors import RFDETRPersonDetector

            detector = RFDETRPersonDetector(
                variant=self._variant,
                confidence=self._conf,
                device=self._device,
                resolution=self._resolution,
            )
            pose_estimator = None
            if self.pose_enabled:
                from openv.pose import PoseEstimator

                pose_estimator = PoseEstimator()
        except Exception as exc:  # noqa: BLE001 - surfaced to the browser
            with self._lock:
                self._error = f"model failed to load: {exc}"
            source.release()
            return

        store = None
        if self.db_path is not None:
            from openv.store import EventStore

            store = EventStore(self.db_path)
            session_id = store.start_session(
                source=self.source_spec,
                fps=fps,
                width=self.width,
                height=self.height,
                calibration=(
                    self._floor_map.as_dict() if self._floor_map is not None else None
                ),
            )
            with self._lock:
                self._session_id = session_id

        tracker = PersonTracker(fps=fps, algorithm="bytetrack")
        frame_index = 0
        last_findings = 0.0
        frame_budget = 1.0 / fps if source.paced else 0.0

        try:
            while not self._stop.is_set():
                loop_start = time.perf_counter()
                frame = source.read()

                if frame is None:
                    if source.is_live or not self.loop:
                        break
                    # A file ran out. Rewind and carry on: the console is a live
                    # instrument, and a demo that stops after 30 seconds is not one.
                    source.rewind()
                    tracker = PersonTracker(fps=fps, algorithm="bytetrack")
                    self._close_open_spans(store)
                    frame_index = 0
                    continue

                detections = detector.detect(frame)
                people = tracker.update(detections, frame)
                poses = (
                    pose_estimator.estimate(frame, people)
                    if pose_estimator is not None
                    else {}
                )
                result = FrameResult(
                    frame_index=frame_index,
                    timestamp_s=frame_index / fps,
                    people=people,
                    poses=poses,
                )

                self._consume(result, frame, fps, store)
                frame_index += 1

                now = time.perf_counter()
                self._fps_window.append(now - loop_start)
                mean = sum(self._fps_window) / len(self._fps_window)
                with self._lock:
                    self._fps = 1.0 / mean if mean > 0 else 0.0

                if store is not None and time.time() - last_findings > FINDINGS_INTERVAL_S:
                    self._refresh_findings(store)
                    last_findings = time.time()

                # Replaying a file faster than real time would make every dwell
                # measurement meaningless relative to the wall clock the viewer
                # is watching against.
                if frame_budget:
                    slack = frame_budget - (time.perf_counter() - loop_start)
                    if slack > 0:
                        time.sleep(slack)
        finally:
            self._close_open_spans(store)
            if store is not None:
                self._refresh_findings(store)
                store.close()
            source.release()

    def _close_open_spans(self, store) -> None:
        """Flush anyone still inside a zone so they reach the numbers."""
        with self._lock:
            visits, reaches, session_id = self._visits, self._reaches, self._session_id

        spans: list[ZoneSpan] = visits.flush() if visits is not None else []
        reach_spans: list[ZoneSpan] = reaches.flush() if reaches is not None else []

        if store is not None and session_id is not None:
            store.add_visits(session_id, spans)
            store.add_reaches(session_id, reach_spans)
        self._record(spans, reach_spans)

        with self._lock:
            self._visits = None
            self._reaches = None

    def _refresh_findings(self, store) -> None:
        from openv.analysis import analyze

        with self._lock:
            session_id = self._session_id
        try:
            analysis = analyze(store, session_id)
        except Exception:  # noqa: BLE001 - never let reporting kill capture
            return

        payload = [
            {
                "zone": f.zone,
                "kind": f.kind,
                "severity": f.severity,
                "headline": f.headline,
                "passed": f.funnel.passed,
                "stopped": f.funnel.stopped,
                "reached": f.funnel.reached,
            }
            for f in analysis.findings
        ]
        with self._lock:
            self._findings = payload

    def _consume(self, result: FrameResult, frame: np.ndarray, fps: float, store) -> None:
        with self._lock:
            zones = self._zones
            session_id = self._session_id
            if self._visits is None and len(zones.floor):
                self._visits = VisitTracker(
                    zones=zones,
                    fps=fps,
                    min_frames_inside=max(1, round(0.2 * fps)),
                    min_frames_outside=max(1, round(0.5 * fps)),
                    track_timeout_frames=max(1, round(1.5 * fps)),
                )
            if self._reaches is None and len(zones.shelf) and self.pose_enabled:
                self._reaches = ReachTracker(
                    zones=zones,
                    fps=fps,
                    min_arm_extension=self.min_arm_extension,
                    frame_size=(self.width, self.height),
                )
            if self._renderer is None:
                self._renderer = Renderer(
                    resolution_wh=(self.width, self.height),
                    draw_traces=True,
                    zones=zones if len(zones) else None,
                )
            visits, reaches, renderer = self._visits, self._reaches, self._renderer

        done_visits = visits.update(result) if visits is not None else []
        done_reaches = (
            reaches.update(result, dict(result.poses)) if reaches is not None else []
        )

        if store is not None and session_id is not None:
            if done_visits:
                store.add_visits(session_id, done_visits)
            if done_reaches:
                store.add_reaches(session_id, done_reaches)

        occupancy: dict[str, int] = {}
        for person in result.people:
            for name in zones.floor.containing(person.box.foot_point):
                occupancy[name] = occupancy.get(name, 0) + 1
            pose = result.poses.get(person.track_id)
            if pose is not None:
                for wrist in pose.wrists():
                    for name in zones.shelf.containing(wrist):
                        occupancy[name] = occupancy.get(name, 0) + 1

        floor_now, floor_sampled = self._project_floor(result)
        if store is not None and session_id is not None and floor_sampled:
            store.add_positions(session_id, floor_sampled)

        canvas = renderer.annotate(frame, result)
        ok, buffer = cv2.imencode(
            ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )

        with self._lock:
            self._people_now = result.count
            self._seen_ids.update(p.track_id for p in result.people)
            self._occupancy = occupancy
            self._floor_now = floor_now
            if ok:
                self._jpeg = buffer.tobytes()

        self._record(done_visits, done_reaches)

    # Trails are for looking at, not for recording, so they are held in memory
    # and dropped once a shopper has been gone a few seconds. The durable copy
    # is whatever the recorder sampled into the event store.
    TRAIL_POINTS = 90
    TRAIL_TTL_S = 4.0

    def _project_floor(self, result: FrameResult) -> tuple[dict, list]:
        """Current floor positions, plus whatever the sampler wants persisted."""
        if self._floor_map is None:
            return {}, []

        now = self._floor_map.project_people(result)
        sampled = (
            self._floor_recorder.update(result)
            if self._floor_recorder is not None
            else []
        )

        for track_id, point in now.items():
            trail = self._floor_trails.setdefault(track_id, [])
            trail.append(point)
            if len(trail) > self.TRAIL_POINTS:
                del trail[: -self.TRAIL_POINTS]
            self._floor_seen[track_id] = result.timestamp_s

        stale = {
            track_id
            for track_id, seen in self._floor_seen.items()
            if result.timestamp_s - seen > self.TRAIL_TTL_S
        }
        for track_id in stale:
            self._floor_trails.pop(track_id, None)
            self._floor_seen.pop(track_id, None)
        if stale and self._floor_recorder is not None:
            # Track ids die when a person leaves frame; nothing about them stays.
            self._floor_recorder.forget(stale)

        return now, sampled

    def floor_payload(self) -> dict[str, Any]:
        """Plan-view state for the browser, in floor coordinates."""
        with self._lock:
            if self._floor_map is None:
                return {"enabled": False}
            return {
                "enabled": True,
                "units": self._floor_map.units,
                "extent": list(self._floor_extent),
                "verified": self._floor_map.is_verifiable,
                "positions": [
                    {"track_id": t, "x": round(x, 3), "y": round(y, 3)}
                    for t, (x, y) in self._floor_now.items()
                ],
                "trails": [
                    {"track_id": t, "points": [[round(x, 3), round(y, 3)] for x, y in pts]}
                    for t, pts in self._floor_trails.items()
                    if len(pts) > 1
                ],
            }

    def _record(self, visits: list[ZoneSpan], reaches: list[ZoneSpan]) -> None:
        with self._lock:
            for span, bucket, label in (
                *((v, self._rollups, "visit") for v in visits),
                *((r, self._reach_rollups, "reach") for r in reaches),
            ):
                rollup = bucket.setdefault(span.zone, _ZoneRollup())
                rollup.visits += 1
                rollup.total_dwell += span.dwell_s
                rollup.shoppers.add(span.track_id)
                self._recent.appendleft(
                    {
                        "track_id": span.track_id,
                        "zone": span.zone,
                        "dwell": round(span.dwell_s, 1),
                        "kind": label,
                    }
                )


__all__ = ["LiveEngine"]
