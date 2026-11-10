"""The 3D store map: one scene from every camera, replayed or live.

`planview.py` draws a session as aggregates, for a file that can be handed to
somebody. This is the operator's view of the same store: fixtures standing on
the floor plan, each camera's view outlined on the floor, and shoppers as
markers where the tracks put them, moving.

**It shows positions, so it is not an export.** A marker on this map is one
shopper's place at one moment. That is the same anonymous floor coordinate the
event store already holds, with no identity behind it, and it is here for the
people running the store to look at. It is deliberately not in the list of
outbound modules, and nothing that leaves for a third party is built from it.
Markers are numbered within the scene and carry no stored id.

Two ways to get a scene:

- `build_scene` reads a recorded session. For a fused session that is the
  store-wide paths, with fixture findings gathered from the cameras that
  measured each zone.
- `LiveMap` tails the event store while cameras write to it, fuses what they
  report as it arrives, and serves the current positions. Each camera runs as
  its own process against the same store, and this is where they meet.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openv.analysis import Finding
from openv.fusion import FloorFuser, Observation
from openv.products import analyze_products
from openv.store import EventStore

STATIC = Path(__file__).parent / "web" / "static"

#: How close, in floor units and seconds, a camera's sample must be to a fused
#: position to count as that camera watching it. Display only.
_WATCH_RADIUS = 1.0
_WATCH_WINDOW_S = 1.2


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _fixture(fixture, finding: Finding | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "zone": fixture.zone,
        "label": fixture.label,
        "plan": [list(p) for p in fixture.plan] if fixture.plan is not None else None,
        "brands": list(fixture.brands),
        "measured": finding is not None,
        "severity": finding.severity if finding is not None else "none",
    }
    if finding is None:
        out.update(rate="n/a", counts="not measured", readout="NOT MEASURED")
        return out
    funnel = finding.funnel
    counts = f"{funnel.reached} / {funnel.passed}"
    if funnel.has_confidence:
        rate = _pct(funnel.reach_rate)
        out.update(rate=rate, counts=counts, readout=f"REACH {rate}  {counts}")
    else:
        # The same refusal the tables make: counts, and no percentage.
        out.update(rate="n<30", counts=counts, readout=f"{funnel.passed} PASSED  NOT A RATE YET")
    return out


def camera_footprint(calibration: dict, width: int, height: int) -> list[list[float]]:
    """The patch of floor a camera sees, from its calibration and frame size.

    The frame's border is projected onto the floor. Border points at or above
    the horizon have no floor position and are left out, so a camera looking
    down an aisle gets the part of its view that is actually floor.
    """
    from openv.floor import FloorMap

    floor_map = FloorMap(
        tuple(
            (tuple(c["image"]), tuple(c["floor"])) for c in calibration["correspondences"]
        ),
        units=calibration.get("units", "m"),
    )
    steps = 8
    border = (
        [(width * i / steps, height) for i in range(steps + 1)]
        + [(width, height * (1 - i / steps)) for i in range(1, steps + 1)]
        + [(width * (1 - i / steps), 0) for i in range(1, steps + 1)]
        + [(0, height * i / steps) for i in range(1, steps)]
    )
    floor_points = [c["floor"] for c in calibration["correspondences"]]
    xs = [p[0] for p in floor_points]
    ys = [p[1] for p in floor_points]
    reach = 3.0 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    out = []
    for point in border:
        projected = floor_map.project(point)
        if projected is None:
            continue
        x, y = projected
        # Near the horizon a pixel maps to somewhere very far away. Clamp it,
        # so one grazing ray does not turn the outline into a spike.
        x = min(cx + reach, max(cx - reach, x))
        y = min(cy + reach, max(cy - reach, y))
        out.append([round(x, 3), round(y, 3)])
    return out


def _camera_label(session: dict) -> str:
    source = str(session["source"])
    name = source.split(":")[-1].strip() if source.startswith("synthetic") else Path(source).stem
    return name or f"session {session['id']}"


def _cameras(store: EventStore, session_ids: list[int]) -> list[dict[str, Any]]:
    cameras = []
    for session_id in session_ids:
        session = store.session(session_id)
        calibration = store.session_calibration(session_id)
        if session is None or calibration is None or not session["width"]:
            continue
        cameras.append(
            {
                "label": _camera_label(session),
                "session": session_id,
                "footprint": camera_footprint(
                    calibration, int(session["width"]), int(session["height"])
                ),
            }
        )
    return cameras


def _fixtures(store: EventStore, session_ids: list[int]) -> tuple[Any, list[dict]]:
    """Fixtures with the finding from whichever camera measured each zone."""
    store_map = None
    findings: dict[str, Finding] = {}
    for session_id in session_ids:
        products = analyze_products(store, session_id)
        if products is None:
            continue
        store_map = store_map or products.store_map
        for shelf in products.shelves:
            known = findings.get(shelf.zone)
            # Two cameras may both have been given a zone. The one that saw
            # more shoppers pass it is the better measurement of it.
            if known is None or shelf.finding.funnel.passed > known.funnel.passed:
                findings[shelf.zone] = shelf.finding
    if store_map is None:
        return None, []
    return store_map, [_fixture(f, findings.get(f.zone)) for f in store_map.fixtures]


def _extent(outline, fixtures, cameras, tracks) -> list[float]:
    points: list[tuple[float, float]] = []
    if outline:
        points += [tuple(p) for p in outline]
    else:
        for fixture in fixtures:
            points += [tuple(p) for p in fixture["plan"] or []]
        for camera in cameras:
            points += [tuple(p) for p in camera["footprint"]]
        for track in tracks:
            points += list(zip(track["x"], track["y"]))
    if not points:
        return [0.0, 0.0, 10.0, 10.0]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def build_scene(store: EventStore, session_id: int) -> dict[str, Any]:
    """Everything the map draws for one recorded session."""
    session = store.session(session_id)
    if session is None:
        raise ValueError(f"no session {session_id}")

    members = store.fusion_members(session_id)
    member_ids = [m for m, _ in members] or [session_id]
    offsets = dict(members) or {session_id: 0.0}

    paths = store.paths(session_id)
    if not paths:
        raise ValueError(
            f"session {session_id} has no floor positions, so there is nobody to "
            "put on the map. Record with --floor, or fuse camera sessions that were"
        )

    # Which cameras were watching each fused sample, for the "seen by 2+"
    # marker. Looked up by time bucket, since every camera sample would
    # otherwise be compared with every fused one.
    buckets: dict[int, list[tuple[int, float, float, float]]] = {}
    if members:
        for member in member_ids:
            for _, t, x, y in store.position_rows(member):
                wall = t + offsets[member]
                buckets.setdefault(int(wall), []).append((member, wall, x, y))

    def watchers(t: float, x: float, y: float) -> int:
        if not members:
            return 1
        seen = set()
        for second in (int(t) - 1, int(t), int(t) + 1):
            for member, wall, ox, oy in buckets.get(second, ()):
                if abs(wall - t) <= _WATCH_WINDOW_S and (
                    (ox - x) ** 2 + (oy - y) ** 2 <= _WATCH_RADIUS**2
                ):
                    seen.add(member)
        return max(1, len(seen))

    start = min(points[0][0] for points in paths.values())
    tracks = []
    # Numbered in order of appearance. The stored track id stays in the store.
    for index, points in enumerate(sorted(paths.values(), key=lambda p: p[0][0]), 1):
        tracks.append(
            {
                "id": index,
                "tag": f"S-{index:03d}",
                "t": [round(t - start, 2) for t, _, _ in points],
                "x": [round(x, 3) for _, x, _ in points],
                "y": [round(y, 3) for _, _, y in points],
                "c": [watchers(t, x, y) for t, x, y in points],
            }
        )
    duration = max(track["t"][-1] for track in tracks)

    cameras = _cameras(store, member_ids)
    store_map, fixtures = _fixtures(store, member_ids)
    outline = [list(p) for p in store_map.outline] if store_map and store_map.outline else None

    if members:
        camera_tracks = sum(len(store.paths(m, min_points=1)) for m in member_ids)
        caption = f"session {session_id} / fused from {len(member_ids)} cameras"
        fusion = [
            ["camera tracks in", str(camera_tracks)],
            ["fused shoppers out", str(len(store.paths(session_id, min_points=1)))],
        ]
        note = (
            "Cameras are joined on floor position and time only. No appearance "
            "is used. A shopper count is an upper bound: two shoppers too close "
            "to tell apart are kept as separate tracks, never merged."
        )
    else:
        caption = f"session {session_id} / one camera"
        fusion = []
        note = (
            "One camera. Track ids last for this session only and carry no "
            "identity. A shopper count is an upper bound."
        )

    return {
        "site": {
            "id": store_map.site if store_map else "unmapped",
            "name": store_map.name if store_map else None,
        },
        "units": store_map.units if store_map else "m",
        "caption": caption,
        "outline": outline,
        "extent": _extent(outline, fixtures, cameras, tracks),
        "fixtures": fixtures,
        "cameras": cameras,
        "tracks": tracks,
        "duration": duration,
        # Every track, including the ones seen once and so not drawn as a path.
        "shoppers": len(store.paths(session_id, min_points=1)),
        "fusion": fusion,
        "note": note,
    }


def render_html(scene: dict[str, Any], start_s: float = 0.0, paused: bool = False) -> str:
    """A self-contained page: theme, renderer and scene in one file.

    Nothing is fetched, so it opens from disk on a machine with no network.
    """
    page = (STATIC / "map.html").read_text(encoding="utf-8")
    theme = (STATIC / "theme.css").read_text(encoding="utf-8")
    script = (STATIC / "map3d.js").read_text(encoding="utf-8")
    # A scene is data from the store map, which somebody typed. Closing tags
    # inside it must not be able to end the script block it is written into.
    data = json.dumps(scene, separators=(",", ":")).replace("</", "<\\/")
    setup = (
        f"<script>window.OPENV_SCENE={data};window.OPENV_START={float(start_s)};"
        f"window.OPENV_PAUSED={'true' if paused else 'false'};</script>"
    )
    return (
        page.replace("<!--THEME-->", f"<style>\n{theme}</style>")
        .replace("<!--SCENE-->", setup)
        .replace("<!--SCRIPT-->", f"<script>\n{script}</script>")
    )


def live_page() -> str:
    """The same page, served: it asks the server for its scene and positions."""
    page = (STATIC / "map.html").read_text(encoding="utf-8")
    links = '<link rel="stylesheet" href="/static/theme.css">'
    if (STATIC / "brand.css").exists():
        links += '\n<link rel="stylesheet" href="/static/brand.css">'
    return (
        page.replace("<!--THEME-->", links)
        .replace("<!--SCENE-->", "")
        .replace("<!--SCRIPT-->", '<script src="/static/map3d.js"></script>')
    )


class LiveMap:
    """Tails the event store and fuses the cameras writing to it.

    Each camera is its own `openv live` or `openv track` process writing floor
    positions to the same store. This reads what arrives, puts every camera on
    one clock from its session's start time, and runs the fuser over it.

    The clock is the data's, not this machine's: "now" is the newest
    observation seen. That keeps a replayed or delayed feed coherent, and it
    means a store with no camera writing simply shows nobody.
    """

    #: Rows are held this long before fusing, so two cameras whose writes
    #: land slightly out of order are still fed to the fuser in time order.
    REORDER_S = 0.7
    #: A camera that has written nothing for this long is taken off the map.
    #: Long on purpose: positions are the only sign of life a camera gives
    #: here, and an empty aisle is a camera seeing nobody, not a camera down.
    CAMERA_IDLE_S = 600.0

    def __init__(self, db_path: str | Path, poll_s: float = 0.4, **settings: float) -> None:
        self.db_path = Path(db_path)
        self.poll_s = poll_s
        self._fuser = FloorFuser(**settings)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_id: int | None = None
        self._epoch: datetime | None = None
        self._starts: dict[int, float] = {}
        self._buffer: list[Observation] = []
        self._now = 0.0
        self._camera_seen: dict[int, float] = {}
        self._positions: list[dict[str, Any]] = []
        self._tags: dict[int, int] = {}
        self._scene: dict[str, Any] | None = None
        self._scene_for: tuple[int, ...] | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="openv-live-map", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with EventStore(self.db_path) as store:
                    self.step(store)
            except Exception:  # noqa: BLE001 - a locked store is retried next tick
                pass
            self._stop.wait(self.poll_s)

    def _offset(self, store: EventStore, session_id: int) -> float | None:
        if session_id not in self._starts:
            session = store.session(session_id)
            if session is None or store.fusion_members(session_id):
                return None
            started = datetime.fromisoformat(session["started_at"])
            if self._epoch is None:
                self._epoch = started
            self._starts[session_id] = (started - self._epoch).total_seconds()
        return self._starts[session_id]

    def step(self, store: EventStore) -> None:
        """Read what has arrived since the last step and fuse it."""
        if self._last_id is None:
            # Start from now. Positions already in the store are history, and
            # replaying them here would put yesterday's shoppers on a live map.
            self._last_id = store.max_position_id()
            return

        for row in store.positions_since(self._last_id):
            self._last_id = int(row["id"])
            offset = self._offset(store, int(row["session_id"]))
            if offset is None:
                continue
            wall = offset + float(row["t_s"])
            self._buffer.append(
                Observation(
                    camera=str(row["session_id"]), track_id=int(row["track_id"]),
                    t_s=wall, x=float(row["x"]), y=float(row["y"]),
                )
            )
            self._now = max(self._now, wall)
            self._camera_seen[int(row["session_id"])] = wall

        ready = [o for o in self._buffer if o.t_s <= self._now - self.REORDER_S]
        self._buffer = [o for o in self._buffer if o.t_s > self._now - self.REORDER_S]
        for obs in sorted(ready, key=lambda o: (o.t_s, o.camera, o.track_id)):
            self._fuser.update(obs)

        positions = []
        for p in self._fuser.live(self._now - self.REORDER_S):
            tag = self._tags.setdefault(p.fused_id, len(self._tags) + 1)
            positions.append(
                {"id": tag, "tag": f"S-{tag:03d}", "x": round(p.x, 3), "y": round(p.y, 3),
                 "cameras": len(p.cameras)}
            )
        active = tuple(sorted(self.active_cameras()))
        scene = self._scene
        if active != self._scene_for:
            scene = self._build_scene(store, list(active))
        with self._lock:
            self._positions = positions
            self._scene, self._scene_for = scene, active

    def active_cameras(self) -> list[int]:
        return [
            session_id
            for session_id, seen in self._camera_seen.items()
            if self._now - seen <= self.CAMERA_IDLE_S
        ]

    def _build_scene(self, store: EventStore, sessions: list[int]) -> dict[str, Any]:
        # With no camera writing yet, the newest session's map still gives the
        # page a store to draw while it waits.
        latest = store.latest_session_id()
        source = sessions or ([latest] if latest is not None else [])
        cameras = _cameras(store, sessions)
        store_map, fixtures = _fixtures(store, source)
        outline = (
            [list(p) for p in store_map.outline] if store_map and store_map.outline else None
        )
        return {
            "site": {
                "id": store_map.site if store_map else "unmapped",
                "name": store_map.name if store_map else None,
            },
            "units": store_map.units if store_map else "m",
            "caption": f"live / {len(cameras)} camera{'' if len(cameras) == 1 else 's'}",
            "outline": outline,
            "extent": _extent(outline, fixtures, cameras, []),
            "fixtures": fixtures,
            "cameras": cameras,
            "tracks": [],
            "duration": 0,
            "shoppers": 0,
            "fusion": [],
            "note": (
                "Cameras are joined on floor position and time only. No "
                "appearance is used. A shopper new to a camera appears about a "
                "second late, which is the fuser waiting to see where they went."
            ),
        }

    def scene_payload(self) -> dict[str, Any]:
        with self._lock:
            if self._scene is not None:
                return self._scene
        with EventStore(self.db_path) as store:
            return self._build_scene(store, [])

    def live_payload(self) -> dict[str, Any]:
        with self._lock:
            report = self._fuser.report
            return {
                "positions": list(self._positions),
                "cameras": len(self._scene_for or ()),
                "shoppers": report.fused_tracks,
                "ambiguous": report.ambiguous,
                "updated": time.time(),
            }


def create_map_app(live: LiveMap):
    """The live map as a small web app."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    from openv.web.server import mount_static

    app = FastAPI(title="OpenV store map", docs_url=None, redoc_url=None)

    @app.on_event("startup")
    def _start() -> None:
        live.start()

    @app.on_event("shutdown")
    def _stop() -> None:
        live.stop()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return live_page()

    @app.get("/api/scene")
    def scene() -> JSONResponse:
        return JSONResponse(live.scene_payload())

    @app.get("/api/live")
    def positions() -> JSONResponse:
        return JSONResponse(live.live_payload())

    mount_static(app)
    return app
