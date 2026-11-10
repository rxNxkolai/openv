"""A synthetic store, so the product layer can be tried with no camera and no GPU.

Everything above the `FrameResult` seam is deterministic code over an event
store, and none of it needs a model or footage to run. This writes event stores
for two invented stores so `analyze`, `measure`, `digest`, `export`, `rollup`
and `planview` all have something to read thirty seconds after cloning.

**These numbers are invented.** They are drawn from a seeded generator to look
like a small grocery store with one dead endcap, one shelf people stop at and
do not reach into, and one healthy shelf. They demonstrate what the commands
say about a store shaped like that. They are evidence of nothing about any real
store, and every session written here is labelled `synthetic` in its source so
it cannot be mistaken for a measurement later.

The brands are made up as well.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from openv.events import ZoneVisit
from openv.floor import FloorPosition
from openv.store import EventStore
from openv.storemap import StoreMap
from openv.zones import Zone, ZoneSet

FPS = 10.0

ZONES = ZoneSet(
    zones=(
        Zone("aisle-6", ((300, 100), (700, 100), (700, 900), (300, 900)), "floor"),
        Zone("shelf-cereal", ((100, 100), (300, 100), (300, 500), (100, 500)), "shelf"),
        Zone("shelf-oats", ((100, 500), (300, 500), (300, 900), (100, 900)), "shelf"),
        Zone("aisle-9", ((1100, 100), (1500, 100), (1500, 900), (1100, 900)), "floor"),
        Zone("shelf-snacks", ((900, 100), (1100, 100), (1100, 900), (900, 900)), "shelf"),
        Zone("endcap-floor", ((700, 900), (1100, 900), (1100, 1050), (700, 1050)), "floor"),
        Zone("shelf-endcap", ((800, 780), (1000, 780), (1000, 900), (800, 900)), "shelf"),
    )
)

# Where each floor zone sits on the plan: (x centre, y start, y end), in metres.
_PLAN = {
    "aisle-6": (3.2, 2.0, 10.0),
    "aisle-9": (11.0, 2.0, 10.0),
    "endcap-floor": (7.5, 0.3, 0.9),
}


def store_map(site: str, name: str) -> StoreMap:
    def item(sku, product, brand, category, facings, level):
        return {"sku": sku, "product": product, "brand": brand, "category": category,
                "facings": facings, "level": level}

    return StoreMap.from_dict(
        {
            "store": {"id": site, "name": name, "units": "m",
                      "outline": [[0, 0], [16, 0], [16, 11], [0, 11]]},
            "fixtures": [
                {"zone": "shelf-cereal", "label": "Aisle 6, cereal",
                 "plan": [[1, 6], [2, 6], [2, 10], [1, 10]],
                 "products": [
                     item("100234", "Oat cereal 500g", "Northfield", "cereal", 6, "eye"),
                     item("100241", "Bran flakes 750g", "Northfield", "cereal", 4, "mid"),
                 ]},
                # Two brands in one zone, on purpose: the case a reach cannot
                # be attributed in, which the export has to say out loud.
                {"zone": "shelf-oats", "label": "Aisle 6, oats",
                 "plan": [[1, 2], [2, 2], [2, 6], [1, 6]],
                 "products": [
                     item("100260", "Rolled oats 1kg", "Northfield", "cereal", 3, "mid"),
                     item("200618", "Porridge oats 1kg", "Oatwright", "cereal", 3, "bottom"),
                 ]},
                {"zone": "shelf-snacks", "label": "Aisle 9, snacks",
                 "plan": [[9, 2], [10, 2], [10, 10], [9, 10]],
                 "products": [
                     item("300102", "Sea salt crisps 150g", "Harrow & Pike", "snacks", 8, "eye"),
                     item("300117", "Roasted nuts 200g", "Harrow & Pike", "snacks", 4, "top"),
                 ]},
                {"zone": "shelf-endcap", "label": "Endcap",
                 "plan": [[6, 1], [9, 1], [9, 1.7], [6, 1.7]],
                 "products": [
                     item("300140", "Rice cakes 130g", "Tallow Lane", "snacks", 12, "eye"),
                 ]},
            ],
        }
    )


def _span(track_id: int, zone: str, entered_s: float, exited_s: float) -> ZoneVisit:
    return ZoneVisit(
        track_id=track_id,
        zone=zone,
        entered_frame=int(entered_s * FPS),
        entered_s=entered_s,
        exited_frame=int(exited_s * FPS),
        exited_s=exited_s,
    )


def _session(store: EventStore, label: str, site: StoreMap, seed: int, endcap_reach: float) -> int:
    """One synthetic day. `endcap_reach` is the only thing that differs between
    the 'before' and 'after' days, which is what `measure` is then asked about."""
    rng = random.Random(seed)
    session_id = store.start_session(
        f"synthetic: {label}", fps=FPS, width=1600, height=1100,
        zones=ZONES, pose=True, store_map=site,
    )
    visits: list[ZoneVisit] = []
    reaches: list[ZoneVisit] = []
    positions: list[FloorPosition] = []
    track = 0

    def shopper(aisle: str, dwell: float) -> int:
        nonlocal track
        track += 1
        visits.append(_span(track, aisle, 0.0, dwell))
        x, y0, y1 = _PLAN[aisle]
        phase = rng.random()
        steps = max(1, round(dwell))
        for i in range(steps):
            positions.append(
                FloorPosition(
                    track_id=track, frame=int(i * FPS), t_s=float(i),
                    x=x + rng.uniform(-0.9, 0.9),
                    y=y0 + (y1 - y0) * min(1.0, (i + phase) / steps),
                )
            )
        return track

    for _ in range(120):
        dwell = rng.choice([1.2, 1.5, 3.5, 6.0, 9.0])
        who = shopper("aisle-6", dwell)
        if dwell >= 3.5 and rng.random() < 0.52:
            reaches.append(_span(who, "shelf-cereal", 1.0, 1.8))
        if dwell >= 3.5 and rng.random() < 0.20:
            reaches.append(_span(who, "shelf-oats", 2.0, 2.6))

    for _ in range(110):  # people stop here and almost nobody reaches
        who = shopper("aisle-9", rng.choice([1.5, 3.0, 4.0, 5.0, 6.5]))
        if rng.random() < 0.05:
            reaches.append(_span(who, "shelf-snacks", 1.0, 1.6))

    for _ in range(140):  # the endcap everybody walks past
        who = shopper("endcap-floor", rng.choice([0.8, 1.0, 1.2, 1.4, 2.5]))
        if rng.random() < endcap_reach:
            reaches.append(_span(who, "shelf-endcap", 0.3, 0.9))

    store.add_visits(session_id, visits)
    store.add_reaches(session_id, reaches)
    store.add_positions(session_id, positions)
    return session_id


# --------------------------------------------------------------------------
# Three cameras over one store, for fusion and the 3D map
# --------------------------------------------------------------------------

#: Each camera's view of the floor as a rectangle on the plan, in metres:
#: (min x, min y, max x, max y). They overlap on purpose. The overlaps are
#: where fusion has two opinions about one shopper, and the edges are where a
#: shopper is handed from one camera to the next.
CAMERAS = {
    "cam-a": (0.0, 0.0, 6.6, 11.0),
    "cam-b": (5.6, 1.6, 16.0, 11.0),
    "cam-c": (0.0, 0.0, 16.0, 3.0),
}
#: The cameras were not started in the same instant, as real ones never are.
CAMERA_START_S = {"cam-a": 0.0, "cam-b": 0.4, "cam-c": 0.9}
_FRAME = (1600, 1100)

#: Zones each camera was given, as plan rectangles: (kind, x0, y0, x1, y1).
CAMERA_ZONES = {
    "cam-a": {
        "aisle-6": ("floor", 2.0, 2.0, 4.6, 10.0),
        "shelf-cereal": ("shelf", 1.0, 6.0, 2.0, 10.0),
        "shelf-oats": ("shelf", 1.0, 2.0, 2.0, 6.0),
    },
    "cam-b": {
        "aisle-9": ("floor", 10.0, 2.0, 12.2, 10.0),
        "shelf-snacks": ("shelf", 9.0, 2.0, 10.0, 10.0),
    },
    "cam-c": {
        "endcap-floor": ("floor", 5.5, 0.1, 9.5, 1.0),
        "shelf-endcap": ("shelf", 6.0, 1.0, 9.0, 1.7),
    },
}


def _to_pixels(camera: str, x: float, y: float) -> tuple[float, float]:
    """An overhead camera: its rectangle of floor fills its frame, y flipped."""
    x0, y0, x1, y1 = CAMERAS[camera]
    return (
        (x - x0) / (x1 - x0) * _FRAME[0],
        (y1 - y) / (y1 - y0) * _FRAME[1],
    )


def camera_calibration(camera: str) -> dict:
    """The floor calibration a camera with that view would have been given."""
    x0, y0, x1, y1 = CAMERAS[camera]
    floor_points = [
        (x0, y0), (x1, y0), (x1, y1), (x0, y1), ((x0 + x1) / 2, (y0 + y1) / 2),
    ]
    return {
        "units": "m",
        "correspondences": [
            {"image": list(_to_pixels(camera, x, y)), "floor": [x, y]}
            for x, y in floor_points
        ],
    }


def camera_zones(camera: str) -> ZoneSet:
    zones = []
    for name, (kind, x0, y0, x1, y1) in CAMERA_ZONES[camera].items():
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        zones.append(Zone(name, tuple(_to_pixels(camera, x, y) for x, y in corners), kind))
    return ZoneSet(zones=tuple(zones))


@dataclass
class _Walker:
    """One invented shopper: where they were at every moment, and what they did."""

    keys: list[tuple[float, float, float]]  # (t, x, y), time increasing
    reaches: list[tuple[str, float, float]] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.keys[0][0]

    @property
    def end(self) -> float:
        return self.keys[-1][0]

    def at(self, t: float) -> tuple[float, float]:
        for (t0, x0, y0), (t1, x1, y1) in zip(self.keys, self.keys[1:]):
            if t0 <= t <= t1:
                f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                return (x0 + f * (x1 - x0), y0 + f * (y1 - y0))
        return self.keys[-1][1:]


def _walk(rng: random.Random, start: float, stops: list[tuple]) -> _Walker:
    """A walker from a list of (x, y) waypoints, or (x, y, dwell, shelf, p_reach)."""
    speed = rng.uniform(1.0, 1.5)
    t = start
    keys = [(t, stops[0][0], stops[0][1])]
    reaches = []
    for stop in stops[1:]:
        x, y = stop[0], stop[1]
        t += math.hypot(x - keys[-1][1], y - keys[-1][2]) / speed
        keys.append((t, x, y))
        if len(stop) > 2:
            dwell, shelf, p_reach = stop[2], stop[3], stop[4]
            if shelf and rng.random() < p_reach and dwell >= 2.0:
                reaches.append((shelf, t + dwell * 0.3, t + dwell * 0.3 + 0.8))
            t += dwell
            keys.append((t, x, y))
    return _Walker(keys=keys, reaches=reaches)


def _routes(rng: random.Random, start: float) -> _Walker:
    """One of four trips through the store, each crossing a camera seam."""
    door = (14.6 + rng.uniform(-0.4, 0.4), 0.3)
    lane = rng.uniform(0.45, 0.85)  # the walkway along the front
    a6 = 3.0 + rng.uniform(-0.5, 0.9)
    a9 = 11.0 + rng.uniform(-0.6, 0.8)
    trip = rng.random()
    if trip < 0.34:  # aisle 6 and back out
        y = rng.uniform(3.0, 9.5)
        shelf = "shelf-cereal" if y >= 6.0 else "shelf-oats"
        p = 0.62 if shelf == "shelf-cereal" else 0.22
        dwell = rng.choice([0.0, 2.5, 4.0, 7.0])
        stops = [door, (a6, lane), (a6, y, dwell, shelf, p), (a6, lane), door]
    elif trip < 0.62:  # aisle 9, where people stop and do not reach
        y = rng.uniform(3.0, 9.5)
        dwell = rng.choice([2.0, 3.0, 4.5, 6.0])
        stops = [door, (a9, lane), (a9, y, dwell, "shelf-snacks", 0.06), (a9, lane), door]
    elif trip < 0.80:  # up aisle 9, across the back, down aisle 6
        y6 = rng.uniform(6.2, 9.5)
        stops = [
            door, (a9, lane), (a9, 10.4), (a6, 10.4),
            (a6, y6, rng.choice([2.5, 5.0]), "shelf-cereal", 0.55), (a6, lane), door,
        ]
    else:  # walks past the endcap and leaves
        stops = [door, (rng.uniform(6.5, 8.5), lane, rng.choice([0.0, 0.0, 1.5]), None, 0.0),
                 (1.0, lane), (1.0, 0.2)]
    return _Walker(keys=_walk(rng, start, stops).keys, reaches=_walk_reaches(rng, start, stops))


def _walk_reaches(rng: random.Random, start: float, stops: list[tuple]) -> list:
    # Reaches are drawn on a second pass with its own generator state, so the
    # path a walker takes does not depend on whether they happened to reach.
    return _walk(random.Random(rng.random()), start, stops).reaches


def simulate_cameras(
    seed: int = 21, walkers: int = 110, duration_s: float = 600.0
) -> dict:
    """What three cameras would each have recorded of one set of shoppers.

    Returns per camera: floor observations as (track id, t, x, y), visits and
    reaches against that camera's zones, and the number of shoppers there
    really were. Each camera numbers its own tracks, starts its own clock, and
    places a shopper a few centimetres from where the next camera does.
    """
    rng = random.Random(seed)
    people = []
    for _ in range(walkers):
        template = _routes(rng, 0.0)
        start = rng.uniform(0.0, max(1.0, duration_s - template.end))
        people.append(_walk_shift(template, start))

    out = {"truth": len(people), "cameras": {}}
    step = 0.1
    for camera, (x0, y0, x1, y1) in CAMERAS.items():
        cam_rng = random.Random(f"{seed}:{camera}")
        clock = CAMERA_START_S[camera]
        next_track = 1
        positions: list[FloorPosition] = []
        visits: list[ZoneVisit] = []
        reaches: list[ZoneVisit] = []
        # Which invented shopper each camera track really was. Returned so a
        # test can score fusion against the truth, and never written anywhere:
        # a real camera has no such column, and the store has nowhere to put it.
        truth: dict[int, int] = {}

        for index, walker in enumerate(people):
            track = None
            last_sample = -1e9
            open_visits: dict[str, float] = {}

            def close(zone: str, track_id: int, until: float) -> None:
                entered = open_visits.pop(zone)
                if until - entered >= 0.2:
                    visits.append(_span(track_id, zone, entered, until))

            t = max(walker.start, clock)
            while t <= walker.end:
                x, y = walker.at(t)
                local = t - clock
                inside = x0 <= x <= x1 and y0 <= y <= y1
                if inside and track is not None and cam_rng.random() < 0.0008:
                    # The tracker loses them for an instant and starts a new id.
                    for zone in list(open_visits):
                        close(zone, track, local)
                    track = None
                if inside and track is None:
                    track = next_track
                    next_track += 1
                    last_sample = -1e9
                    truth[track] = index
                if not inside and track is not None:
                    for zone in list(open_visits):
                        close(zone, track, local)
                    track = None
                if track is not None:
                    if local - last_sample >= 1.0:
                        last_sample = local
                        positions.append(
                            FloorPosition(
                                track_id=track, frame=int(local * FPS), t_s=round(local, 2),
                                x=x + cam_rng.gauss(0.0, 0.07), y=y + cam_rng.gauss(0.0, 0.07),
                            )
                        )
                    for zone, (kind, zx0, zy0, zx1, zy1) in CAMERA_ZONES[camera].items():
                        if kind != "floor":
                            continue
                        within = zx0 <= x <= zx1 and zy0 <= y <= zy1
                        if within and zone not in open_visits:
                            open_visits[zone] = local
                        elif not within and zone in open_visits:
                            close(zone, track, local)
                    for shelf, r0, r1 in walker.reaches:
                        if shelf in CAMERA_ZONES[camera] and abs(t - r0) < step / 2:
                            reaches.append(_span(track, shelf, r0 - clock, r1 - clock))
                t += step
            if track is not None:
                for zone in list(open_visits):
                    close(zone, track, walker.end - clock)

        positions.sort(key=lambda p: p.t_s)
        out["cameras"][camera] = {
            "positions": positions, "visits": visits, "reaches": reaches,
            "tracks": next_track - 1, "truth": truth,
        }
    return out


def _walk_shift(walker: _Walker, by: float) -> _Walker:
    return _Walker(
        keys=[(t + by, x, y) for t, x, y in walker.keys],
        reaches=[(shelf, a + by, b + by) for shelf, a, b in walker.reaches],
    )


def write_camera_demo(path: str | Path, seed: int = 21) -> dict:
    """Write the three-camera scene as three sessions in one event store."""
    scene = simulate_cameras(seed=seed)
    site = store_map("store-014", "Riverside")
    base = datetime(2026, 3, 2, 10, 0, 0, tzinfo=timezone.utc)
    sessions = {}
    with EventStore(path) as store:
        for camera, recorded in scene["cameras"].items():
            session_id = store.start_session(
                f"synthetic: {camera}", fps=FPS, width=_FRAME[0], height=_FRAME[1],
                calibration=camera_calibration(camera), zones=camera_zones(camera),
                pose=True, store_map=site,
                started_at=base + timedelta(seconds=CAMERA_START_S[camera]),
            )
            store.add_visits(session_id, recorded["visits"])
            store.add_reaches(session_id, recorded["reaches"])
            store.add_positions(session_id, recorded["positions"])
            sessions[camera] = session_id
    return {"truth": scene["truth"], "sessions": sessions}


def write_demo(out_dir: str | Path) -> dict[str, Path]:
    """Write two synthetic stores. Refuses to write over an existing store.

    Each store gets a 'before' and an 'after' session. At the first store the
    endcap is dead before and reached after, so `openv measure shelf-endcap`
    has a real change to find.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "store-014": out_dir / "store-014.db",
        "store-022": out_dir / "store-022.db",
        "cameras": out_dir / "cameras.db",
        "zones": out_dir / "demo.zones.json",
        "map": out_dir / "demo.map.json",
    }
    existing = [str(p) for p in paths.values() if p.suffix == ".db" and p.exists()]
    if existing:
        # Appending two more invented sessions to a store somebody has since
        # recorded real footage into would be a quiet way to ruin it.
        raise FileExistsError(
            f"{', '.join(existing)} already exist. Delete them or pick another --out"
        )

    riverside = store_map("store-014", "Riverside")
    harbour = store_map("store-022", "Harbour")
    ZONES.save(paths["zones"])
    riverside.save(paths["map"])

    with EventStore(paths["store-014"]) as store:
        _session(store, "week 1", riverside, seed=7, endcap_reach=0.0)
        _session(store, "week 2, endcap moved to eye level", riverside, seed=8, endcap_reach=0.16)
    with EventStore(paths["store-022"]) as store:
        _session(store, "week 1", harbour, seed=11, endcap_reach=0.12)
        _session(store, "week 2", harbour, seed=12, endcap_reach=0.12)
    write_camera_demo(paths["cameras"])
    return paths
