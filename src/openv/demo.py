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

import random
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
        "zones": out_dir / "demo.zones.json",
        "map": out_dir / "demo.map.json",
    }
    existing = [str(p) for key, p in paths.items() if key.startswith("store-") and p.exists()]
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
    return paths
