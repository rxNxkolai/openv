"""The store map: what sits on each shelf, and where each fixture is on the plan.

OpenV measures zones. A camera on the ceiling cannot read a label, and SKU
vision from that mount is not reliable enough to sell (CLAUDE.md constraint 3).
The retailer already knows what is on every shelf, though: that is what a
planogram is. If their system says which item sits in a shelf position, and
OpenV says a reach happened in the zone covering that position, the join is
product-level insight with no product vision at all and no new hardware.

So this file holds **store data, not shopper data**. A fixture, the shelf zone
a camera watches it through, where it stands on the floor plan, and the items
on it. Nothing here describes a shopper, and the privacy posture is untouched
by it.

## What the join can and cannot say

A reach is recorded against a zone. If the zone covers one item, the reach is a
reach for that item. If the zone covers six, the reach was for one of the six
and nothing in the data says which. Dividing the reaches by facing count would
produce a number per item that looks measured and is invented, so nothing here
does that. `products.py` reports an item as measured only when its zone holds
nothing else, and otherwise says the zone is shared and by how many.

That makes zone granularity the granularity of product insight. One polygon
over a whole bay answers bay-level questions; one polygon per section answers
section-level ones. The map does not hide that trade, it is the reason to draw
smaller zones.

## The file

```json
{
  "store": {"id": "store-014", "name": "Riverside", "units": "m",
            "outline": [[0, 0], [30, 0], [30, 18], [0, 18]]},
  "fixtures": [
    {"zone": "left-shelf", "label": "Aisle 6, left bay",
     "plan": [[2, 4], [2.6, 4], [2.6, 12], [2, 12]],
     "products": [
       {"sku": "100234", "product": "Oat cereal 500g", "brand": "Northfield",
        "category": "cereal", "facings": 4, "level": "eye"}
     ]}
  ]
}
```

`zone` names a shelf zone from the zones file. `plan` and `outline` are
optional and are in the same floor units a `floor.json` calibrates to, which is
what lets recorded shopper positions be drawn over the fixtures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openv.zones import ZoneSet

Polygon = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Placement:
    """One item on one fixture, in the retailer's own terms."""

    zone: str
    sku: str
    product: str
    brand: str
    category: str
    facings: int = 1
    level: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "sku": self.sku,
            "product": self.product,
            "brand": self.brand,
            "category": self.category,
            "facings": self.facings,
        }
        if self.level is not None:
            out["level"] = self.level
        return out


@dataclass(frozen=True)
class Fixture:
    """A shelf, the zone it is watched through, and where it is on the plan."""

    zone: str
    label: str | None = None
    plan: Polygon | None = None
    placements: tuple[Placement, ...] = ()

    @property
    def brands(self) -> tuple[str, ...]:
        return tuple(sorted({p.brand for p in self.placements}))

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({p.category for p in self.placements}))

    @property
    def facings(self) -> int:
        return sum(p.facings for p in self.placements)


@dataclass(frozen=True)
class StoreMap:
    site: str
    name: str | None = None
    units: str = "m"
    outline: Polygon | None = None
    fixtures: tuple[Fixture, ...] = field(default_factory=tuple)

    @property
    def placements(self) -> tuple[Placement, ...]:
        return tuple(p for f in self.fixtures for p in f.placements)

    def fixture(self, zone: str) -> Fixture | None:
        return next((f for f in self.fixtures if f.zone == zone), None)

    def problems(self, zones: ZoneSet) -> list[str]:
        """Reasons this map cannot be used with these zones.

        A fixture pointing at a zone that does not exist, or at a floor zone,
        would attach products to something no reach can ever land in. That
        reads later as a product nobody touches, which is the most damaging
        thing this layer can say about an item and must not be a typo.
        """
        by_name = {z.name: z for z in zones}
        out = []
        for fixture in self.fixtures:
            zone = by_name.get(fixture.zone)
            if zone is None:
                out.append(
                    f"fixture names zone {fixture.zone!r}, which is not in the zones "
                    f"({', '.join(sorted(by_name)) or 'none drawn'})"
                )
            elif not zone.is_shelf:
                out.append(
                    f"fixture names zone {fixture.zone!r}, which is kind "
                    f"{zone.kind!r}. Products sit on shelf zones, because a reach "
                    "is a wrist in a shelf zone"
                )
        return out

    def unmapped(self, zones: ZoneSet) -> list[str]:
        """Shelf zones with nothing mapped onto them.

        Not an error. Their zone-level numbers are still measured; they just
        cannot be spoken about in product terms.
        """
        mapped = {f.zone for f in self.fixtures if f.placements}
        return [z.name for z in zones.shelf if z.name not in mapped]

    def as_dict(self) -> dict[str, Any]:
        store: dict[str, Any] = {"id": self.site, "units": self.units}
        if self.name is not None:
            store["name"] = self.name
        if self.outline is not None:
            store["outline"] = [list(p) for p in self.outline]
        fixtures = []
        for f in self.fixtures:
            entry: dict[str, Any] = {"zone": f.zone}
            if f.label is not None:
                entry["label"] = f.label
            if f.plan is not None:
                entry["plan"] = [list(p) for p in f.plan]
            entry["products"] = [p.as_dict() for p in f.placements]
            fixtures.append(entry)
        return {"store": store, "fixtures": fixtures}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, raw: Any, where: str = "store map") -> StoreMap:
        if not isinstance(raw, dict) or "store" not in raw or "fixtures" not in raw:
            raise ValueError(f"{where} needs a 'store' and a 'fixtures' key")

        store = raw["store"]
        if not isinstance(store, dict) or not str(store.get("id", "")).strip():
            # The id is what tells two stores' exports apart once they are in
            # the same room. A default here would merge them silently.
            raise ValueError(f"{where}: 'store' needs an 'id'")
        if not isinstance(raw["fixtures"], list):
            raise ValueError(f"{where}: 'fixtures' must be a list")

        fixtures = []
        for index, entry in enumerate(raw["fixtures"]):
            label = f"{where}: fixture {index + 1}"
            if not isinstance(entry, dict) or not str(entry.get("zone", "")).strip():
                raise ValueError(f"{label} needs a 'zone'")
            zone = str(entry["zone"])
            placements = []
            products = entry.get("products", [])
            if not isinstance(products, list):
                raise ValueError(f"{label} ({zone}): 'products' must be a list")
            for item in products:
                placements.append(_placement(item, zone, f"{label} ({zone})"))
            skus = [p.sku for p in placements]
            repeated = sorted({s for s in skus if skus.count(s) > 1})
            if repeated:
                raise ValueError(
                    f"{label} ({zone}) lists {repeated} more than once. Use "
                    "'facings' for several facings of one item"
                )
            fixtures.append(
                Fixture(
                    zone=zone,
                    label=entry.get("label"),
                    plan=_polygon(entry.get("plan"), f"{label} ({zone}) plan"),
                    placements=tuple(placements),
                )
            )

        zones = [f.zone for f in fixtures]
        repeated = sorted({z for z in zones if zones.count(z) > 1})
        if repeated:
            raise ValueError(f"{where}: more than one fixture for zone {repeated}")

        return cls(
            site=str(store["id"]).strip(),
            name=store.get("name"),
            units=str(store.get("units", "m")),
            outline=_polygon(store.get("outline"), f"{where}: store outline"),
            fixtures=tuple(fixtures),
        )

    @classmethod
    def load(cls, path: str | Path) -> StoreMap:
        # Same rule as ZoneSet.load: every failure names the file, because this
        # one is edited by hand or exported from somebody else's system.
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"could not read {path}: {exc}") from exc
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path} is not valid JSON: {exc.msg} at line {exc.lineno}"
            ) from exc
        return cls.from_dict(raw, where=str(path))


def _placement(item: Any, zone: str, where: str) -> Placement:
    if not isinstance(item, dict):
        raise ValueError(f"{where}: a product must be an object")
    missing = [k for k in ("sku", "brand", "category") if not str(item.get(k, "")).strip()]
    if missing:
        raise ValueError(f"{where}: a product is missing {', '.join(missing)}")
    try:
        facings = int(item.get("facings", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: facings must be a whole number") from exc
    if facings < 1:
        raise ValueError(f"{where}: facings must be at least 1, got {facings}")
    sku = str(item["sku"]).strip()
    return Placement(
        zone=zone,
        sku=sku,
        product=str(item.get("product") or sku).strip(),
        brand=str(item["brand"]).strip(),
        category=str(item["category"]).strip(),
        facings=facings,
        level=(str(item["level"]).strip() if item.get("level") else None),
    )


def _polygon(raw: Any, where: str) -> Polygon | None:
    if raw is None:
        return None
    try:
        polygon = tuple((float(x), float(y)) for x, y in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} is malformed: {exc}") from exc
    if len(polygon) < 3:
        raise ValueError(f"{where} needs at least 3 points, got {len(polygon)}")
    return polygon
