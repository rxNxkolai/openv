"""The data product: what a retailer or a brand actually receives.

Everything else in OpenV serves the store that owns the cameras. This is the
file that goes to somebody else, so it is the one place where "anonymised" has
to be a property of the output and not a description of the intent.

**What is in it.** Zones, the retailer's own product data for those zones, and
counts of shoppers who passed, stopped and reached, for one session of one
store. Rates are the guarded rates the rest of the system uses, so a row with
too few shoppers carries its counts and a stated refusal, never a percentage.

**What cannot be in it.** There is no code path from a track id, a position, a
path or a frame to this file. It is built from `ProductAnalysis`, which holds
aggregates only, and a test walks every key of the output to hold it to that.
Nothing here says anything about any individual, and nothing links a shopper
between two sessions or two stores, because nothing upstream can.

**Two audiences, one builder.** The retailer's export is the whole store. A
brand's export is narrowed to what concerns that brand: the zones it is in, its
own items, and its category as a benchmark. Other brands are counted, not
named. That narrowing is commercial confidentiality between a retailer and its
suppliers and has nothing to do with shopper privacy, which is identical in
both. The retailer decides what a brand is shown; this is the narrow default.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from openv.analysis import MIN_SHOPPERS_FOR_CONFIDENCE, Finding, ZoneFunnel
from openv.products import EXCLUSIVE, ProductAnalysis, Rollup, Shelf, analyze_products
from openv.store import EventStore
from openv.tools import Rate

SCHEMA_ID = "openv.export/1"

#: Stated in every file, because the buyer's counsel reads the file and not the
#: repository. Each line is something the pipeline does not produce at all.
EXCLUDES = (
    "no video, frames or stills",
    "no track identifiers, and no row per shopper",
    "no individual positions or paths",
    "no faces and no biometric identifier of any kind",
    "no demographic inference and no emotion inference",
    "no link between a shopper in one session and the same shopper in another",
)

COUNT_NOTE = (
    "Shopper counts are an upper bound: tracking can give a new id to a shopper "
    "who was hidden and seen again. A reach is a hand entering the shelf zone "
    "with the arm extended. It does not confirm a pickup or a purchase."
)


class ExportUnavailable(ValueError):
    """There is nothing exportable, and the message says which thing is missing."""


def _finding(finding: Finding) -> dict[str, Any]:
    return {
        "kind": finding.kind,
        "severity": finding.severity,
        "headline": finding.headline,
    }


def _counts(funnel: ZoneFunnel) -> dict[str, Any]:
    return {
        "passed": funnel.passed,
        "stopped": funnel.stopped,
        "reached": funnel.reached,
        "stop_rate": Rate.from_funnel(funnel, funnel.stopped, funnel.stop_rate).as_dict(),
        "reach_rate": Rate.from_funnel(funnel, funnel.reached, funnel.reach_rate).as_dict(),
    }


def zone_row(shelf: Shelf, only_brand: str | None = None) -> dict[str, Any]:
    funnel = shelf.finding.funnel
    placements = shelf.placements
    out: dict[str, Any] = {
        "zone": shelf.zone,
        "label": shelf.fixture.label if shelf.fixture is not None else None,
        "floor_zone": funnel.floor_zone,
        "paired_by": funnel.paired_by,
        "brand_attribution": shelf.brand_attribution,
        **_counts(funnel),
        "mean_dwell_s": round(funnel.mean_dwell_s, 2),
        "finding": _finding(shelf.finding),
    }
    if only_brand is None:
        out["products"] = [p.as_dict() for p in placements]
    else:
        out["products"] = [p.as_dict() for p in placements if p.brand == only_brand]
        out["other_brands_in_zone"] = len(set(shelf.brands) - {only_brand})
    return out


def rollup_row(rollup: Rollup, median: float | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        rollup.by: rollup.name,
        "facings": rollup.facings,
        "measured": rollup.measured,
        "measured_zones": list(rollup.measured_zones),
        "shared_zones": [
            {
                "zone": s.zone,
                "facings": s.facings,
                "zone_facings": s.zone_facings,
                "share_of_facings": round(s.share_of_facings, 3),
                f"other_{'brands' if rollup.by == 'brand' else 'categories'}": s.others,
            }
            for s in rollup.shared_zones
        ],
        "unwatched_zones": list(rollup.unwatched_zones),
    }
    if rollup.finding is not None:
        out.update(_counts(rollup.finding.funnel))
        out["finding"] = _finding(rollup.finding)
        out["store_median_reach_rate"] = median
    else:
        out["not_measured"] = _why_not_measured(rollup)
    return out


def _why_not_measured(rollup: Rollup) -> str:
    kind = "brands" if rollup.by == "brand" else "categories"
    if rollup.shared_zones:
        return (
            f"every watched zone holding {rollup.name} also holds other {kind}, "
            "so no reach can be attributed to it. A share of facings is a fact "
            "about the shelf and is not a share of the reaches. A zone drawn "
            "over its section alone would measure it"
        )
    return (
        f"no zone holding {rollup.name} was watched with pose on in this session"
    )


def build_export(
    store: EventStore,
    session_id: int,
    brand: str | None = None,
    stop_threshold_s: float = 2.0,
) -> dict[str, Any]:
    """The export for one session, for the retailer or narrowed to one brand."""
    session = store.session(session_id)
    if session is None:
        raise ExportUnavailable(f"no session {session_id}")

    products = analyze_products(store, session_id, stop_threshold_s=stop_threshold_s)
    if products is None:
        raise ExportUnavailable(
            f"session {session_id} has no store map recorded, so there is nothing "
            "to say in product terms. Record with `track --map`, or attach one "
            "with `storemap --attach`"
        )

    return export_from(products, session, brand=brand)


def export_from(
    products: ProductAnalysis, session: dict[str, Any], brand: str | None = None
) -> dict[str, Any]:
    store_map = products.store_map
    median = products.analysis.median_reach_rate

    shelves = products.shelves
    brands = products.brands
    categories = products.categories
    items = products.items
    scope: dict[str, Any] = {"audience": "retailer"}

    if brand is not None:
        rollup = products.brand(brand)
        if rollup is None:
            known = ", ".join(b.name for b in products.brands) or "none"
            raise ExportUnavailable(
                f"no brand {brand!r} in the store map for this session. "
                f"Brands on it: {known}"
            )
        name = rollup.name
        own_categories = {p.category for p in store_map.placements if p.brand == name}
        shelves = tuple(s for s in shelves if name in s.brands)
        brands = (rollup,)
        categories = tuple(c for c in categories if c.name in own_categories)
        items = tuple(i for i in items if i.placement.brand == name)
        scope = {
            "audience": "brand",
            "brand": name,
            "note": (
                "Narrowed to this brand. Other brands are counted, never named. "
                "Category rows include every brand in the category and are the "
                "benchmark."
            ),
        }
        brand = name

    return {
        "schema": SCHEMA_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "site": {"id": store_map.site, "name": store_map.name},
        "session": {
            "id": session["id"],
            "started_at": session["started_at"],
            "pose": bool(session["pose"]),
        },
        "scope": scope,
        "privacy": {
            "unit": "aggregate counts per zone, for one session of one store",
            "min_shoppers_for_a_rate": MIN_SHOPPERS_FOR_CONFIDENCE,
            "excludes": list(EXCLUDES),
        },
        "notes": [
            COUNT_NOTE,
            "A brand or item is measured only where it has a zone to itself. "
            "Reaches in a shared zone are never split between its occupants.",
            "The benchmark is this store's own median reach rate across its "
            "shelf zones, not an industry figure.",
        ],
        "total_shoppers": products.analysis.total_shoppers,
        "store_median_reach_rate": median,
        "zones": [zone_row(s, brand) for s in shelves],
        "brands": [rollup_row(b, median) for b in brands],
        "categories": [rollup_row(c, median) for c in categories],
        "products": [
            {
                **item.placement.as_dict(),
                "zone": item.placement.zone,
                "attribution": item.attribution,
                "shares_zone_with": item.shares_zone_with,
                # An item claims the zone's numbers only when it is alone in
                # it. Otherwise the zone row above is the most that is known.
                **(
                    {**_counts(item.finding.funnel), "finding": _finding(item.finding)}
                    if item.attribution == EXCLUSIVE and item.finding is not None
                    else {}
                ),
            }
            for item in items
        ],
        "unmapped_zones": [] if brand is not None else list(products.unmapped_zones),
        "unwatched_zones": [
            z
            for z in products.unwatched_zones
            if brand is None or brand in (store_map.fixture(z).brands if store_map.fixture(z) else ())
        ],
    }


def _rate_text(rate: dict[str, Any]) -> str:
    return "not enough" if rate["value"] is None else f"{rate['value'] * 100:.0f}%"


def render(export: dict[str, Any]) -> str:
    """The same content as a table, for a terminal. Nothing is recomputed."""
    site = export["site"]
    lines = [
        f"{site['name'] or site['id']}  (store {site['id']}, session "
        f"{export['session']['id']}, {export['session']['started_at']})",
    ]
    if export["scope"]["audience"] == "brand":
        lines.append(f"narrowed to brand: {export['scope']['brand']}")
    median = export["store_median_reach_rate"]
    lines.append(
        f"shoppers observed {export['total_shoppers']}, store median reach "
        + ("n/a" if median is None else f"{median * 100:.0f}%")
    )

    marks = {"high": "!!", "medium": " !", "low": "  ", "none": " ?"}

    def table(title: str, rows: list[dict[str, Any]], key: str) -> None:
        if not rows:
            return
        lines.append("")
        header = (
            f"   {title:<22}{'facings':>8}{'passed':>8}{'stopped':>9}"
            f"{'reached':>9}{'reach':>12}  measured on"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for row in rows:
            if row["measured"]:
                mark = marks[row["finding"]["severity"]]
                lines.append(
                    f"{mark} {row[key]:<22}{row['facings']:>8}{row['passed']:>8}"
                    f"{row['stopped']:>9}{row['reached']:>9}"
                    f"{_rate_text(row['reach_rate']):>12}  "
                    f"{', '.join(row['measured_zones'])}"
                )
            else:
                shared = ", ".join(s["zone"] for s in row["shared_zones"])
                where = f"shared: {shared}" if shared else "not watched"
                lines.append(
                    f" - {row[key]:<22}{row['facings']:>8}{'':>8}{'':>9}{'':>9}"
                    f"{'no number':>12}  {where}"
                )

    table("brand", export["brands"], "brand")
    table("category", export["categories"], "category")

    lines.append("")
    header = f"   {'shelf zone':<22}{'passed':>8}{'reached':>9}{'reach':>12}  on it"
    lines.append(header)
    lines.append("-" * len(header))
    for zone in export["zones"]:
        names = ", ".join(sorted({p["brand"] for p in zone["products"]})) or "nothing mapped"
        others = zone.get("other_brands_in_zone")
        if others:
            names += f" (+{others} other brand{'' if others == 1 else 's'})"
        lines.append(
            f"{marks[zone['finding']['severity']]} {zone['zone']:<22}{zone['passed']:>8}"
            f"{zone['reached']:>9}{_rate_text(zone['reach_rate']):>12}  {names}"
        )

    shared_only = [b for b in export["brands"] if not b["measured"] and b["shared_zones"]]
    if shared_only:
        lines.append("")
        lines.append(
            "no number: the brand shares every zone it is in, and a reach in a "
            "shared zone"
        )
        lines.append(
            "cannot be given to one occupant. Draw a zone over its section "
            "alone to measure it."
        )
    if export["unwatched_zones"]:
        lines.append("")
        lines.append(
            "on the map but not measured this session: "
            + ", ".join(export["unwatched_zones"])
        )
    if export["unmapped_zones"]:
        lines.append("")
        lines.append(
            "measured but nothing mapped onto them: " + ", ".join(export["unmapped_zones"])
        )

    lines.append("")
    lines.append(
        f"!! high   ! medium   ? below {export['privacy']['min_shoppers_for_a_rate']} "
        "shoppers, not a rate yet"
    )
    return "\n".join(lines)
