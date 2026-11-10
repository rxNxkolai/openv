"""Product, brand and category numbers, from zone measurements and the store map.

`analysis.py` says what happened at a shelf zone. `storemap.py` says what the
retailer put on it. This joins the two, deterministically and with no model,
and it is the layer a brand buys: a category manager cares about the endcap, a
brand cares about its own facing on it.

## The one rule

**A number is attributed to a brand only when the brand had the zone to
itself.** A reach lands in a zone. If the zone holds one brand, every reach
there was a reach for that brand, and the funnel is the brand's funnel. If it
holds three, each reach was for one of the three and the data does not say
which.

The tempting move is to split the zone's reaches by share of facings. That
produces a tidy number for every brand, and every one of them is an assumption
about shopper behaviour presented as a measurement of it. It is also the exact
assumption a brand is paying to have tested. So a shared zone is reported as
shared: the zone's own funnel, who else is in it, and the brand's share of the
facings as a fact about the shelf and nothing more.

The same rule holds one level down. An item is measured only when its zone
holds nothing else.

A brand measured across several zones counts each shopper once. Someone who
walked past two of its fixtures is one shopper who had two chances, not two
shoppers, and track ids are stable within a session so that is exact rather
than estimated.

Every rate here is the same guarded rate `analysis.py` produces. Below
`MIN_SHOPPERS_FOR_CONFIDENCE` shoppers there is a count and no rate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from openv.analysis import Finding, StoreAnalysis, ZoneFunnel, analyze, assess
from openv.store import EventStore
from openv.storemap import Fixture, Placement, StoreMap

EXCLUSIVE = "exclusive"
SHARED = "shared"
UNMAPPED = "unmapped"

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "none": 3}


@dataclass(frozen=True)
class Shelf:
    """One measured shelf zone, and what the map says is on it."""

    finding: Finding
    fixture: Fixture | None

    @property
    def zone(self) -> str:
        return self.finding.funnel.shelf_zone

    @property
    def placements(self) -> tuple[Placement, ...]:
        return self.fixture.placements if self.fixture is not None else ()

    @property
    def brands(self) -> tuple[str, ...]:
        return self.fixture.brands if self.fixture is not None else ()

    @property
    def categories(self) -> tuple[str, ...]:
        return self.fixture.categories if self.fixture is not None else ()

    @property
    def brand_attribution(self) -> str:
        """Whether this zone's numbers belong to one brand."""
        if not self.placements:
            return UNMAPPED
        return EXCLUSIVE if len(self.brands) == 1 else SHARED


@dataclass(frozen=True)
class SharedZone:
    """A zone a brand or category sits in alongside others."""

    zone: str
    facings: int
    zone_facings: int
    others: int

    @property
    def share_of_facings(self) -> float:
        # A fact about the shelf. Not a share of the reaches, which is the
        # number it must never be mistaken for.
        return self.facings / self.zone_facings if self.zone_facings else 0.0


@dataclass(frozen=True)
class Rollup:
    """A brand or a category, measured over the zones it has to itself."""

    name: str
    by: str  # brand | category
    measured_zones: tuple[str, ...]
    shared_zones: tuple[SharedZone, ...]
    unwatched_zones: tuple[str, ...]
    facings: int
    finding: Finding | None

    @property
    def measured(self) -> bool:
        return self.finding is not None

    @property
    def funnel(self) -> ZoneFunnel | None:
        return self.finding.funnel if self.finding is not None else None


@dataclass(frozen=True)
class Item:
    """One placement, with the zone numbers it can or cannot claim."""

    placement: Placement
    finding: Finding | None  # None when the zone was not measured this session
    attribution: str  # exclusive | shared | unwatched
    shares_zone_with: int


@dataclass(frozen=True)
class ProductAnalysis:
    session_id: int
    store_map: StoreMap
    analysis: StoreAnalysis
    shelves: tuple[Shelf, ...]
    brands: tuple[Rollup, ...]
    categories: tuple[Rollup, ...]
    items: tuple[Item, ...]
    #: Measured shelf zones the map says nothing about.
    unmapped_zones: tuple[str, ...]
    #: Fixtures in the map whose zone was not measured in this session. Their
    #: products have no numbers, which is not the same as nobody reaching.
    unwatched_zones: tuple[str, ...]

    def brand(self, name: str) -> Rollup | None:
        wanted = name.strip().casefold()
        return next((b for b in self.brands if b.name.casefold() == wanted), None)


def analyze_products(
    store: EventStore, session_id: int, stop_threshold_s: float = 2.0
) -> ProductAnalysis | None:
    """Join one session's measurements to the store map it was recorded with.

    None when the session has no store map, because there is then nothing to
    say in product terms and the zone-level analysis is the whole answer.

    One session at a time, deliberately. The map is a fact about a session
    (planograms are reset), and counting each shopper once across a brand's
    fixtures only means something inside one.
    """
    store_map = store.session_store_map(session_id)
    if store_map is None:
        return None

    analysis = analyze(store, session_id, stop_threshold_s=stop_threshold_s)
    by_zone = {f.funnel.shelf_zone: f for f in analysis.findings}

    shelves = tuple(
        Shelf(finding=f, fixture=store_map.fixture(f.funnel.shelf_zone))
        for f in analysis.findings
    )
    unmapped = tuple(s.zone for s in shelves if not s.placements)
    unwatched = tuple(
        f.zone for f in store_map.fixtures if f.placements and f.zone not in by_zone
    )

    def rollups(by: str) -> tuple[Rollup, ...]:
        def key(p: Placement) -> str:
            return p.brand if by == "brand" else p.category

        names = sorted({key(p) for p in store_map.placements})
        out = []
        for name in names:
            measured: list[str] = []
            shared: list[SharedZone] = []
            missing: list[str] = []
            facings = 0
            for fixture in store_map.fixtures:
                mine = [p for p in fixture.placements if key(p) == name]
                if not mine:
                    continue
                facings += sum(p.facings for p in mine)
                if fixture.zone not in by_zone:
                    missing.append(fixture.zone)
                    continue
                keys_here = {key(p) for p in fixture.placements}
                if len(keys_here) == 1:
                    measured.append(fixture.zone)
                else:
                    shared.append(
                        SharedZone(
                            zone=fixture.zone,
                            facings=sum(p.facings for p in mine),
                            zone_facings=fixture.facings,
                            others=len(keys_here) - 1,
                        )
                    )
            out.append(
                Rollup(
                    name=name,
                    by=by,
                    measured_zones=tuple(measured),
                    shared_zones=tuple(shared),
                    unwatched_zones=tuple(missing),
                    facings=facings,
                    finding=_rollup_finding(
                        store, session_id, name, measured, by_zone,
                        analysis.median_reach_rate, stop_threshold_s,
                    ),
                )
            )
        return tuple(
            sorted(
                out,
                key=lambda r: (
                    _SEVERITY_ORDER[r.finding.severity] if r.finding else 4,
                    -(r.finding.funnel.passed if r.finding else 0),
                    r.name,
                ),
            )
        )

    items = []
    for fixture in store_map.fixtures:
        finding = by_zone.get(fixture.zone)
        for placement in fixture.placements:
            others = len(fixture.placements) - 1
            if finding is None:
                attribution = "unwatched"
            else:
                attribution = EXCLUSIVE if others == 0 else SHARED
            items.append(
                Item(
                    placement=placement,
                    finding=finding,
                    attribution=attribution,
                    shares_zone_with=others,
                )
            )

    return ProductAnalysis(
        session_id=session_id,
        store_map=store_map,
        analysis=analysis,
        shelves=shelves,
        brands=rollups("brand"),
        categories=rollups("category"),
        items=tuple(items),
        unmapped_zones=unmapped,
        unwatched_zones=unwatched,
    )


def _rollup_finding(
    store: EventStore,
    session_id: int,
    name: str,
    zones: list[str],
    by_zone: dict[str, Finding],
    median_reach: float | None,
    stop_threshold_s: float,
) -> Finding | None:
    """The funnel for a set of zones one brand or category has to itself.

    Each shopper is counted once across the set. For a single zone this is the
    zone's own funnel, number for number, which a test holds it to.
    """
    if not zones:
        return None
    if len(zones) == 1:
        # One zone: the brand's funnel is that zone's funnel and nothing is
        # recounted, so the two can never disagree by a shopper.
        funnel = replace(by_zone[zones[0]].funnel, shelf_zone=name, paired_by=None)
        return _reword(assess(funnel, median_reach), name, 1)

    floors = sorted(
        {by_zone[z].funnel.floor_zone for z in zones if by_zone[z].funnel.floor_zone}
    )
    funnel = ZoneFunnel(
        shelf_zone=name,
        floor_zone=", ".join(floors) if floors else None,
        passed=store.shoppers_in(floors, session_id),
        stopped=store.shoppers_in(floors, session_id, min_dwell_s=stop_threshold_s),
        reached=store.reachers_in(zones, session_id),
        # Not carried across zones. A mean of means over different aisles is a
        # number nobody could reproduce by hand, and it is not needed here.
        mean_dwell_s=0.0,
    )
    return _reword(assess(funnel, median_reach), name, len(zones))


def _reword(finding: Finding, name: str, zones: int) -> Finding:
    if finding.kind != "dead":
        return finding
    # `assess` words this for a fixture. The numbers are the same.
    where = "its shelf" if zones == 1 else f"its {zones} shelves"
    return replace(
        finding,
        headline=(
            f"{name}: {finding.funnel.passed} shoppers walked past {where} "
            "and not one reached"
        ),
    )
