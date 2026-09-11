"""Deterministic funnel analysis.

This is the layer that turns visit and reach rows into the sentence a retailer
recognises: "aisle 6 end-cap: 71% walked past, 4% reached." No model involved.
That is deliberate. The agent in M3 writes the recommendation, but the numbers it
reasons over have to be computed, reproducible, and defensible on their own, or
the recommendation is just prose about nothing.

Shelf zones are paired to floor zones **from the data**, not from configuration:
a shelf's reachers were standing somewhere while they reached, and the floor zone
they were most often standing in is the one that shelf serves. That means the
pairing cannot drift out of sync with reality the way a hand-maintained mapping
would.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from openv.store import EventStore
from openv.zones import Zone, ZoneSet

# Below this many shoppers, a rate is noise. Reporting "0% reach rate" off three
# shoppers as a finding would be worse than reporting nothing.
MIN_SHOPPERS_FOR_CONFIDENCE = 30

# A zone whose reach rate is this fraction of the store median (or worse) is
# underperforming rather than merely below average.
UNDERPERFORM_RATIO = 0.5


@dataclass(frozen=True)
class ZoneFunnel:
    """The pass-by to engagement funnel for one shelf and the floor it serves."""

    shelf_zone: str
    floor_zone: str | None
    passed: int
    stopped: int
    reached: int
    mean_dwell_s: float
    # How the floor zone was chosen: 'reachers' when learned from where
    # reachers stood, 'geometry' when nobody reached and the nearest floor
    # polygon had to stand in. It travels with the funnel because a number
    # whose denominator was picked by a fallback should say so.
    paired_by: str | None = None

    @property
    def stop_rate(self) -> float | None:
        return self.stopped / self.passed if self.passed else None

    @property
    def pass_by_rate(self) -> float | None:
        stop = self.stop_rate
        return None if stop is None else 1.0 - stop

    @property
    def reach_rate(self) -> float | None:
        return self.reached / self.passed if self.passed else None

    @property
    def has_confidence(self) -> bool:
        return self.passed >= MIN_SHOPPERS_FOR_CONFIDENCE


@dataclass(frozen=True)
class Finding:
    """One thing worth telling a retailer, with the evidence attached."""

    zone: str
    kind: str  # dead | underperforming | low_stop_rate | healthy | insufficient_data
    severity: str  # high | medium | low | none
    headline: str
    funnel: ZoneFunnel
    benchmark_reach_rate: float | None

    @property
    def confident(self) -> bool:
        return self.funnel.has_confidence


@dataclass(frozen=True)
class StoreAnalysis:
    session_id: int | None
    findings: tuple[Finding, ...]
    median_reach_rate: float | None
    total_shoppers: int

    @property
    def actionable(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity in ("high", "medium"))


# Two-sided significance for a change. Not a tuning knob: 0.05 is the convention
# a retailer's own analysts will assume, and moving it to make a result look
# better is the exact failure this whole comparison exists to prevent.
SIGNIFICANCE = 0.05

# The normal approximation behind the test needs a few expected outcomes in every
# cell. Below this it quietly stops being valid, so the verdict says so instead.
MIN_EXPECTED_PER_CELL = 5


@dataclass(frozen=True)
class Change:
    """One zone measured twice, and whether the difference means anything.

    A retailer changes an endcap and wants to know if it worked. The honest
    answer is usually "we cannot tell yet", and a tool that always produces a
    confident percentage difference would be worse than useless: it would make
    noise look like evidence, repeatedly, in a document someone plans against.
    """

    zone: str
    before: ZoneFunnel
    after: ZoneFunnel
    verdict: str  # improved | worsened | indistinguishable | not_enough_data | not_comparable
    reason: str
    delta: float | None = None
    p_value: float | None = None

    @property
    def conclusive(self) -> bool:
        return self.verdict in ("improved", "worsened")


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _two_proportion_p(x1: int, n1: int, x2: int, n2: int) -> float | None:
    """Two-sided p for the difference between two proportions.

    Pooled two-proportion z-test. Returns None when the normal approximation
    does not apply, rather than a number that looks like the others but is not
    comparable to them.
    """
    if n1 <= 0 or n2 <= 0:
        return None

    pooled = (x1 + x2) / (n1 + n2)
    if pooled <= 0.0 or pooled >= 1.0:
        # Every shopper reached, or none did, in both periods together. There is
        # no variance to test against.
        return None

    expected = [pooled * n1, (1 - pooled) * n1, pooled * n2, (1 - pooled) * n2]
    if min(expected) < MIN_EXPECTED_PER_CELL:
        return None

    standard_error = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if standard_error == 0.0:
        return None

    z = (x2 / n2 - x1 / n1) / standard_error
    return 2.0 * (1.0 - _normal_cdf(abs(z)))


def measure_change(
    store: EventStore, zone: str, before_session: int, after_session: int
) -> Change | None:
    """Compare one shelf zone across two sessions.

    Returns None if the zone was not measured in either session, because
    "changed from nothing to nothing" is not a measurement. A shelf that was
    drawn, watched with pose on, and never reached **was** measured: zero is
    the number, and comparing it against the fixed fixture is the point.

    Refuses outright when the polygon moved between the sessions. The live
    console already resets its counts on a zone change because numbers
    gathered against different boundaries are not comparable; this is the
    same rule applied where it has consequences, in the persisted store.
    """
    before = _funnel_for(store, zone, before_session)
    after = _funnel_for(store, zone, after_session)
    if before is None or after is None:
        return None

    if store.zone_moved(zone, before_session, after_session):
        return Change(
            zone=zone,
            before=before,
            after=after,
            verdict="not_comparable",
            reason=(
                f"the polygon for {zone} changed between session {before_session} "
                f"and {after_session}, so the two measurements describe different "
                "boundaries. Redraw it as it was, or start counting from here"
            ),
        )

    if not (before.has_confidence and after.has_confidence):
        thin = [
            label
            for label, funnel in (("before", before), ("after", after))
            if not funnel.has_confidence
        ]
        return Change(
            zone=zone,
            before=before,
            after=after,
            verdict="not_enough_data",
            reason=(
                f"{' and '.join(thin)} saw fewer than {MIN_SHOPPERS_FOR_CONFIDENCE} "
                f"shoppers, so there is no rate to compare"
            ),
        )

    p_value = _two_proportion_p(
        before.reached, before.passed, after.reached, after.passed
    )
    delta = after.reach_rate - before.reach_rate

    if p_value is None:
        return Change(
            zone=zone,
            before=before,
            after=after,
            verdict="not_enough_data",
            reason=(
                "too few reaches for the comparison to be valid, whatever the "
                "shopper counts. More traffic will not fix this; more reaches would"
            ),
            delta=delta,
        )

    if p_value >= SIGNIFICANCE:
        return Change(
            zone=zone,
            before=before,
            after=after,
            verdict="indistinguishable",
            reason=(
                f"the difference is within what this many shoppers would produce "
                f"by chance (p = {p_value:.2f})"
            ),
            delta=delta,
            p_value=p_value,
        )

    return Change(
        zone=zone,
        before=before,
        after=after,
        verdict="improved" if delta > 0 else "worsened",
        reason=f"larger than chance would produce at this sample size (p = {p_value:.3f})",
        delta=delta,
        p_value=p_value,
    )


def _funnel_for(store: EventStore, zone: str, session_id: int) -> ZoneFunnel | None:
    for finding in analyze(store, session_id).findings:
        if finding.funnel.shelf_zone == zone:
            return finding.funnel
    return None


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def analyze(
    store: EventStore,
    session_id: int | None = None,
    stop_threshold_s: float = 2.0,
) -> StoreAnalysis:
    """Compute the funnel for every shelf zone and rank what is worth acting on.

    A shelf zone is in the analysis if it recorded a reach, or if it was drawn
    in a session where pose ran and could have. The second case is the dead
    fixture, and before zones were persisted it was the one shelf the product
    could not mention: no reach rows meant no row at all, so the worse a
    fixture performed the less was said about it.
    """
    pairs = store.shelf_floor_pairs(session_id)
    floor_rows = {r["zone"]: r for r in store.zone_summary(session_id)}
    reach_rows = {r["zone"]: r for r in store.reach_summary(session_id)}
    drawn = _zones_for(store, session_id)

    funnels: list[ZoneFunnel] = []
    for shelf_zone in sorted(set(reach_rows) | store.shelf_zones_measured(session_id)):
        paired_by: str | None = None
        floor_zone = pairs.get(shelf_zone)
        if floor_zone is not None:
            paired_by = "reachers"
        elif drawn is not None:
            floor_zone = _nearest_floor_zone(shelf_zone, drawn)
            paired_by = "geometry" if floor_zone is not None else None
        floor = floor_rows.get(floor_zone) if floor_zone else None

        passed = int(floor["shoppers"]) if floor else 0
        mean_dwell = float(floor["mean_dwell_s"] or 0.0) if floor else 0.0
        stopped = (
            store.shoppers_stopping(floor_zone, stop_threshold_s, session_id)
            if floor_zone
            else 0
        )
        reach = reach_rows.get(shelf_zone)
        reached = int(reach["shoppers"]) if reach is not None else 0

        funnels.append(
            ZoneFunnel(
                shelf_zone=shelf_zone,
                floor_zone=floor_zone,
                passed=passed,
                stopped=stopped,
                reached=reached,
                mean_dwell_s=mean_dwell,
                paired_by=paired_by,
            )
        )

    # The benchmark is the store's own median, not an industry number. A retailer
    # can argue with an external benchmark; they cannot argue with their own other
    # aisles measured the same way on the same day.
    rates = [
        f.reach_rate
        for f in funnels
        if f.reach_rate is not None and f.has_confidence
    ]
    median_reach = statistics.median(rates) if rates else None

    findings = tuple(
        sorted(
            (_assess(f, median_reach) for f in funnels),
            key=lambda f: (
                {"high": 0, "medium": 1, "low": 2, "none": 3}[f.severity],
                -f.funnel.passed,
            ),
        )
    )

    return StoreAnalysis(
        session_id=session_id,
        findings=findings,
        median_reach_rate=median_reach,
        total_shoppers=store.total_shoppers(session_id),
    )


def _zones_for(store: EventStore, session_id: int | None) -> ZoneSet | None:
    """The zone set to pair against when nobody reached.

    For a whole-store analysis the latest session's zones stand for the
    store, which is right when the layout is stable and stated here because
    it is an assumption.
    """
    if session_id is None:
        session_id = store.latest_session_id()
    return store.session_zones(session_id) if session_id is not None else None


def _nearest_floor_zone(shelf_name: str, zones: ZoneSet) -> str | None:
    """The floor zone a shelf most plausibly faces, from the drawing alone.

    Only used when no reacher ever stood anywhere, which is exactly the dead
    fixture. The pairing learned from reachers is better evidence and takes
    over the moment a reach is recorded. Nearest is measured from the shelf
    polygon's centroid to each floor polygon, signed so that a centroid
    inside a floor zone beats one merely close to it.
    """
    shelf = next((z for z in zones.shelf if z.name == shelf_name), None)
    if shelf is None or not len(zones.floor):
        return None
    cx = sum(x for x, _ in shelf.polygon) / len(shelf.polygon)
    cy = sum(y for _, y in shelf.polygon) / len(shelf.polygon)
    return max(zones.floor, key=lambda z: _signed_distance(z, (cx, cy))).name


def _signed_distance(zone: Zone, point: tuple[float, float]) -> float:
    import cv2

    return float(cv2.pointPolygonTest(zone.contour, (float(point[0]), float(point[1])), True))


def _assess(funnel: ZoneFunnel, median_reach: float | None) -> Finding:
    if not funnel.has_confidence:
        return Finding(
            zone=funnel.shelf_zone,
            kind="insufficient_data",
            severity="none",
            headline=(
                f"{funnel.shelf_zone}: only {funnel.passed} "
                f"shopper{'' if funnel.passed == 1 else 's'} observed, "
                f"below the {MIN_SHOPPERS_FOR_CONFIDENCE} needed to call a rate"
            ),
            funnel=funnel,
            benchmark_reach_rate=median_reach,
        )

    # Nobody reached at all. This does not need a benchmark to be a finding:
    # zero engagement from a confident number of shoppers is the most
    # actionable thing a merchandiser can be told about a fixture, and it is
    # reported before the median so a store with one shelf still hears it.
    if funnel.reached == 0:
        return Finding(
            zone=funnel.shelf_zone,
            kind="dead",
            severity="high" if funnel.passed >= MIN_SHOPPERS_FOR_CONFIDENCE * 2 else "medium",
            headline=(
                f"{funnel.shelf_zone}: {funnel.passed} shoppers walked past and not "
                f"one reached. A dead fixture"
            ),
            funnel=funnel,
            benchmark_reach_rate=median_reach,
        )

    reach = funnel.reach_rate
    if reach is not None and median_reach is not None and median_reach > 0:
        if reach <= median_reach * UNDERPERFORM_RATIO:
            return Finding(
                zone=funnel.shelf_zone,
                kind="underperforming",
                severity="high" if funnel.passed >= MIN_SHOPPERS_FOR_CONFIDENCE * 2 else "medium",
                headline=(
                    f"{funnel.shelf_zone}: {_pct(funnel.pass_by_rate)} of "
                    f"{funnel.passed} shoppers walked past, only {_pct(reach)} reached. "
                    f"Store median is {_pct(median_reach)}"
                ),
                funnel=funnel,
                benchmark_reach_rate=median_reach,
            )

    # Plenty of people stop but few engage: an attention problem at the shelf
    # face rather than a traffic problem in the aisle.
    stop = funnel.stop_rate
    if stop is not None and reach is not None and stop >= 0.4 and reach < stop * 0.25:
        return Finding(
            zone=funnel.shelf_zone,
            kind="low_stop_rate",
            severity="medium",
            headline=(
                f"{funnel.shelf_zone}: {_pct(stop)} of shoppers stopped but only "
                f"{_pct(reach)} reached. They looked and did not engage"
            ),
            funnel=funnel,
            benchmark_reach_rate=median_reach,
        )

    return Finding(
        zone=funnel.shelf_zone,
        kind="healthy",
        severity="low",
        headline=(
            f"{funnel.shelf_zone}: {_pct(reach)} reach rate across "
            f"{funnel.passed} shoppers, at or above the store median"
        ),
        funnel=funnel,
        benchmark_reach_rate=median_reach,
    )
