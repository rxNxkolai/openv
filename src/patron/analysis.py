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

import statistics
from dataclasses import dataclass

from patron.store import EventStore

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
    kind: str  # underperforming | low_stop_rate | healthy | insufficient_data
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


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def analyze(
    store: EventStore,
    session_id: int | None = None,
    stop_threshold_s: float = 2.0,
) -> StoreAnalysis:
    """Compute the funnel for every shelf zone and rank what is worth acting on."""
    pairs = store.shelf_floor_pairs(session_id)
    floor_rows = {r["zone"]: r for r in store.zone_summary(session_id)}
    reach_rows = {r["zone"]: r for r in store.reach_summary(session_id)}

    funnels: list[ZoneFunnel] = []
    for shelf_zone in sorted(reach_rows) or ():
        floor_zone = pairs.get(shelf_zone)
        floor = floor_rows.get(floor_zone) if floor_zone else None

        passed = int(floor["shoppers"]) if floor else 0
        mean_dwell = float(floor["mean_dwell_s"] or 0.0) if floor else 0.0
        stopped = (
            store.shoppers_stopping(floor_zone, stop_threshold_s, session_id)
            if floor_zone
            else 0
        )
        reached = int(reach_rows[shelf_zone]["shoppers"])

        funnels.append(
            ZoneFunnel(
                shelf_zone=shelf_zone,
                floor_zone=floor_zone,
                passed=passed,
                stopped=stopped,
                reached=reached,
                mean_dwell_s=mean_dwell,
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


def _assess(funnel: ZoneFunnel, median_reach: float | None) -> Finding:
    if not funnel.has_confidence:
        return Finding(
            zone=funnel.shelf_zone,
            kind="insufficient_data",
            severity="none",
            headline=(
                f"{funnel.shelf_zone}: only {funnel.passed} shoppers observed, "
                f"below the {MIN_SHOPPERS_FOR_CONFIDENCE} needed to call a rate"
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
