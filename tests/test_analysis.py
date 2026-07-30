"""The funnel math is what the agent reasons over and what a retailer argues with."""

import pytest

from patron.analysis import MIN_SHOPPERS_FOR_CONFIDENCE, analyze
from patron.events import ZoneSpan
from patron.store import EventStore


def span(track_id: int, zone: str, start: float, end: float) -> ZoneSpan:
    return ZoneSpan(
        track_id=track_id,
        zone=zone,
        entered_frame=int(start * 30),
        entered_s=start,
        exited_frame=int(end * 30),
        exited_s=end,
    )


def build(tmp_path, visits, reaches):
    store = EventStore(tmp_path / "a.db")
    session = store.start_session("t.mp4", fps=30.0, width=1920, height=1080)
    store.add_visits(session, visits)
    store.add_reaches(session, reaches)
    return store, session


def test_shelf_is_paired_to_the_aisle_its_reachers_stood_in(tmp_path):
    # Shopper 1 stands in aisle-6 and reaches shelf-a. Shopper 2 stands in
    # aisle-9 at a different time. The pairing must follow the overlap.
    visits = [span(1, "aisle-6", 0, 10), span(2, "aisle-9", 100, 110)]
    reaches = [span(1, "shelf-a", 3, 4)]
    store, session = build(tmp_path, visits, reaches)

    assert store.shelf_floor_pairs(session) == {"shelf-a": "aisle-6"}
    store.close()


def test_pairing_ignores_a_zone_the_shopper_was_not_in_at_the_time(tmp_path):
    # Same shopper visits both aisles, but the reach only overlaps the second.
    visits = [span(1, "aisle-6", 0, 10), span(1, "aisle-9", 20, 30)]
    reaches = [span(1, "shelf-a", 22, 23)]
    store, session = build(tmp_path, visits, reaches)

    assert store.shelf_floor_pairs(session) == {"shelf-a": "aisle-9"}
    store.close()


def test_funnel_rates_are_computed_over_distinct_shoppers(tmp_path):
    # 40 shoppers pass through aisle-6. 20 linger. 4 reach the shelf.
    visits = [span(i, "aisle-6", i * 10, i * 10 + (5 if i < 20 else 0.5)) for i in range(40)]
    reaches = [span(i, "shelf-a", i * 10 + 1, i * 10 + 2) for i in range(4)]
    store, session = build(tmp_path, visits, reaches)

    result = analyze(store, session)
    funnel = result.findings[0].funnel

    assert funnel.floor_zone == "aisle-6"
    assert funnel.passed == 40
    assert funnel.stopped == 20
    assert funnel.reached == 4
    assert funnel.reach_rate == pytest.approx(0.10)
    assert funnel.stop_rate == pytest.approx(0.50)
    assert funnel.pass_by_rate == pytest.approx(0.50)
    store.close()


def test_small_samples_are_flagged_rather_than_reported_as_a_rate(tmp_path):
    # Three shoppers, none reached. "0% reach rate" would be a lie dressed as data.
    visits = [span(i, "aisle-6", i * 10, i * 10 + 5) for i in range(3)]
    reaches = [span(0, "shelf-a", 1, 2)]
    store, session = build(tmp_path, visits, reaches)

    result = analyze(store, session)
    finding = result.findings[0]

    assert finding.kind == "insufficient_data"
    assert finding.severity == "none"
    assert finding.confident is False
    assert str(MIN_SHOPPERS_FOR_CONFIDENCE) in finding.headline
    assert result.actionable == ()
    store.close()


def test_zone_far_below_the_store_median_is_flagged_high(tmp_path):
    visits, reaches = [], []
    # Two healthy aisles at 40% reach, one bad aisle at 5%.
    for aisle, shelf, reach_count in (
        ("aisle-1", "shelf-1", 40),
        ("aisle-2", "shelf-2", 40),
        ("aisle-3", "shelf-3", 5),
    ):
        base = hash(aisle) % 1000
        for i in range(100):
            tid = base * 1000 + i
            visits.append(span(tid, aisle, i * 10, i * 10 + 5))
            if i < reach_count:
                reaches.append(span(tid, shelf, i * 10 + 1, i * 10 + 2))

    store, session = build(tmp_path, visits, reaches)
    result = analyze(store, session)

    assert result.median_reach_rate == pytest.approx(0.40)
    worst = result.findings[0]
    assert worst.zone == "shelf-3"
    assert worst.kind == "underperforming"
    assert worst.severity == "high"
    assert "5%" in worst.headline
    # The healthy zones must not be dressed up as problems.
    assert {f.kind for f in result.findings[1:]} == {"healthy"}
    store.close()


def test_shoppers_who_stop_but_never_reach_are_their_own_finding(tmp_path):
    # One zone only: no median to compare against, so this must be caught by the
    # stopped-but-not-engaged rule rather than by benchmarking.
    visits = [span(i, "aisle-6", i * 10, i * 10 + 8) for i in range(60)]
    reaches = [span(i, "shelf-a", i * 10 + 1, i * 10 + 2) for i in range(3)]
    store, session = build(tmp_path, visits, reaches)

    finding = analyze(store, session).findings[0]

    assert finding.kind == "low_stop_rate"
    assert finding.severity == "medium"
    assert finding.funnel.stopped == 60
    store.close()


def test_findings_are_ranked_worst_first(tmp_path):
    visits, reaches = [], []
    for aisle, shelf, reach_count, traffic in (
        ("aisle-1", "shelf-1", 40, 100),   # healthy
        ("aisle-2", "shelf-2", 2, 100),    # bad, high traffic
        ("aisle-3", "shelf-3", 1, 40),     # bad, lower traffic
    ):
        base = abs(hash(aisle)) % 1000
        for i in range(traffic):
            tid = base * 1000 + i
            visits.append(span(tid, aisle, i * 10, i * 10 + 5))
            if i < reach_count:
                reaches.append(span(tid, shelf, i * 10 + 1, i * 10 + 2))

    store, session = build(tmp_path, visits, reaches)
    ranked = [f.zone for f in analyze(store, session).findings]

    # Highest severity first; within a severity, the busiest zone leads because
    # fixing it moves more shoppers.
    assert ranked[0] == "shelf-2"
    assert ranked.index("shelf-2") < ranked.index("shelf-3") < ranked.index("shelf-1")
    store.close()


def test_analysis_of_an_empty_store_does_not_invent_findings(tmp_path):
    store, session = build(tmp_path, [], [])
    result = analyze(store, session)

    assert result.findings == ()
    assert result.median_reach_rate is None
    assert result.total_shoppers == 0
    store.close()
