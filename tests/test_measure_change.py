"""Did the change work?

The question a retailer asks after acting on a recommendation, and the one that
makes this renewable rather than a one-off study. It is also the easiest place
to produce confident nonsense: two rates always differ by something, and
reporting that difference as a result would make noise look like evidence in a
document someone plans against.

So the answer is a verdict, and "indistinguishable" is a real one.
"""

import pytest

from patron.analysis import MIN_SHOPPERS_FOR_CONFIDENCE, measure_change
from patron.events import ZoneVisit
from patron.store import EventStore
from patron.tools import call


def span(track_id, zone, a, b):
    return ZoneVisit(
        track_id=track_id, zone=zone, entered_frame=int(a * 30), entered_s=a,
        exited_frame=int(b * 30), exited_s=b,
    )


def session_with(store, shoppers, reachers, source="t.mp4"):
    """One session where `shoppers` passed and `reachers` of them reached."""
    session = store.start_session(source, fps=30.0, width=1920, height=1080)
    store.add_visits(session, [span(i, "aisle", 0.0, 5.0) for i in range(1, shoppers + 1)])
    store.add_reaches(session, [span(i, "shelf", 1.0, 1.5) for i in range(1, reachers + 1)])
    return session


def test_a_large_real_improvement_is_called(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        before = session_with(store, 200, 20)   # 10%
        after = session_with(store, 200, 60)    # 30%
        change = measure_change(store, "shelf", before, after)

    assert change.verdict == "improved"
    assert change.conclusive is True
    assert change.delta == pytest.approx(0.20)
    assert change.p_value < 0.05


def test_a_large_real_decline_is_called(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        before = session_with(store, 200, 60)
        after = session_with(store, 200, 20)
        change = measure_change(store, "shelf", before, after)

    assert change.verdict == "worsened"
    assert change.delta < 0


def test_a_small_difference_on_thin_traffic_is_not_called(tmp_path):
    """The failure this exists to prevent.

    30 shoppers each, 5 reaches versus 7. That is a 40% relative improvement if
    you are careless, and it is nothing.
    """
    with EventStore(tmp_path / "e.db") as store:
        before = session_with(store, 30, 5)
        after = session_with(store, 30, 7)
        change = measure_change(store, "shelf", before, after)

    assert change.verdict == "indistinguishable"
    assert change.conclusive is False
    assert "by chance" in change.reason
    # The delta is still reported, so nobody thinks the number was hidden.
    assert change.delta == pytest.approx(2 / 30)


def test_below_the_confidence_floor_there_is_nothing_to_compare(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        before = session_with(store, 10, 1)
        after = session_with(store, 200, 60)
        change = measure_change(store, "shelf", before, after)

    assert change.verdict == "not_enough_data"
    assert "before" in change.reason
    assert str(MIN_SHOPPERS_FOR_CONFIDENCE) in change.reason
    assert change.p_value is None


def test_too_few_reaches_invalidates_the_test_even_with_traffic(tmp_path):
    """Plenty of shoppers, almost no reaches.

    The normal approximation behind the test needs a few expected outcomes in
    every cell. Returning a p-value here would look like the others and not be
    comparable to them.
    """
    with EventStore(tmp_path / "e.db") as store:
        before = session_with(store, 400, 1)
        after = session_with(store, 400, 2)
        change = measure_change(store, "shelf", before, after)

    assert change.verdict == "not_enough_data"
    assert "more reaches" in change.reason


def test_identical_periods_are_indistinguishable_not_improved(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        before = session_with(store, 200, 40)
        after = session_with(store, 200, 40)
        change = measure_change(store, "shelf", before, after)

    assert change.verdict == "indistinguishable"
    assert change.delta == pytest.approx(0.0)


def test_a_zone_absent_from_one_session_is_not_a_measurement(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        before = session_with(store, 200, 40)
        after = store.start_session("later.mp4", fps=30.0, width=1920, height=1080)
        store.add_visits(after, [span(i, "aisle", 0.0, 5.0) for i in range(1, 201)])

        assert measure_change(store, "shelf", before, after) is None


# --------------------------------------------------------------------------
# Through the tool layer
# --------------------------------------------------------------------------

def test_the_tool_returns_a_verdict_and_the_two_funnels(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        before = session_with(store, 200, 20)
        after = session_with(store, 200, 60)
        out = call(
            "measure_change", store, zone="shelf",
            before_session=before, after_session=after,
        )

    assert out["measurable"] is True
    assert out["verdict"] == "improved"
    assert out["before"]["reach_rate"]["value"] == pytest.approx(0.10)
    assert out["after"]["reach_rate"]["value"] == pytest.approx(0.30)
    assert out["reach_rate_delta"] == pytest.approx(0.20)


def test_the_tool_withholds_the_delta_when_a_rate_was_withheld(tmp_path):
    """A delta beside two withheld rates would be a number with no parents."""
    with EventStore(tmp_path / "e.db") as store:
        before = session_with(store, 10, 1)
        after = session_with(store, 200, 60)
        out = call(
            "measure_change", store, zone="shelf",
            before_session=before, after_session=after,
        )

    assert out["verdict"] == "not_enough_data"
    assert "reach_rate_delta" not in out
    assert out["before"]["reach_rate"]["value"] is None
    assert out["before"]["reach_rate"]["withheld"]


def test_sessions_holding_a_zone_come_back_newest_first(tmp_path):
    """What lets `measure` default to the right pair without magic numbers."""
    with EventStore(tmp_path / "e.db") as store:
        first = session_with(store, 50, 5)
        # A session that saw traffic but never a reach at this shelf.
        middle = store.start_session("quiet.mp4", fps=30.0, width=1920, height=1080)
        store.add_visits(middle, [span(i, "aisle", 0.0, 5.0) for i in range(1, 51)])
        last = session_with(store, 50, 20)

        holding = store.sessions_with_reaches("shelf")

    assert holding == [last, first]
    assert middle not in holding


def test_sessions_lists_what_each_run_holds(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        session_with(store, 50, 5)
        rows = store.sessions()

    assert rows[0]["visits"] == 50
    assert rows[0]["reaches"] == 5
    assert rows[0]["positions"] == 0


def test_the_tool_reports_an_unmeasurable_zone_plainly(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        before = session_with(store, 200, 40)
        after = session_with(store, 200, 40)
        out = call(
            "measure_change", store, zone="nope",
            before_session=before, after_session=after,
        )

    assert out["measurable"] is False
    assert "no reach data" in out["reason"]
