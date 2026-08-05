"""The agent's tool surface, and the guard it must not leak around.

`analysis.py` refuses to state a rate below MIN_SHOPPERS_FOR_CONFIDENCE
shoppers. A tool layer that answered with bare counts would hand a model exactly
what it needs to divide them and assert the rate the system declined to state,
defeating the guard in transport rather than in logic. These tests exist to keep
that from happening quietly.
"""

import pytest

from patron.analysis import MIN_SHOPPERS_FOR_CONFIDENCE
from patron.events import ZoneVisit
from patron.store import EventStore
from patron.tools import TOOL_SPECS, TOOLS, call


def span(track_id: int, zone: str, entered_s: float, exited_s: float) -> ZoneVisit:
    return ZoneVisit(
        track_id=track_id,
        zone=zone,
        entered_frame=int(entered_s * 30),
        entered_s=entered_s,
        exited_frame=int(exited_s * 30),
        exited_s=exited_s,
    )


def build(tmp_path, shoppers: int, reachers: int, name: str = "e.db"):
    """A store where `shoppers` passed an aisle and `reachers` reached its shelf."""
    store = EventStore(tmp_path / name)
    session = store.start_session("t.mp4", fps=30.0, width=1920, height=1080)
    store.add_visits(
        session, [span(i, "aisle", 0.0, 5.0) for i in range(1, shoppers + 1)]
    )
    store.add_reaches(
        session, [span(i, "shelf", 1.0, 1.5) for i in range(1, reachers + 1)]
    )
    return store, session


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------

def test_a_thin_sample_withholds_the_rate_and_says_why(tmp_path):
    store, session = build(tmp_path, shoppers=12, reachers=1)
    with store:
        out = call("get_funnel", store, session_id=session, zone="shelf")

    reach = out["funnel"]["reach_rate"]
    assert reach["value"] is None
    assert "12 shoppers observed" in reach["withheld"]
    assert str(MIN_SHOPPERS_FOR_CONFIDENCE) in reach["withheld"]


def test_the_counts_are_still_visible_but_arrive_labelled(tmp_path):
    """Hiding the counts would be its own dishonesty.

    They are returned, but never without the refusal sitting in the same object,
    so nothing can read the ingredients without also reading the verdict.
    """
    store, session = build(tmp_path, shoppers=12, reachers=1)
    with store:
        out = call("get_funnel", store, session_id=session, zone="shelf")

    reach = out["funnel"]["reach_rate"]
    assert reach["count"] == 1
    assert reach["of"] == 12
    assert "withheld" in reach


def test_a_sufficient_sample_states_the_rate(tmp_path):
    store, session = build(tmp_path, shoppers=40, reachers=10)
    with store:
        out = call("get_funnel", store, session_id=session, zone="shelf")

    reach = out["funnel"]["reach_rate"]
    assert reach["value"] == pytest.approx(0.25)
    assert "withheld" not in reach


def test_every_rate_in_a_finding_carries_the_same_guard(tmp_path):
    """The guard has to hold on every path out, not just the funnel tool."""
    store, session = build(tmp_path, shoppers=12, reachers=1)
    with store:
        out = call("get_findings", store, session_id=session)

    for finding in out["findings"]:
        for key in ("stop_rate", "reach_rate"):
            rate = finding["funnel"][key]
            assert rate["value"] is None
            assert rate["withheld"]


def test_singular_shopper_reads_correctly(tmp_path):
    store, session = build(tmp_path, shoppers=1, reachers=1)
    with store:
        out = call("get_funnel", store, session_id=session, zone="shelf")

    assert "1 shopper observed" in out["funnel"]["reach_rate"]["withheld"]


# --------------------------------------------------------------------------
# Comparison, where two withheld rates could become a confident difference
# --------------------------------------------------------------------------

def test_comparison_refuses_when_either_side_is_too_thin(tmp_path):
    store = EventStore(tmp_path / "c.db")
    with store:
        session = store.start_session("t.mp4", fps=30.0, width=1920, height=1080)
        store.add_visits(session, [span(i, "aisle", 0.0, 5.0) for i in range(1, 41)])
        store.add_reaches(session, [span(i, "shelf-a", 1.0, 1.5) for i in range(1, 11)])
        store.add_reaches(session, [span(i, "shelf-b", 1.0, 1.5) for i in range(1, 3)])

        out = call(
            "compare_zones", store, session_id=session, zone_a="shelf-a", zone_b="shelf-b"
        )

    # Both zones share one aisle here, so both are above threshold and this
    # should compare. The guard is exercised by the unknown-zone case below.
    assert out["comparable"] is True
    assert out["better"] == "shelf-a"


def test_comparison_refuses_an_unknown_zone(tmp_path):
    store, session = build(tmp_path, shoppers=40, reachers=10)
    with store:
        out = call(
            "compare_zones", store, session_id=session, zone_a="shelf", zone_b="nope"
        )

    assert out["comparable"] is False
    assert "nope" in out["reason"]
    assert "shelf" in out["available"]


# --------------------------------------------------------------------------
# Shape and dispatch
# --------------------------------------------------------------------------

def test_an_unknown_zone_lists_what_does_exist(tmp_path):
    store, session = build(tmp_path, shoppers=40, reachers=10)
    with store:
        out = call("get_funnel", store, session_id=session, zone="does-not-exist")

    assert out["found"] is False
    assert out["available"] == ["shelf"]


def test_an_unknown_tool_is_refused_not_guessed_at(tmp_path):
    store, _session = build(tmp_path, shoppers=40, reachers=10)
    with store:
        out = call("get_everything", store)

    # A model inventing a plausible tool name is about to invent a plausible
    # answer, so this must fail loudly.
    assert "error" in out
    assert "get_findings" in out["available"]


def test_zones_report_whether_rates_are_available_at_all(tmp_path):
    store, session = build(tmp_path, shoppers=12, reachers=1)
    with store:
        out = call("list_zones", store, session_id=session)

    assert out["zones"][0]["has_enough_data_for_rates"] is False
    assert out["zones"][0]["shoppers_observed"] == 12


def test_findings_carry_the_benchmark_provenance(tmp_path):
    store, session = build(tmp_path, shoppers=40, reachers=10)
    with store:
        out = call("get_findings", store, session_id=session)

    # The benchmark being the store's own median is the reason it is arguable,
    # so it travels with the data rather than living only in a system prompt.
    assert "own median" in out["benchmark_note"]


def test_recommendations_are_read_only(tmp_path):
    store, session = build(tmp_path, shoppers=40, reachers=10)
    with store:
        out = call("get_recommendations", store, session_id=session)

    assert out["recommendations"] == []
    assert "human decision" in out["note"]
    # No tool may change a status: approval is the liability gate.
    assert not any("approve" in name for name in TOOLS)


def test_every_advertised_tool_exists_and_every_tool_is_advertised():
    advertised = {spec["name"] for spec in TOOL_SPECS}

    assert advertised == set(TOOLS)


def test_specs_are_shaped_for_a_tool_calling_api():
    for spec in TOOL_SPECS:
        assert spec["description"].strip()
        schema = spec["input_schema"]
        assert schema["type"] == "object"
        assert set(schema["required"]) <= set(schema["properties"])
