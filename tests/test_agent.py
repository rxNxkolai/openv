"""What the agent is handed, and what it is allowed to hand back.

The API call itself needs credentials and is not exercised here. Everything
around it is, because the prompt is where the guardrails live and the parser is
where malformed output has to fail loudly rather than silently.
"""

import json

import pytest

from openv.agent import (
    SCHEMA,
    SYSTEM,
    AdvisorUnavailable,
    build_prompt,
    parse_recommendations,
)
from openv.analysis import analyze
from openv.events import ZoneSpan
from openv.store import EventStore


def span(track_id, zone, start, end):
    return ZoneSpan(
        track_id=track_id,
        zone=zone,
        entered_frame=int(start * 30),
        entered_s=start,
        exited_frame=int(end * 30),
        exited_s=end,
    )


@pytest.fixture
def analysis(tmp_path):
    store = EventStore(tmp_path / "a.db")
    session = store.start_session("t.mp4", fps=30.0, width=1920, height=1080)
    visits, reaches = [], []
    for aisle, shelf, reach_count in (("aisle-1", "shelf-1", 40), ("aisle-2", "shelf-2", 2)):
        base = abs(hash(aisle)) % 1000
        for i in range(100):
            tid = base * 1000 + i
            visits.append(span(tid, aisle, i * 10, i * 10 + 5))
            if i < reach_count:
                reaches.append(span(tid, shelf, i * 10 + 1, i * 10 + 2))
    store.add_visits(session, visits)
    store.add_reaches(session, reaches)
    result = analyze(store, session)
    store.close()
    return result


def test_prompt_carries_the_computed_numbers_not_raw_rows(analysis):
    payload = json.loads(build_prompt(analysis).split("\n\n")[1])

    finding = payload["findings"][0]
    assert finding["shelf_zone"] == "shelf-2"
    assert finding["shoppers_who_passed"] == 100
    assert finding["shoppers_who_reached"] == 2
    assert finding["reach_rate"] == pytest.approx(0.02)
    assert finding["store_median_reach_rate"] is not None
    # The agent must never be handed rows to do arithmetic over.
    assert "visits" not in payload
    assert "track_id" not in json.dumps(payload)


def test_prompt_ships_the_measurement_caveats(analysis):
    payload = json.loads(build_prompt(analysis).split("\n\n")[1])
    caveats = " ".join(payload["measurement_caveats"]).lower()

    # The two things a recommendation could otherwise silently overclaim.
    assert "upper bound" in caveats
    assert "does not confirm a product was picked up" in caveats


def test_system_prompt_forbids_inventing_products_and_numbers():
    lowered = SYSTEM.lower()
    assert "never invent a sku" in lowered
    assert "do not compute new numbers" in lowered
    assert "zones, not products" in lowered


def test_schema_requires_every_field_and_forbids_extras():
    item = SCHEMA["properties"]["recommendations"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == set(item["properties"])
    assert item["properties"]["confidence"]["enum"] == ["high", "medium", "low"]


def test_parses_well_formed_output():
    text = json.dumps(
        {
            "recommendations": [
                {
                    "zone": "shelf-2",
                    "diagnosis": "Traffic is high and engagement is not.",
                    "action": "Move the range to the eye-level band.",
                    "rationale": "2% reach against a 40% store median.",
                    "expected_effect": "Reach rate should rise toward the median.",
                    "confidence": "high",
                    "drafted_change": "Relocate shelf-2 stock up one shelf.",
                }
            ]
        }
    )
    (rec,) = parse_recommendations(text)

    assert rec.zone == "shelf-2"
    assert rec.confidence == "high"
    assert rec.drafted_change.startswith("Relocate")


def test_malformed_json_fails_loudly_rather_than_returning_nothing():
    # Silently returning [] would look identical to "no problems found".
    with pytest.raises(AdvisorUnavailable, match="unparseable JSON"):
        parse_recommendations("here are some thoughts, not JSON")


def test_missing_field_is_an_error_not_a_blank_recommendation():
    text = json.dumps({"recommendations": [{"zone": "shelf-2", "action": "Move it."}]})
    with pytest.raises(KeyError):
        parse_recommendations(text)


def test_empty_recommendation_list_is_valid():
    assert parse_recommendations(json.dumps({"recommendations": []})) == []


def test_missing_credentials_surface_as_advisor_unavailable(analysis, monkeypatch):
    """The commonest first-run failure must not be a stack trace.

    With no credentials the SDK raises TypeError from header validation at
    request time, not AuthenticationError and not at construction, so catching
    only AuthenticationError lets it escape.
    """
    import anthropic

    from openv import agent

    class _NoCreds:
        def __init__(self, *a, **kw):
            self.beta = self
            self.messages = self

        def create(self, *a, **kw):
            raise TypeError("Could not resolve authentication method.")

    monkeypatch.setattr(anthropic, "Anthropic", _NoCreds)

    with pytest.raises(AdvisorUnavailable, match="no usable credentials"):
        agent.advise(analysis)


def test_no_findings_means_no_api_call(analysis, monkeypatch):
    import anthropic

    from openv import agent
    from openv.analysis import StoreAnalysis

    def _explode(*a, **kw):
        raise AssertionError("should not have contacted the API")

    monkeypatch.setattr(anthropic, "Anthropic", _explode)

    empty = StoreAnalysis(
        session_id=1, findings=(), median_reach_rate=None, total_shoppers=0
    )
    assert agent.advise(empty) == []
