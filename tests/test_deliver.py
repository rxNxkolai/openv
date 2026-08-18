"""Getting a digest in front of someone.

No network here. The transport is injected, so what is under test is the
decision layer around it: that a quiet day is not a failure, that a failed send
is never swallowed, and that a dry run shows the real body rather than an
approximation of it.
"""

import json

import pytest

from openv.deliver import DEFAULT_TIMEOUT_S, Delivery, build_payload, send
from openv.digest import build_digest
from openv.events import ZoneVisit
from openv.store import EventStore


def span(i, z, a, b):
    return ZoneVisit(
        track_id=i, zone=z, entered_frame=int(a * 30), entered_s=a,
        exited_frame=int(b * 30), exited_s=b,
    )


def digest_with_a_problem(store):
    session = store.start_session("t.mp4", fps=30.0, width=1920, height=1080)
    store.add_visits(session, [span(i, "aisle", 0.0, 5.0) for i in range(1, 201)])
    store.add_reaches(session, [span(i, "endcap", 1.0, 1.5) for i in range(1, 3)])
    return build_digest(store, session)


def quiet_digest(store):
    session = store.start_session("t.mp4", fps=30.0, width=1920, height=1080)
    store.add_visits(session, [span(i, "aisle", 0.0, 5.0) for i in range(1, 201)])
    store.add_reaches(session, [span(i, "endcap", 1.0, 1.5) for i in range(1, 101)])
    return build_digest(store, session)


class Recorder:
    """Stands in for the network and remembers what it was handed."""

    def __init__(self, result=None):
        self.calls = []
        self.result = result or Delivery(ok=True, status=200, detail="sent")

    def __call__(self, url, body, timeout):
        self.calls.append({"url": url, "body": body, "timeout": timeout})
        return self.result


def test_a_quiet_day_posts_nothing_and_is_not_a_failure(tmp_path):
    """Scheduled jobs run on quiet days too."""
    with EventStore(tmp_path / "e.db") as store:
        digest = quiet_digest(store)
        transport = Recorder()
        result = send(digest, "https://example.invalid/hook", transport=transport)

    assert transport.calls == []
    assert result.ok is True
    assert "nothing worth sending" in result.detail


def test_a_real_finding_is_posted(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        digest = digest_with_a_problem(store)
        transport = Recorder()
        result = send(digest, "https://example.invalid/hook", transport=transport)

    assert result.ok is True
    assert len(transport.calls) == 1
    body = json.loads(transport.calls[0]["body"])
    assert "endcap" in body["text"]


def test_the_default_body_is_what_chat_webhooks_accept(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        payload = build_payload(digest_with_a_problem(store))

    # Slack, Discord, Teams and Google Chat all take {"text": ...}, which is
    # what avoids a per-vendor adapter.
    assert set(payload) == {"text"}
    assert isinstance(payload["text"], str)


def test_the_json_format_carries_the_structure(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        payload = build_payload(digest_with_a_problem(store), fmt="json")

    assert payload["worth_sending"] is True
    assert payload["findings"][0]["zone"] == "endcap"


def test_an_unknown_format_is_refused_before_anything_is_sent(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        with pytest.raises(ValueError, match="unknown format"):
            build_payload(digest_with_a_problem(store), fmt="xml")


def test_nothing_about_an_individual_leaves_the_machine(tmp_path):
    """What a legal review will actually ask."""
    with EventStore(tmp_path / "e.db") as store:
        payload = build_payload(digest_with_a_problem(store), fmt="json")

    text = json.dumps(payload)
    for forbidden in ("track_id", "image", "frame", "jpeg", "embedding"):
        assert forbidden not in text


def test_a_failed_send_is_reported_not_swallowed(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        digest = digest_with_a_problem(store)
        transport = Recorder(Delivery(ok=False, status=404, detail="no_such_channel"))
        result = send(digest, "https://example.invalid/hook", transport=transport)

    # Numbers that look attended to and are not being read are worse than
    # numbers nobody scheduled.
    assert result.ok is False
    assert result.status == 404
    assert "no_such_channel" in result.detail


def test_the_timeout_reaches_the_transport(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        transport = Recorder()
        send(digest_with_a_problem(store), "https://x.invalid", transport=transport)

    assert transport.calls[0]["timeout"] == DEFAULT_TIMEOUT_S


def test_the_url_is_passed_through_untouched(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        transport = Recorder()
        url = "https://hooks.example.invalid/services/A/B/C?x=1"
        send(digest_with_a_problem(store), url, transport=transport)

    assert transport.calls[0]["url"] == url


def test_a_delivery_serialises(tmp_path):
    assert json.dumps(Delivery(ok=True, status=200, detail="sent").as_dict())
