"""What gets sent, and when nothing does.

The hard part of delivery is not sending, it is deciding. A digest listing every
zone buries the line that mattered; a daily "nothing to report" teaches people
the channel is noise and they mute it before anything matters.
"""

import json

import pytest

from patron.digest import build_digest
from patron.events import ZoneVisit
from patron.store import EventStore


def span(i, zone, a, b):
    return ZoneVisit(
        track_id=i, zone=zone, entered_frame=int(a * 30), entered_s=a,
        exited_frame=int(b * 30), exited_s=b,
    )


def session_with(store, shoppers, reachers, stop_seconds=5.0, shelf="endcap"):
    session = store.start_session("t.mp4", fps=30.0, width=1920, height=1080)
    store.add_visits(
        session, [span(i, "aisle", 0.0, stop_seconds) for i in range(1, shoppers + 1)]
    )
    store.add_reaches(session, [span(i, shelf, 1.0, 1.5) for i in range(1, reachers + 1)])
    return session


def test_a_healthy_store_says_nothing(tmp_path):
    """The restraint that keeps the channel worth reading."""
    with EventStore(tmp_path / "e.db") as store:
        # Plenty of traffic and a good reach rate: no finding rises to notify.
        session = session_with(store, 200, 100)
        digest = build_digest(store, session)

    assert digest.worth_sending is False
    assert digest.headline == "nothing to report"
    assert digest.render() == "Nothing to report."


def test_a_real_problem_is_reported(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        session = session_with(store, 200, 2)  # 1% reach, plenty of traffic
        digest = build_digest(store, session)

    assert digest.worth_sending is True
    assert digest.findings
    assert "worth a look" in digest.headline
    assert "endcap" in digest.render()


def test_thin_data_is_not_a_reason_to_interrupt_someone(tmp_path):
    """An insufficient_data finding is real, and it is not news."""
    with EventStore(tmp_path / "e.db") as store:
        session = session_with(store, 5, 1)
        digest = build_digest(store, session)

    assert digest.worth_sending is False


def test_a_confirmed_improvement_is_worth_sending_on_its_own(tmp_path):
    """The most useful message this system sends is 'the thing you changed worked'."""
    with EventStore(tmp_path / "e.db") as store:
        before = session_with(store, 200, 2)
        after = session_with(store, 200, 30)
        digest = build_digest(store, after, compare_with=before)

    assert digest.worth_sending is True
    assert any(c.verdict == "improved" for c in digest.changes)
    assert "improved" in digest.headline
    assert f"since session {before}" in digest.render()


def test_an_inconclusive_change_does_not_by_itself_justify_a_message(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        before = session_with(store, 200, 4)
        after = session_with(store, 200, 5)
        digest = build_digest(store, after, compare_with=before)

    # The finding still fires, but the change is noise and must not read as
    # movement in the headline.
    assert all(not c.conclusive for c in digest.changes)
    assert "improved" not in digest.headline


def test_a_fixed_zone_still_gets_its_success_announced(tmp_path):
    """The gap that only checking the present would leave.

    A zone that was a problem, was changed, and is no longer a problem has by
    definition dropped out of the current findings. Looking only at those would
    announce every failure and no success, which is the wrong half.
    """
    with EventStore(tmp_path / "e.db") as store:
        before = store.start_session("a.mp4", fps=30.0, width=1920, height=1080)
        store.add_visits(before, [span(i, "aisle", 0.0, 5.0) for i in range(1, 201)])
        store.add_reaches(before, [span(i, "endcap", 1.0, 1.5) for i in range(1, 3)])
        store.add_reaches(before, [span(i, "healthy", 1.0, 1.5) for i in range(1, 101)])

        # The endcap was fixed and is now as good as the healthy shelf.
        after = store.start_session("b.mp4", fps=30.0, width=1920, height=1080)
        store.add_visits(after, [span(i, "aisle", 0.0, 5.0) for i in range(1, 201)])
        store.add_reaches(after, [span(i, "endcap", 1.0, 1.5) for i in range(1, 101)])
        store.add_reaches(after, [span(i, "healthy", 1.0, 1.5) for i in range(1, 101)])

        digest = build_digest(store, after, compare_with=before)

    endcap = next(c for c in digest.changes if c.zone == "endcap")
    assert endcap.verdict == "improved"
    assert digest.worth_sending is True
    assert "improved" in digest.headline


def test_only_zones_already_worth_worrying_about_are_re_measured(tmp_path):
    """Re-measuring a shelf nobody is worried about produces a verdict nobody
    asked for, and a longer message for no reason."""
    with EventStore(tmp_path / "e.db") as store:
        before = store.start_session("a.mp4", fps=30.0, width=1920, height=1080)
        store.add_visits(before, [span(i, "aisle", 0.0, 5.0) for i in range(1, 201)])
        store.add_reaches(before, [span(i, "endcap", 1.0, 1.5) for i in range(1, 3)])
        store.add_reaches(before, [span(i, "healthy", 1.0, 1.5) for i in range(1, 101)])

        after = store.start_session("b.mp4", fps=30.0, width=1920, height=1080)
        store.add_visits(after, [span(i, "aisle", 0.0, 5.0) for i in range(1, 201)])
        store.add_reaches(after, [span(i, "endcap", 1.0, 1.5) for i in range(1, 4)])
        store.add_reaches(after, [span(i, "healthy", 1.0, 1.5) for i in range(1, 111)])

        digest = build_digest(store, after, compare_with=before)

    measured = {c.zone for c in digest.changes}
    assert "healthy" not in measured


def test_low_severity_does_not_reach_anybody_unprompted(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        session = session_with(store, 200, 100)
        digest = build_digest(store, session)

    assert all(f.severity in ("high", "medium") for f in digest.findings)


def test_the_rendering_stays_plain(tmp_path):
    """Every destination mangles markdown differently, so there is none."""
    with EventStore(tmp_path / "e.db") as store:
        session = session_with(store, 200, 2)
        rendered = build_digest(store, session).render()

    for markup in ("**", "##", "`", "|"):
        assert markup not in rendered


def test_it_serialises_for_a_connector(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        before = session_with(store, 200, 2)
        after = session_with(store, 200, 30)
        payload = build_digest(store, after, compare_with=before).as_dict()

    assert json.dumps(payload)
    assert payload["worth_sending"] is True
    assert payload["compared_with"] == before
    assert payload["changes"][0]["verdict"] == "improved"


def test_the_render_names_its_evidence(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        session = session_with(store, 200, 2)
        rendered = build_digest(store, session).render()

    # A number without its denominator invites the reader to invent one.
    assert "passed 200" in rendered
    assert "200 shoppers observed" in rendered
    assert "measured, not estimated" in rendered


def test_an_empty_store_is_not_an_error(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        digest = build_digest(store, None)

    assert digest.worth_sending is False
    assert digest.findings == []
