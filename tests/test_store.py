from patron.events import ZoneVisit
from patron.store import EventStore


def visit(track_id: int, zone: str, entered_s: float, exited_s: float) -> ZoneVisit:
    return ZoneVisit(
        track_id=track_id,
        zone=zone,
        entered_frame=int(entered_s * 30),
        entered_s=entered_s,
        exited_frame=int(exited_s * 30),
        exited_s=exited_s,
    )


def test_summary_counts_shoppers_separately_from_visits(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        session = store.start_session("test.mp4", fps=30.0, width=1920, height=1080)
        store.add_visits(
            session,
            [
                # Shopper 1 comes back to the endcap twice: 2 visits, 1 shopper.
                visit(1, "endcap", 0.0, 5.0),
                visit(1, "endcap", 20.0, 24.0),
                visit(2, "endcap", 3.0, 3.5),  # passing through, under 2s
            ],
        )
        rows = {r["zone"]: r for r in store.zone_summary(session)}

    endcap = rows["endcap"]
    assert endcap["visits"] == 3
    assert endcap["shoppers"] == 2
    assert endcap["stopped"] == 2  # the two 4s+ visits, not the 0.5s pass-through
    assert endcap["max_dwell_s"] == 5.0


def test_sessions_are_isolated_but_can_be_aggregated(tmp_path):
    db = tmp_path / "e.db"
    with EventStore(db) as store:
        first = store.start_session("a.mp4", fps=30.0, width=640, height=480)
        store.add_visits(first, [visit(1, "endcap", 0.0, 4.0)])

        second = store.start_session("b.mp4", fps=30.0, width=640, height=480)
        store.add_visits(second, [visit(1, "endcap", 0.0, 4.0)])

        assert store.zone_summary(first)[0]["visits"] == 1
        assert store.zone_summary(second)[0]["visits"] == 1
        assert store.zone_summary(None)[0]["visits"] == 2
        assert store.latest_session_id() == second


def test_store_reopens_existing_database(tmp_path):
    db = tmp_path / "e.db"
    with EventStore(db) as store:
        session = store.start_session("a.mp4", fps=30.0, width=640, height=480)
        store.add_visits(session, [visit(1, "endcap", 0.0, 4.0)])

    with EventStore(db) as reopened:
        assert reopened.latest_session_id() == session
        assert reopened.zone_summary(session)[0]["visits"] == 1


def test_adding_no_visits_is_a_noop(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        session = store.start_session("a.mp4", fps=30.0, width=640, height=480)
        assert store.add_visits(session, []) == 0
        assert store.zone_summary(session) == []
