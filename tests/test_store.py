from patron.events import ZoneVisit
from patron.floor import FloorPosition
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


def position(track_id: int, t_s: float, x: float, y: float) -> FloorPosition:
    return FloorPosition(
        track_id=track_id, frame=int(t_s * 30), t_s=t_s, x=x, y=y
    )


def test_paths_come_back_grouped_and_in_time_order(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        session = store.start_session("test.mp4", fps=30.0, width=1920, height=1080)
        # Inserted out of order on purpose.
        store.add_positions(
            session,
            [
                position(1, 2.0, 3.0, 1.0),
                position(2, 0.0, 9.0, 9.0),
                position(1, 0.0, 1.0, 1.0),
                position(1, 1.0, 2.0, 1.0),
                position(2, 1.0, 9.5, 8.0),
            ],
        )
        paths = store.paths(session)

    assert set(paths) == {1, 2}
    assert [t for t, _x, _y in paths[1]] == [0.0, 1.0, 2.0]
    assert [x for _t, x, _y in paths[1]] == [1.0, 2.0, 3.0]


def test_a_single_sighting_is_not_a_path(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        session = store.start_session("test.mp4", fps=30.0, width=1920, height=1080)
        store.add_positions(
            session,
            [
                position(1, 0.0, 1.0, 1.0),
                position(1, 1.0, 2.0, 1.0),
                position(7, 0.0, 5.0, 5.0),  # seen once, a detection blip
            ],
        )
        paths = store.paths(session)
        # Dropped from paths, but still counted: the sighting happened.
        assert store.position_count(session) == 3

    # A shopper who teleported in and vanished is noise, not a walked path.
    assert set(paths) == {1}


def test_positions_are_scoped_to_their_session(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        first = store.start_session("a.mp4", fps=30.0, width=1920, height=1080)
        second = store.start_session("b.mp4", fps=30.0, width=1920, height=1080)
        store.add_positions(first, [position(1, 0.0, 1.0, 1.0), position(1, 1.0, 2.0, 2.0)])
        store.add_positions(second, [position(1, 0.0, 8.0, 8.0), position(1, 1.0, 9.0, 9.0)])

        # Track id 1 exists in both and means a different person in each. Nothing
        # may join them. See CLAUDE.md constraint 2.
        assert store.paths(first)[1][0][1] == 1.0
        assert store.paths(second)[1][0][1] == 8.0
        assert store.position_count(first) == 2
        assert store.position_count() == 4


def test_adding_no_positions_is_not_an_error(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        session = store.start_session("test.mp4", fps=30.0, width=1920, height=1080)
        assert store.add_positions(session, []) == 0


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
