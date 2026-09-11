"""The store records what a session was measured against, not only what it measured.

Three defects shared one root cause and are pinned here together. A shelf zone
that recorded no reaches used to leave no trace, so the deadest fixture in the
store was the one `analyze` could not mention. `measure` matched sessions on a
zone's name and nothing else, so moving the polygon between two runs produced a
confident p-value across two different boundaries. And a `.db` could not be
read without the zones file that happened to sit beside it.
"""

import sqlite3

from openv.analysis import analyze, measure_change
from openv.events import ZoneSpan
from openv.store import EventStore
from openv.tools import list_zones
from openv.zones import Zone, ZoneSet

SHELF = Zone(name="endcap", kind="shelf", polygon=((0, 0), (100, 0), (100, 50), (0, 50)))
MOVED_SHELF = Zone(name="endcap", kind="shelf", polygon=((200, 0), (300, 0), (300, 50), (200, 50)))
AISLE = Zone(name="aisle", polygon=((0, 60), (100, 60), (100, 200), (0, 200)))
FAR_AISLE = Zone(name="far-aisle", polygon=((500, 500), (600, 500), (600, 600), (500, 600)))
ZONES = ZoneSet(zones=(SHELF, AISLE, FAR_AISLE))


def span(track_id: int, zone: str, start: float, end: float) -> ZoneSpan:
    return ZoneSpan(
        track_id=track_id,
        zone=zone,
        entered_frame=int(start * 30),
        entered_s=start,
        exited_frame=int(end * 30),
        exited_s=end,
    )


def shoppers(zone: str, count: int, offset: int = 0) -> list[ZoneSpan]:
    return [span(offset + i, zone, i * 10, i * 10 + 5) for i in range(count)]


def session(store, zones=ZONES, pose=True, visits=(), reaches=()) -> int:
    sid = store.start_session("t.mp4", fps=30.0, width=1920, height=1080, zones=zones, pose=pose)
    store.add_visits(sid, visits)
    store.add_reaches(sid, reaches)
    return sid


def test_zones_round_trip_through_the_store(tmp_path):
    with EventStore(tmp_path / "z.db") as store:
        sid = session(store)
        assert store.session_zones(sid) == ZONES


def test_a_session_without_zones_says_so_rather_than_claiming_none_were_drawn(tmp_path):
    with EventStore(tmp_path / "z.db") as store:
        sid = store.start_session("t.mp4", fps=30.0, width=1, height=1)
        assert store.session_zones(sid) is None
        assert store.shelf_zones_measured(sid) == set()
        assert store.zone_moved("endcap", sid, sid) is None


def test_a_database_from_before_zones_were_recorded_still_opens_and_analyses(tmp_path):
    """The pre-existing schema: no zones table, no pose column, rows already in it."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE sessions (
            id INTEGER PRIMARY KEY, source TEXT NOT NULL, started_at TEXT NOT NULL,
            fps REAL NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
            calibration TEXT
        );
        CREATE TABLE visits (
            id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, track_id INTEGER NOT NULL,
            zone TEXT NOT NULL, entered_frame INTEGER NOT NULL, entered_s REAL NOT NULL,
            exited_frame INTEGER NOT NULL, exited_s REAL NOT NULL, dwell_s REAL NOT NULL
        );
        CREATE TABLE reaches (
            id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, track_id INTEGER NOT NULL,
            zone TEXT NOT NULL, entered_frame INTEGER NOT NULL, entered_s REAL NOT NULL,
            exited_frame INTEGER NOT NULL, exited_s REAL NOT NULL, dwell_s REAL NOT NULL
        );
        INSERT INTO sessions VALUES (1, 'old.mp4', '2026-08-01T00:00:00+00:00', 30, 1920, 1080, NULL);
        INSERT INTO visits VALUES (1, 1, 7, 'aisle', 0, 0.0, 150, 5.0, 5.0);
        INSERT INTO reaches VALUES (1, 1, 7, 'endcap', 30, 1.0, 60, 2.0, 1.0);
        """
    )
    conn.commit()
    conn.close()

    with EventStore(path) as store:
        assert store.session_zones(1) is None
        result = analyze(store, 1)
        assert [f.funnel.shelf_zone for f in result.findings] == ["endcap"]
        assert result.findings[0].funnel.paired_by == "reachers"
        # Reopening runs the migration again and must be a no-op.
    with EventStore(path) as store:
        assert store.total_shoppers(1) == 1


def test_a_dead_fixture_is_reported_and_ranked_high(tmp_path):
    # 95 shoppers walk the aisle in front of a shelf nobody touches. Before the
    # zone set was recorded this printed "no shelf zones with reach data yet".
    with EventStore(tmp_path / "z.db") as store:
        sid = session(store, visits=shoppers("aisle", 95))
        result = analyze(store, sid)

    (finding,) = result.findings
    assert finding.kind == "dead"
    assert finding.severity == "high"
    assert finding.funnel.passed == 95
    assert finding.funnel.reached == 0
    assert finding.funnel.floor_zone == "aisle"
    assert finding.funnel.paired_by == "geometry"
    assert "95 shoppers walked past and not one reached" in finding.headline
    assert result.actionable == (finding,)


def test_the_agent_can_see_the_dead_fixture_too(tmp_path):
    # `list_zones` was the agent's only discovery mechanism, so a dead shelf
    # was invisible to it while `total_shoppers` still reported a busy store.
    with EventStore(tmp_path / "z.db") as store:
        sid = session(store, visits=shoppers("aisle", 95))
        out = list_zones(store, sid)

    assert [z["shelf_zone"] for z in out["zones"]] == ["endcap"]
    assert out["zones"][0]["shoppers_observed"] == 95
    assert out["total_shoppers"] == 95


def test_a_shelf_watched_without_pose_is_not_called_dead(tmp_path):
    # Zero reaches with pose off is a setting, not a finding.
    with EventStore(tmp_path / "z.db") as store:
        sid = session(store, pose=False, visits=shoppers("aisle", 95))
        result = analyze(store, sid)

    assert result.findings == ()


def test_evidence_from_reachers_outranks_the_drawing(tmp_path):
    # Geometrically the shelf sits beside "aisle", but every reacher stood in
    # "far-aisle". Where people actually stood is the better claim.
    with EventStore(tmp_path / "z.db") as store:
        sid = session(
            store,
            visits=shoppers("aisle", 40) + shoppers("far-aisle", 40, offset=100),
            reaches=[span(100 + i, "endcap", i * 10 + 1, i * 10 + 2) for i in range(6)],
        )
        (finding,) = analyze(store, sid).findings

    assert finding.funnel.floor_zone == "far-aisle"
    assert finding.funnel.paired_by == "reachers"


def test_measure_refuses_to_compare_across_a_moved_polygon(tmp_path):
    with EventStore(tmp_path / "z.db") as store:
        before = session(
            store,
            visits=shoppers("aisle", 60),
            reaches=[span(i, "endcap", i * 10 + 1, i * 10 + 2) for i in range(10)],
        )
        after = session(
            store,
            zones=ZoneSet(zones=(MOVED_SHELF, AISLE, FAR_AISLE)),
            visits=shoppers("aisle", 60),
            reaches=[span(i, "endcap", i * 10 + 1, i * 10 + 2) for i in range(20)],
        )
        assert store.zone_moved("endcap", before, after) is True
        change = measure_change(store, "endcap", before, after)

    assert change.verdict == "not_comparable"
    assert change.conclusive is False
    assert change.delta is None
    assert "changed between session" in change.reason


def test_measure_compares_a_dead_fixture_against_its_fix(tmp_path):
    # The comparison the command exists for: nothing before, something after.
    # Under the old store the "before" had no rows and was unmeasurable.
    with EventStore(tmp_path / "z.db") as store:
        before = session(store, visits=shoppers("aisle", 60))
        after = session(
            store,
            visits=shoppers("aisle", 60),
            reaches=[span(i, "endcap", i * 10 + 1, i * 10 + 2) for i in range(20)],
        )
        assert store.sessions_measuring("endcap") == [after, before]
        assert store.sessions_with_reaches("endcap") == [after]
        change = measure_change(store, "endcap", before, after)

    assert change.verdict == "improved"
    assert change.before.reached == 0
    assert change.after.reached == 20


def test_the_live_console_opens_a_fresh_session_when_zones_change(tmp_path):
    """The console resets its counts on a zone change. The store must follow.

    Otherwise one session id holds visits gathered against two boundaries, and
    `measure` would happily compare them.
    """
    from openv.web.engine import LiveEngine

    engine = LiveEngine(source="webcam:0", db_path=tmp_path / "live.db", pose=True)
    engine.set_zones([{"name": "endcap", "kind": "shelf", "polygon": [[0, 0], [1, 0], [1, 1]]}])
    with EventStore(tmp_path / "live.db") as store:
        first = engine._open_session(store, fps=30.0)
        assert engine._zones_changed is False
        engine.set_zones([{"name": "endcap", "kind": "shelf", "polygon": [[5, 5], [9, 5], [9, 9]]}])
        assert engine._zones_changed is True
        second = engine._open_session(store, fps=30.0)

        assert second != first
        assert store.zone_moved("endcap", first, second) is True
        assert store.shelf_zones_measured(second) == {"endcap"}
