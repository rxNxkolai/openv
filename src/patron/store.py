"""Event store.

SQLite locally, and the schema is deliberately plain so it ports to Postgres
unchanged when the cloud aggregation layer lands. The schema is the asset here:
it is the behavioral data model the whole product sits on.

Note what is absent and must stay absent: no images, no face data, no cross-session
identifier. `track_id` is only unique within a session and is meaningless outside
it. See CLAUDE.md constraint 2.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from patron.events import ZoneVisit

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY,
    source      TEXT    NOT NULL,
    started_at  TEXT    NOT NULL,
    fps         REAL    NOT NULL,
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS visits (
    id             INTEGER PRIMARY KEY,
    session_id     INTEGER NOT NULL REFERENCES sessions(id),
    track_id       INTEGER NOT NULL,
    zone           TEXT    NOT NULL,
    entered_frame  INTEGER NOT NULL,
    entered_s      REAL    NOT NULL,
    exited_frame   INTEGER NOT NULL,
    exited_s       REAL    NOT NULL,
    dwell_s        REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_visits_session_zone ON visits(session_id, zone);

-- Same shape as visits. A reach is a wrist inside a shelf zone rather than a
-- foot point inside a floor zone, and it is the signal that separates a shopper
-- who stood in front of a shelf from one who engaged with it.
CREATE TABLE IF NOT EXISTS reaches (
    id             INTEGER PRIMARY KEY,
    session_id     INTEGER NOT NULL REFERENCES sessions(id),
    track_id       INTEGER NOT NULL,
    zone           TEXT    NOT NULL,
    entered_frame  INTEGER NOT NULL,
    entered_s      REAL    NOT NULL,
    exited_frame   INTEGER NOT NULL,
    exited_s       REAL    NOT NULL,
    dwell_s        REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reaches_session_zone ON reaches(session_id, zone);

-- Agent output. `status` starts at 'proposed' and there is deliberately no code
-- path that sets it to 'approved' automatically: a recommendation becomes an
-- action only when a human says so. That gate is the liability boundary for an
-- agent that changes what a store looks like. See CLAUDE.md.
CREATE TABLE IF NOT EXISTS recommendations (
    id              INTEGER PRIMARY KEY,
    session_id      INTEGER NOT NULL REFERENCES sessions(id),
    zone            TEXT    NOT NULL,
    diagnosis       TEXT    NOT NULL,
    action          TEXT    NOT NULL,
    rationale       TEXT    NOT NULL,
    expected_effect TEXT    NOT NULL,
    confidence      TEXT    NOT NULL,
    drafted_change  TEXT    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'proposed',
    created_at      TEXT    NOT NULL
);
"""


class EventStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def start_session(
        self, source: str, fps: float, width: int, height: int
    ) -> int:
        cursor = self._conn.execute(
            "INSERT INTO sessions (source, started_at, fps, width, height)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                source,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                float(fps),
                int(width),
                int(height),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def add_reaches(self, session_id: int, reaches: Iterable[ZoneVisit]) -> int:
        return self._add_spans("reaches", session_id, reaches)

    def add_visits(self, session_id: int, visits: Iterable[ZoneVisit]) -> int:
        return self._add_spans("visits", session_id, visits)

    def _add_spans(
        self, table: str, session_id: int, spans: Iterable[ZoneVisit]
    ) -> int:
        rows = [
            (
                session_id,
                v.track_id,
                v.zone,
                v.entered_frame,
                v.entered_s,
                v.exited_frame,
                v.exited_s,
                v.dwell_s,
            )
            for v in spans
        ]
        if not rows:
            return 0
        # `table` is never caller-supplied, only the two literals above.
        self._conn.executemany(
            f"INSERT INTO {table} (session_id, track_id, zone, entered_frame,"  # noqa: S608
            " entered_s, exited_frame, exited_s, dwell_s)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def reach_summary(self, session_id: int | None = None) -> list[sqlite3.Row]:
        """Per shelf-zone engagement."""
        where = "WHERE session_id = ?" if session_id is not None else ""
        params = (session_id,) if session_id is not None else ()
        return list(
            self._conn.execute(
                f"""
                SELECT zone,
                       COUNT(*)                 AS reaches,
                       COUNT(DISTINCT track_id) AS shoppers,
                       ROUND(AVG(dwell_s), 2)   AS mean_hold_s,
                       ROUND(MAX(dwell_s), 2)   AS max_hold_s
                FROM reaches
                {where}
                GROUP BY zone
                ORDER BY reaches DESC
                """,  # noqa: S608 - `where` is a fixed literal
                params,
            )
        )

    def zone_summary(self, session_id: int | None = None) -> list[sqlite3.Row]:
        """Per-zone funnel numbers.

        `stopped` uses a 2 second threshold: below that a shopper is passing
        through, above it they are actually looking. That threshold is the crude
        stand-in for the pass-by vs engage split until M2 adds reach detection.
        """
        where = "WHERE session_id = ?" if session_id is not None else ""
        params = (session_id,) if session_id is not None else ()
        return list(
            self._conn.execute(
                f"""
                SELECT zone,
                       COUNT(*)                                    AS visits,
                       COUNT(DISTINCT track_id)                    AS shoppers,
                       ROUND(AVG(dwell_s), 2)                      AS mean_dwell_s,
                       ROUND(MAX(dwell_s), 2)                      AS max_dwell_s,
                       SUM(CASE WHEN dwell_s >= 2.0 THEN 1 ELSE 0 END) AS stopped
                FROM visits
                {where}
                GROUP BY zone
                ORDER BY visits DESC
                """,
                params,
            )
        )

    def add_recommendations(self, session_id: int, recommendations) -> int:
        """Store agent output as proposals. Never as approved actions."""
        rows = [
            (
                session_id,
                r.zone,
                r.diagnosis,
                r.action,
                r.rationale,
                r.expected_effect,
                r.confidence,
                r.drafted_change,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            for r in recommendations
        ]
        if not rows:
            return 0
        self._conn.executemany(
            "INSERT INTO recommendations (session_id, zone, diagnosis, action,"
            " rationale, expected_effect, confidence, drafted_change, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def recommendations(
        self, session_id: int | None = None, status: str | None = None
    ) -> list[sqlite3.Row]:
        clauses, params = [], []
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return list(
            self._conn.execute(
                f"SELECT * FROM recommendations {where} ORDER BY id",  # noqa: S608
                tuple(params),
            )
        )

    def shelf_floor_pairs(self, session_id: int | None = None) -> dict[str, str]:
        """Which floor zone each shelf serves, inferred from where reachers stood.

        A shopper reaching into a shelf was standing somewhere at that moment. The
        floor zone they were most often standing in is the aisle that shelf faces.
        Deriving this from the data rather than from a config field means the
        pairing cannot silently drift out of sync with how the store is laid out.
        """
        where = "AND r.session_id = ?" if session_id is not None else ""
        params = (session_id,) if session_id is not None else ()
        rows = self._conn.execute(
            f"""
            SELECT r.zone AS shelf, v.zone AS floor, COUNT(*) AS overlaps
            FROM reaches r
            JOIN visits v
              ON  v.session_id = r.session_id
              AND v.track_id   = r.track_id
              -- the reach happened while the shopper was inside that floor zone
              AND r.entered_s <= v.exited_s
              AND r.exited_s  >= v.entered_s
            WHERE 1=1 {where}
            GROUP BY r.zone, v.zone
            ORDER BY overlaps DESC
            """,  # noqa: S608 - `where` is a fixed literal
            params,
        ).fetchall()

        pairs: dict[str, str] = {}
        for row in rows:  # ordered by overlaps desc, so first wins
            pairs.setdefault(row["shelf"], row["floor"])
        return pairs

    def shoppers_stopping(
        self, zone: str, threshold_s: float, session_id: int | None = None
    ) -> int:
        """Distinct shoppers who lingered in a zone rather than passing through."""
        where = "AND session_id = ?" if session_id is not None else ""
        params: tuple = (zone, threshold_s, *( (session_id,) if session_id is not None else () ))
        row = self._conn.execute(
            f"""
            SELECT COUNT(DISTINCT track_id) AS n
            FROM visits
            WHERE zone = ? AND dwell_s >= ? {where}
            """,  # noqa: S608 - `where` is a fixed literal
            params,
        ).fetchone()
        return int(row["n"]) if row else 0

    def total_shoppers(self, session_id: int | None = None) -> int:
        """Distinct track ids seen in any zone.

        NOTE: this overcounts when tracking fragments and reassigns a new id to a
        shopper who was occluded. Treat it as an upper bound.
        """
        where = "WHERE session_id = ?" if session_id is not None else ""
        params = (session_id,) if session_id is not None else ()
        row = self._conn.execute(
            f"SELECT COUNT(DISTINCT track_id) AS n FROM visits {where}",  # noqa: S608
            params,
        ).fetchone()
        return int(row["n"]) if row else 0

    def latest_session_id(self) -> int | None:
        row = self._conn.execute("SELECT MAX(id) AS id FROM sessions").fetchone()
        return int(row["id"]) if row and row["id"] is not None else None

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> EventStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
