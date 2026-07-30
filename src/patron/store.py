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

    def latest_session_id(self) -> int | None:
        row = self._conn.execute("SELECT MAX(id) AS id FROM sessions").fetchone()
        return int(row["id"]) if row and row["id"] is not None else None

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> EventStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
