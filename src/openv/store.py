"""Event store.

SQLite locally, and the schema is deliberately plain so it ports to Postgres
unchanged when the cloud aggregation layer lands. The schema is the asset here:
it is the behavioral data model the whole product sits on.

Note what is absent and must stay absent: no images, no face data, no cross-session
identifier. `track_id` is only unique within a session and is meaningless outside
it. See CLAUDE.md constraint 2.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from openv.events import ZoneVisit
from openv.storemap import Fixture, Placement, StoreMap
from openv.zones import Zone, ZoneSet

SCHEMA = """
-- `calibration` is the floor calibration this session's positions were measured
-- against, as JSON, or NULL when no floor map was used. It is the correspondences
-- and units rather than the solved matrix, for the reason floor.py gives: the
-- clicked points are reviewable and re-judgeable later, nine opaque floats are not.
--
-- Without this a stored position is a raw number in an unrecorded frame. You could
-- not say what origin it was measured from, whether the calibration was sound, or
-- whether two sessions are even in the same coordinate space. That matters most
-- for the thing the floor plane exists to enable: several cameras describing one
-- store. Fused cross-camera positions cannot be validated afterwards if none of the
-- homographies that produced them were written down.
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY,
    source      TEXT    NOT NULL,
    started_at  TEXT    NOT NULL,
    fps         REAL    NOT NULL,
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    calibration TEXT,
    -- 1 when pose ran, 0 when it did not, NULL for sessions recorded
    -- before this was written down. A shelf zone with no reaches means
    -- a dead fixture only if a reach could have been seen at all.
    pose        INTEGER
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

-- Where shoppers actually stood, in store floor coordinates rather than camera
-- pixels, so distance and speed mean something and several cameras can describe
-- one space. Sampled at a bounded rate rather than recorded per frame: 1Hz is
-- enough for paths and dwell, and per-frame rows would make this a movement
-- database rather than an event store.
--
-- Still session-scoped and still anonymous. A floor coordinate says where
-- somebody stood, not who they were.
CREATE TABLE IF NOT EXISTS positions (
    id          INTEGER PRIMARY KEY,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    track_id    INTEGER NOT NULL,
    frame       INTEGER NOT NULL,
    t_s         REAL    NOT NULL,
    x           REAL    NOT NULL,
    y           REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_positions_session_track
    ON positions(session_id, track_id, t_s);

-- Conversations with the agent, so a thread survives a restart and a question
-- asked last week can be reopened rather than retyped.
--
-- This is staff text, not shopper data: employees asking about their own store.
-- The privacy posture concerns what the cameras record, and nothing here holds
-- anything about a shopper that the tables above do not already.
--
-- `citations` is the JSON tool-call trail behind an answer. It is stored rather
-- than recomputed because a tool result is a claim about a moment: re-running
-- the same call next month answers a different question, and an answer whose
-- evidence has silently moved underneath it is worse than one with none.
-- The zone set a session was measured against. Without it a shelf zone
-- that recorded no reaches leaves no trace at all, so the deadest fixture
-- in the store is the one `analyze` cannot mention; `measure` cannot tell
-- that a polygon moved between two sessions and compares across the change
-- with a confident p-value; and a `.db` is unreadable without the zones
-- file that happened to sit next to it. The polygon itself is stored rather
-- than a hash of it, for the reason `calibration` stores clicked points: a
-- polygon can be drawn on a frame and judged, a hash cannot.
CREATE TABLE IF NOT EXISTS zones (
    id          INTEGER PRIMARY KEY,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    name        TEXT    NOT NULL,
    kind        TEXT    NOT NULL,
    polygon     TEXT    NOT NULL,
    UNIQUE (session_id, name)
);

-- The store map a session was measured against: which store this is, where
-- each fixture stands on the plan, and what the retailer says is on it. Same
-- reasoning as `zones` above. A planogram is reset every few weeks, so "what
-- was on that shelf" is a fact about a session, not about the store, and a
-- brand number read against today's map for last month's footage would be
-- attributed to whatever replaced it.
--
-- This is store data. Nothing in these three tables describes a shopper.
CREATE TABLE IF NOT EXISTS sites (
    id          INTEGER PRIMARY KEY,
    session_id  INTEGER NOT NULL UNIQUE REFERENCES sessions(id),
    site        TEXT    NOT NULL,
    name        TEXT,
    units       TEXT    NOT NULL,
    outline     TEXT
);

CREATE TABLE IF NOT EXISTS fixtures (
    id          INTEGER PRIMARY KEY,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    zone        TEXT    NOT NULL,
    label       TEXT,
    plan        TEXT,
    UNIQUE (session_id, zone)
);

CREATE TABLE IF NOT EXISTS placements (
    id          INTEGER PRIMARY KEY,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    zone        TEXT    NOT NULL,
    sku         TEXT    NOT NULL,
    product     TEXT    NOT NULL,
    brand       TEXT    NOT NULL,
    category    TEXT    NOT NULL,
    facings     INTEGER NOT NULL DEFAULT 1,
    level       TEXT,
    UNIQUE (session_id, zone, sku)
);

CREATE INDEX IF NOT EXISTS idx_placements_brand ON placements(brand, session_id);

-- A fused session is several cameras' floor positions joined into store-wide
-- paths, and this is which camera sessions it was built from and how their
-- clocks were lined up. Without it a fused path is a claim nobody can audit:
-- the cameras behind it, and the time offset that decided which dots were
-- "at the same moment", would be gone.
--
-- Members are always sessions recorded at the same time. Joining across
-- days would be re-identification, and the code that writes this refuses to.
CREATE TABLE IF NOT EXISTS fusions (
    id              INTEGER PRIMARY KEY,
    session_id      INTEGER NOT NULL REFERENCES sessions(id),
    member_session  INTEGER NOT NULL REFERENCES sessions(id),
    offset_s        REAL    NOT NULL,
    UNIQUE (session_id, member_session)
);

-- Which camera track became which fused track. The audit trail behind a
-- fused path, and what lets a shopper two cameras both saw be counted once.
-- Both ids are scoped to their sessions and neither says who anybody is.
CREATE TABLE IF NOT EXISTS fusion_links (
    id              INTEGER PRIMARY KEY,
    session_id      INTEGER NOT NULL REFERENCES sessions(id),
    member_session  INTEGER NOT NULL REFERENCES sessions(id),
    track_id        INTEGER NOT NULL,
    fused_id        INTEGER NOT NULL,
    UNIQUE (session_id, member_session, track_id)
);

CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY,
    session_id  INTEGER REFERENCES sessions(id),
    title       TEXT    NOT NULL,
    created_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id               INTEGER PRIMARY KEY,
    conversation_id  INTEGER NOT NULL REFERENCES conversations(id),
    role             TEXT    NOT NULL,
    text             TEXT    NOT NULL,
    citations        TEXT,
    truncated        INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, id);

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
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Bring an older database up to the current schema.

        `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists,
        so a column added later never reaches a store somebody has been recording
        into. Additive and idempotent on purpose: a `.db` from an earlier version
        stays readable, and its sessions simply report no calibration, which is
        the truth about them.
        """
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(sessions)")
        }
        if "calibration" not in columns:
            self._conn.execute("ALTER TABLE sessions ADD COLUMN calibration TEXT")
        if "pose" not in columns:
            self._conn.execute("ALTER TABLE sessions ADD COLUMN pose INTEGER")

    def start_session(
        self,
        source: str,
        fps: float,
        width: int,
        height: int,
        calibration: dict | None = None,
        zones: ZoneSet | None = None,
        pose: bool | None = None,
        store_map: StoreMap | None = None,
        started_at: datetime | None = None,
    ) -> int:
        """Open a session and record what it was measured against.

        `zones` and `pose` are the two facts a later reader needs to interpret
        an absence: a shelf with no reach rows is a dead fixture if the zone
        was drawn and pose ran, and nothing at all otherwise. `store_map` is
        what was on those shelves at the time, which is the only version of
        that fact worth joining a reach to.
        """
        cursor = self._conn.execute(
            "INSERT INTO sessions (source, started_at, fps, width, height, calibration, pose)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                source,
                # Milliseconds, because this is the clock several cameras are
                # lined up on. At walking pace a second of error between two
                # cameras is over a metre on the floor.
                (started_at or datetime.now(timezone.utc)).isoformat(
                    timespec="milliseconds"
                ),
                float(fps),
                int(width),
                int(height),
                json.dumps(calibration) if calibration is not None else None,
                None if pose is None else int(bool(pose)),
            ),
        )
        session_id = int(cursor.lastrowid)
        if zones is not None:
            self._conn.executemany(
                "INSERT INTO zones (session_id, name, kind, polygon) VALUES (?, ?, ?, ?)",
                [
                    (session_id, z.name, z.kind, json.dumps([list(p) for p in z.polygon]))
                    for z in zones
                ],
            )
        if store_map is not None:
            self._write_store_map(session_id, store_map)
        self._conn.commit()
        return session_id

    def _write_store_map(self, session_id: int, store_map: StoreMap) -> None:
        self._conn.execute(
            "INSERT INTO sites (session_id, site, name, units, outline)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                session_id,
                store_map.site,
                store_map.name,
                store_map.units,
                json.dumps([list(p) for p in store_map.outline])
                if store_map.outline is not None
                else None,
            ),
        )
        self._conn.executemany(
            "INSERT INTO fixtures (session_id, zone, label, plan) VALUES (?, ?, ?, ?)",
            [
                (
                    session_id,
                    f.zone,
                    f.label,
                    json.dumps([list(p) for p in f.plan]) if f.plan is not None else None,
                )
                for f in store_map.fixtures
            ],
        )
        self._conn.executemany(
            "INSERT INTO placements"
            " (session_id, zone, sku, product, brand, category, facings, level)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (session_id, p.zone, p.sku, p.product, p.brand, p.category, p.facings, p.level)
                for p in store_map.placements
            ],
        )

    def attach_store_map(
        self, session_id: int, store_map: StoreMap, replace: bool = False
    ) -> None:
        """Record a store map against a session that was captured without one.

        Legitimate because the map is store data that exists independently of
        the footage: what was on the shelf that day is a fact whether or not
        anyone typed it in beforehand. It is also the caller's claim that this
        map was the one in effect, which nothing here can check.

        Refuses to overwrite silently. A session whose products changed after
        its numbers were read is a session whose brand figures changed with
        them, and that should take a deliberate flag.
        """
        if self.session(session_id) is None:
            raise ValueError(f"no session {session_id}")
        if self.session_store_map(session_id) is not None:
            if not replace:
                raise ValueError(
                    f"session {session_id} already has a store map recorded"
                )
            for table in ("placements", "fixtures", "sites"):
                self._conn.execute(
                    f"DELETE FROM {table} WHERE session_id = ?",  # noqa: S608 - literals
                    (session_id,),
                )
        self._write_store_map(session_id, store_map)
        self._conn.commit()

    def session_store_map(self, session_id: int) -> StoreMap | None:
        """The store map a session was measured against, or None if unrecorded."""
        site = self._conn.execute(
            "SELECT site, name, units, outline FROM sites WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if site is None:
            return None

        def polygon(text: str | None):
            if text is None:
                return None
            return tuple((float(x), float(y)) for x, y in json.loads(text))

        placements: dict[str, list[Placement]] = {}
        for r in self._conn.execute(
            "SELECT zone, sku, product, brand, category, facings, level"
            " FROM placements WHERE session_id = ? ORDER BY id",
            (session_id,),
        ):
            placements.setdefault(r["zone"], []).append(
                Placement(
                    zone=r["zone"],
                    sku=r["sku"],
                    product=r["product"],
                    brand=r["brand"],
                    category=r["category"],
                    facings=int(r["facings"]),
                    level=r["level"],
                )
            )
        fixtures = tuple(
            Fixture(
                zone=r["zone"],
                label=r["label"],
                plan=polygon(r["plan"]),
                placements=tuple(placements.get(r["zone"], ())),
            )
            for r in self._conn.execute(
                "SELECT zone, label, plan FROM fixtures WHERE session_id = ? ORDER BY id",
                (session_id,),
            )
        )
        return StoreMap(
            site=site["site"],
            name=site["name"],
            units=site["units"],
            outline=polygon(site["outline"]),
            fixtures=fixtures,
        )

    def session(self, session_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def shoppers_in(
        self, zones: Iterable[str], session_id: int, min_dwell_s: float = 0.0
    ) -> int:
        """Distinct shoppers who visited any of these floor zones in one session.

        Distinct across the set, so a shopper who walked past two fixtures of
        the same brand is one shopper and not two. That is only meaningful
        inside a session, which is why the session is not optional here.
        """
        return self._distinct_tracks("visits", zones, session_id, min_dwell_s)

    def reachers_in(self, zones: Iterable[str], session_id: int) -> int:
        """Distinct shoppers who reached into any of these shelf zones."""
        return self._distinct_tracks("reaches", zones, session_id, 0.0)

    def _distinct_tracks(
        self, table: str, zones: Iterable[str], session_id: int, min_dwell_s: float
    ) -> int:
        names = sorted(set(zones))
        if not names:
            return 0
        marks = ", ".join("?" for _ in names)
        members = [m for m, _ in self.fusion_members(session_id)]
        if members:
            # A fused session: the rows live with the cameras that measured
            # them, and a shopper two cameras both saw is one fused track. A
            # camera track that was never placed on the floor has no fused id
            # and stands for itself, under a key no fused id can collide with.
            member_marks = ", ".join("?" for _ in members)
            row = self._conn.execute(
                f"SELECT COUNT(DISTINCT COALESCE(l.fused_id,"  # noqa: S608
                f" -(v.session_id * 10000000 + v.track_id))) AS n FROM {table} v"
                " LEFT JOIN fusion_links l ON l.session_id = ?"
                " AND l.member_session = v.session_id AND l.track_id = v.track_id"
                f" WHERE v.session_id IN ({member_marks}) AND v.dwell_s >= ?"
                f" AND v.zone IN ({marks})",
                (session_id, *members, min_dwell_s, *names),
            ).fetchone()
            return int(row["n"]) if row else 0
        # `table` is one of two literals, and `marks` is only question marks.
        row = self._conn.execute(
            f"SELECT COUNT(DISTINCT track_id) AS n FROM {table}"  # noqa: S608
            f" WHERE session_id = ? AND dwell_s >= ? AND zone IN ({marks})",
            (session_id, min_dwell_s, *names),
        ).fetchone()
        return int(row["n"]) if row else 0

    def session_zones(self, session_id: int) -> ZoneSet | None:
        """The zones a session was measured against, or None if unrecorded.

        None rather than an empty set, because a session that ran with no
        zones and a session from before zones were written down are different
        situations, and only one of them should be read as 'nothing drawn'.
        """
        rows = self._conn.execute(
            "SELECT name, kind, polygon FROM zones WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        if not rows:
            return None
        return ZoneSet(
            zones=tuple(
                Zone(
                    name=r["name"],
                    kind=r["kind"],
                    polygon=tuple((float(x), float(y)) for x, y in json.loads(r["polygon"])),
                )
                for r in rows
            )
        )

    def shelf_zones_measured(self, session_id: int | None = None) -> set[str]:
        """Shelf zones that were drawn in a session where pose ran.

        These are the zones where a reach could have been seen, so an absence
        of reaches means something. A shelf drawn on a session with pose off
        is not on this list, because zero reaches there is a setting, not a
        finding.
        """
        where = "AND z.session_id = ?" if session_id is not None else ""
        params = (session_id,) if session_id is not None else ()
        rows = self._conn.execute(
            f"""
            SELECT DISTINCT z.name
            FROM zones z JOIN sessions s ON s.id = z.session_id
            WHERE z.kind = 'shelf' AND s.pose = 1 {where}
            """,  # noqa: S608 - `where` is a fixed literal
            params,
        ).fetchall()
        return {r["name"] for r in rows}

    def zone_moved(self, zone: str, before_session: int, after_session: int) -> bool | None:
        """Did a zone's polygon change between two sessions?

        None when either session did not record its zones, which is the
        truth about older databases and must not be read as 'unchanged'.
        """
        polygons = []
        for session_id in (before_session, after_session):
            row = self._conn.execute(
                "SELECT polygon FROM zones WHERE session_id = ? AND name = ?",
                (session_id, zone),
            ).fetchone()
            if row is None:
                return None
            polygons.append(json.loads(row["polygon"]))
        return polygons[0] != polygons[1]

    def session_calibration(self, session_id: int) -> dict | None:
        """The floor calibration a session's positions were measured against.

        None when the session ran without a floor map, and also when it predates
        the column. Those are different situations and the caller usually cannot
        tell them apart, which is an argument for recording it from now on rather
        than for pretending the old rows had one.
        """
        row = self._conn.execute(
            "SELECT calibration FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None or row["calibration"] is None:
            return None
        return json.loads(row["calibration"])

    def add_reaches(self, session_id: int, reaches: Iterable[ZoneVisit]) -> int:
        return self._add_spans("reaches", session_id, reaches)

    def add_visits(self, session_id: int, visits: Iterable[ZoneVisit]) -> int:
        return self._add_spans("visits", session_id, visits)

    def add_positions(self, session_id: int, positions: Iterable) -> int:
        rows = [(session_id, p.track_id, p.frame, p.t_s, p.x, p.y) for p in positions]
        if not rows:
            return 0
        self._conn.executemany(
            "INSERT INTO positions (session_id, track_id, frame, t_s, x, y)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def position_rows(self, session_id: int) -> list[tuple[int, float, float, float]]:
        """One session's positions as (track_id, t_s, x, y), in time order."""
        return [
            (int(r["track_id"]), float(r["t_s"]), float(r["x"]), float(r["y"]))
            for r in self._conn.execute(
                "SELECT track_id, t_s, x, y FROM positions"
                " WHERE session_id = ? ORDER BY t_s, id",
                (session_id,),
            )
        ]

    def position_span(self, session_id: int) -> tuple[float, float] | None:
        """First and last position time in a session, or None when it has none."""
        row = self._conn.execute(
            "SELECT MIN(t_s) AS first, MAX(t_s) AS last FROM positions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None or row["first"] is None:
            return None
        return float(row["first"]), float(row["last"])

    def fusion_of(self, members: Iterable[int]) -> int | None:
        """The fused session already built from exactly these members, if any."""
        wanted = sorted(set(members))
        for row in self._conn.execute("SELECT DISTINCT session_id FROM fusions"):
            if [m for m, _ in self.fusion_members(int(row["session_id"]))] == wanted:
                return int(row["session_id"])
        return None

    def latest_fused_session(self) -> int | None:
        row = self._conn.execute("SELECT MAX(session_id) AS id FROM fusions").fetchone()
        return int(row["id"]) if row and row["id"] is not None else None

    def positions_since(self, after_id: int) -> list[sqlite3.Row]:
        """Positions written after a row id, across every session.

        What a live map tails: it remembers the last row it saw and asks for
        what has arrived since, from however many cameras are writing.
        """
        return list(
            self._conn.execute(
                "SELECT id, session_id, track_id, t_s, x, y FROM positions"
                " WHERE id > ? ORDER BY id",
                (after_id,),
            )
        )

    def max_position_id(self) -> int:
        row = self._conn.execute("SELECT MAX(id) AS id FROM positions").fetchone()
        return int(row["id"]) if row and row["id"] is not None else 0

    def add_fusion_members(
        self, session_id: int, members: Iterable[tuple[int, float]]
    ) -> None:
        self._conn.executemany(
            "INSERT INTO fusions (session_id, member_session, offset_s) VALUES (?, ?, ?)",
            [(session_id, member, offset) for member, offset in members],
        )
        self._conn.commit()

    def add_fusion_links(
        self, session_id: int, links: Iterable[tuple[int, int, int]]
    ) -> None:
        """Record (member session, camera track id, fused id) for a fused session."""
        self._conn.executemany(
            "INSERT OR IGNORE INTO fusion_links"
            " (session_id, member_session, track_id, fused_id) VALUES (?, ?, ?, ?)",
            [(session_id, member, track, fused) for member, track, fused in links],
        )
        self._conn.commit()

    def fused_shoppers(self, session_id: int) -> int:
        """Distinct fused tracks in a fused session: the store-wide shopper count."""
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT track_id) AS n FROM positions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["n"]) if row else 0

    def fusion_members(self, session_id: int) -> list[tuple[int, float]]:
        """The camera sessions a fused session was built from, with offsets.

        Empty for an ordinary camera session.
        """
        return [
            (int(r["member_session"]), float(r["offset_s"]))
            for r in self._conn.execute(
                "SELECT member_session, offset_s FROM fusions"
                " WHERE session_id = ? ORDER BY member_session",
                (session_id,),
            )
        ]

    def paths(
        self, session_id: int | None = None, min_points: int = 2
    ) -> dict[int, list[tuple[float, float, float]]]:
        """Walked paths by track id, each a list of (t_s, x, y) in floor units.

        `min_points` drops tracks seen once, which are a detection blip rather
        than a path and would otherwise show up as a shopper who teleported in
        and vanished.
        """
        where, params = ("WHERE session_id = ?", (session_id,)) if session_id else ("", ())
        rows = self._conn.execute(
            "SELECT track_id, t_s, x, y FROM positions"  # noqa: S608
            f" {where} ORDER BY track_id, t_s",
            params,
        ).fetchall()

        out: dict[int, list[tuple[float, float, float]]] = {}
        for row in rows:
            out.setdefault(row["track_id"], []).append(
                (row["t_s"], row["x"], row["y"])
            )
        return {k: v for k, v in out.items() if len(v) >= min_points}

    def position_count(self, session_id: int | None = None) -> int:
        where, params = ("WHERE session_id = ?", (session_id,)) if session_id else ("", ())
        return int(
            self._conn.execute(
                f"SELECT COUNT(*) FROM positions {where}", params  # noqa: S608
            ).fetchone()[0]
        )

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

    def sessions(self) -> list[dict]:
        """Every recorded session, newest first, with what each one holds."""
        rows = self._conn.execute(
            """
            SELECT s.*,
                   (SELECT COUNT(*) FROM visits    v WHERE v.session_id = s.id) AS visits,
                   (SELECT COUNT(*) FROM reaches   r WHERE r.session_id = s.id) AS reaches,
                   (SELECT COUNT(*) FROM positions p WHERE p.session_id = s.id) AS positions,
                   (SELECT COUNT(*) FROM zones     z WHERE z.session_id = s.id) AS zones,
                   (SELECT site FROM sites t WHERE t.session_id = s.id)            AS site
            FROM sessions s
            ORDER BY s.id DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def sessions_with_reaches(self, zone: str) -> list[int]:
        """Session ids holding reach data for a zone, newest first.

        What makes `measure` usable without anyone memorising session numbers.
        """
        rows = self._conn.execute(
            "SELECT DISTINCT session_id FROM reaches WHERE zone = ?"
            " ORDER BY session_id DESC",
            (zone,),
        ).fetchall()
        return [int(r["session_id"]) for r in rows]

    def sessions_measuring(self, zone: str) -> list[int]:
        """Sessions in which a shelf zone was measured, newest first.

        Measured means either a reach was recorded, or the zone was drawn and
        pose ran so that zero reaches is itself the measurement. This is what
        lets `measure` compare a dead fixture against its fixed self, which is
        the comparison the whole command exists for.
        """
        rows = self._conn.execute(
            """
            SELECT session_id FROM reaches WHERE zone = ?
            UNION
            SELECT z.session_id FROM zones z JOIN sessions s ON s.id = z.session_id
            WHERE z.name = ? AND z.kind = 'shelf' AND s.pose = 1
            ORDER BY session_id DESC
            """,
            (zone, zone),
        ).fetchall()
        return [int(r["session_id"]) for r in rows]

    def start_conversation(
        self, title: str, session_id: int | None = None
    ) -> int:
        cursor = self._conn.execute(
            "INSERT INTO conversations (session_id, title, created_at)"
            " VALUES (?, ?, ?)",
            (
                session_id,
                title,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def add_message(
        self,
        conversation_id: int,
        role: str,
        text: str,
        citations: list | None = None,
        truncated: bool = False,
    ) -> int:
        cursor = self._conn.execute(
            "INSERT INTO messages"
            " (conversation_id, role, text, citations, truncated, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                role,
                text,
                json.dumps(citations) if citations else None,
                1 if truncated else 0,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def conversation(self, conversation_id: int) -> list[dict]:
        """One thread's messages, oldest first, with citations decoded."""
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        out = []
        for row in rows:
            message = dict(row)
            message["citations"] = (
                json.loads(message["citations"]) if message["citations"] else []
            )
            message["truncated"] = bool(message["truncated"])
            out.append(message)
        return out

    def conversations(self, session_id: int | None = None) -> list[dict]:
        """Threads, newest first, with a message count for the list view."""
        where, params = (
            ("WHERE c.session_id = ?", (session_id,)) if session_id is not None else ("", ())
        )
        rows = self._conn.execute(
            "SELECT c.*, COUNT(m.id) AS message_count"  # noqa: S608
            " FROM conversations c LEFT JOIN messages m"
            " ON m.conversation_id = c.id"
            f" {where} GROUP BY c.id ORDER BY c.id DESC",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

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
