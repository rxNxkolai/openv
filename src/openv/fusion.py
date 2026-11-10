"""Several cameras, one floor: fusing per-camera tracks into store-wide paths.

Every camera with a floor calibration already reports shoppers in the same
coordinate space, the store's own plan. That is what `floor.py` was built for.
This is the step after it: deciding that the dot camera A calls track 7 and the
dot camera B calls track 3 are one shopper standing in the overlap, and that
the track which just left A's view is the one that appeared in C's a second
later.

**It does this with position and time and nothing else.** Two observations are
the same shopper when they are at the same place on the floor at the same
moment, allowing for how far a person can move between samples. No appearance
is involved: there is no crop, no colour, no descriptor and no model, and
`Observation` has no field that could carry one. Stitching a shopper across
cameras is a form of re-identification, and CLAUDE.md draws the line exactly
here. Spatial-temporal handoff keeps the posture. Appearance matching does not,
and it is the surface BIPA litigation targets.

That choice has a cost and the cost is stated, not hidden:

- **Two shoppers close together at a seam cannot always be told apart.** When a
  new camera track could belong to two fused tracks and neither is clearly
  nearer, nothing is guessed. A new id starts and the event is counted in
  `ambiguous`. That splits one shopper into two, which inflates the count. The
  alternative is to merge two shoppers into one and hand each the other's path,
  and an overcount that says so is the better error.
- **A gap in coverage ends a track.** A shopper who walks out of every camera's
  view for longer than `handoff_gap_s` comes back as somebody new. That is the
  session-scoped rule doing its job: a fused id dies when the shopper is gone.

Fusion only ever runs across cameras recording at the same time. Fusing two
sessions from different days would be linking Monday's shopper to Tuesday's,
and `fuse_sessions` refuses it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Walking pace is about 1.4 m/s. A track with no velocity estimate yet could
#: have moved this fast in any direction since it was last seen.
MAX_SPEED = 2.0

#: How different two velocities can be and still be one shopper carrying on,
#: in metres per second. Somebody walking out of a door and somebody walking
#: in through it differ by about twice walking pace.
MAX_VELOCITY_CHANGE = 1.0


@dataclass(frozen=True)
class Observation:
    """One camera's report of one track at one moment, on the floor.

    These five fields are the whole input to fusion, on purpose. Anything that
    described what a shopper looks like would have to be added here, and a test
    holds this to exactly these names.
    """

    camera: str
    track_id: int
    t_s: float
    x: float
    y: float


@dataclass(frozen=True)
class FusedPosition:
    fused_id: int
    t_s: float
    x: float
    y: float
    cameras: tuple[str, ...]


@dataclass
class FusionReport:
    camera_tracks: int = 0
    fused_tracks: int = 0
    #: A new camera track joined a fused track another camera was watching.
    merged_in_overlap: int = 0
    #: A new camera track picked up a fused track no camera currently had.
    handed_off: int = 0
    #: One camera dropped a track and started a new id on the same shopper.
    rejoined: int = 0
    #: Could have been one of several, and was not guessed.
    ambiguous: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "camera_tracks": self.camera_tracks,
            "fused_tracks": self.fused_tracks,
            "merged_in_overlap": self.merged_in_overlap,
            "handed_off": self.handed_off,
            "rejoined": self.rejoined,
            "ambiguous": self.ambiguous,
        }


@dataclass
class _Fused:
    fused_id: int
    #: Recent (t, x, y), oldest first. Enough to say where this track was at a
    #: moment in the last few seconds without predicting anything.
    history: list[tuple[float, float, float]] = field(default_factory=list)
    vx: float = 0.0
    vy: float = 0.0
    has_velocity: bool = False
    #: camera -> (track id, when that camera last reported it)
    members: dict[str, tuple[int, float]] = field(default_factory=dict)

    @property
    def t_s(self) -> float:
        return self.history[-1][0]

    @property
    def x(self) -> float:
        return self.history[-1][1]

    @property
    def y(self) -> float:
        return self.history[-1][2]

    def at(self, t_s: float) -> tuple[float, float, float]:
        """Where this track was at `t_s`, and how unsure that is, in metres.

        Between two of its own observations the answer is interpolated and the
        slack is zero. Past either end it is a projection, and the slack grows
        with the time projected across.
        """
        first, last = self.history[0], self.history[-1]
        if t_s <= first[0]:
            return first[1], first[2], MAX_SPEED * (first[0] - t_s)
        if t_s >= last[0]:
            dt = t_s - last[0]
            drift = 0.6 if self.has_velocity else MAX_SPEED
            return last[1] + self.vx * dt, last[2] + self.vy * dt, drift * dt
        for (t0, x0, y0), (t1, x1, y1) in zip(self.history, self.history[1:]):
            if t0 <= t_s <= t1:
                f = (t_s - t0) / (t1 - t0) if t1 > t0 else 0.0
                return x0 + f * (x1 - x0), y0 + f * (y1 - y0), 0.0
        return last[1], last[2], 0.0

    def watched_by(self, t_s: float, lost_s: float) -> tuple[str, ...]:
        return tuple(
            sorted(c for c, (_, seen) in self.members.items() if abs(t_s - seen) <= lost_s)
        )


class FloorFuser:
    """Streams observations from several cameras into fused floor tracks.

    Observations must arrive in time order across all cameras.

    **A new camera track is not placed the moment it appears.** It is held for
    `settle_s`, about one sampling interval, and then compared against where
    each fused track actually was at those moments. Cameras sample a shopper
    once a second and not in step with each other, so deciding on arrival
    means predicting up to a second ahead, and a shopper turning into an aisle
    is over a metre from where a straight line put them. Waiting turns that
    prediction into an interpolation between two things that were observed.
    The cost is that a shopper new to a camera shows up a second late.

    `merge_radius` is how far apart two cameras may place the same shopper at
    the same instant, which is a statement about calibration error and not
    about people. `lost_s` is how long a camera can go without reporting a
    track before it is treated as no longer watching it. `handoff_gap_s` is how
    long a shopper can be out of every camera's view and still be the same
    fused track on return.
    """

    def __init__(
        self,
        merge_radius: float = 0.8,
        lost_s: float = 2.5,
        handoff_gap_s: float = 4.0,
        ambiguity_margin: float = 0.35,
        smoothing: float = 0.85,
        settle_s: float = 1.2,
        history_s: float = 6.0,
        keep_links: bool = False,
    ) -> None:
        self.merge_radius = merge_radius
        self.lost_s = lost_s
        self.handoff_gap_s = handoff_gap_s
        self.ambiguity_margin = ambiguity_margin
        self.smoothing = smoothing
        self.settle_s = settle_s
        self.history_s = history_s
        self.report = FusionReport()
        self._assigned: dict[tuple[str, int], int] = {}
        self._pending: dict[tuple[str, int], list[Observation]] = {}
        self._live: dict[int, _Fused] = {}
        self._next_id = 1
        #: (camera, camera track id, fused id) for every camera track placed,
        #: when asked for. The audit trail behind a fused path, and what the
        #: tests score against ground truth.
        self.links: list[tuple[str, int, int]] | None = [] if keep_links else None

    def update(self, obs: Observation) -> list[FusedPosition]:
        """Take one observation in. Returns the fused positions it settled.

        Usually one. None while a new camera track is still being held, and
        several when that hold ends and its observations are placed together.
        """
        out = self._settle(obs.t_s - self.settle_s)
        self._retire(obs.t_s)

        key = (obs.camera, obs.track_id)
        fused = self._live.get(self._assigned.get(key, -1))
        if fused is not None:
            out.append(self._absorb(fused, obs))
        else:
            self._pending.setdefault(key, []).append(obs)
        return out

    def flush(self) -> list[FusedPosition]:
        """Place everything still held. Call when the stream ends."""
        return self._settle(math.inf)

    def _settle(self, up_to: float) -> list[FusedPosition]:
        out: list[FusedPosition] = []
        ready = sorted(
            (key for key, held in self._pending.items() if held[0].t_s <= up_to),
            key=lambda key: self._pending[key][0].t_s,
        )
        for key in ready:
            held = self._pending.pop(key)
            self.report.camera_tracks += 1
            fused = self._associate(held)
            self._assigned[key] = fused.fused_id
            if self.links is not None:
                self.links.append((key[0], key[1], fused.fused_id))
            out.extend(self._absorb(fused, obs) for obs in held)
        return out

    def _retire(self, t_s: float) -> None:
        horizon = self.handoff_gap_s + self.settle_s
        gone = {fid for fid, fused in self._live.items() if t_s - fused.t_s > horizon}
        if gone:
            for fused_id in gone:
                del self._live[fused_id]
            self._assigned = {k: v for k, v in self._assigned.items() if v not in gone}

    def _associate(self, held: list[Observation]) -> _Fused:
        """Find the fused track a new camera track belongs to, or start one."""
        first = held[0]
        velocity = _velocity(held)
        candidates: list[tuple[float, _Fused]] = []
        for fused in self._live.values():
            if first.t_s - fused.t_s > self.handoff_gap_s:
                continue
            if first.t_s > fused.t_s and not self._carries_on(fused, velocity):
                continue
            member = fused.members.get(first.camera)
            # One camera sees one shopper once. If this camera went on
            # reporting this fused track under another id after the new id
            # appeared, the two ids are two shoppers as far as it can see.
            if member is not None and member[1] >= first.t_s:
                continue
            # Every held observation is compared with where the track was at
            # that same moment. Two samples agreeing is much stronger than one.
            total = 0.0
            for obs in held:
                x, y, slack = fused.at(obs.t_s)
                total += max(0.0, math.hypot(obs.x - x, obs.y - y) - slack)
            distance = total / len(held)
            if distance <= self.merge_radius:
                candidates.append((distance, fused))

        if candidates:
            candidates.sort(key=lambda c: c[0])
            clear = (
                len(candidates) == 1
                or candidates[1][0] - candidates[0][0] >= self.ambiguity_margin
            )
            if clear:
                chosen = candidates[0][1]
                # An overlap is another camera still reporting the track once
                # this one had appeared. Anything else was a pickup across a
                # gap, which is the weaker claim and is counted apart.
                if chosen.t_s >= first.t_s:
                    self.report.merged_in_overlap += 1
                elif first.camera in chosen.members:
                    self.report.rejoined += 1
                else:
                    self.report.handed_off += 1
                return chosen
            # Two shoppers it could be, and no way to say which from where they
            # are standing. Guessing would swap their paths half the time.
            self.report.ambiguous += 1

        fused = _Fused(fused_id=self._next_id)
        self._next_id += 1
        self._live[fused.fused_id] = fused
        self.report.fused_tracks += 1
        return fused

    @staticmethod
    def _carries_on(fused: _Fused, velocity: tuple[float, float] | None) -> bool:
        """Could a track that appeared after this one stopped be it, continuing?

        Only asked when nothing was watching the fused track at that moment, so
        its position is a projection. Position alone is not enough there, and a
        doorway is why: one shopper walks out, another walks in through the
        same metre of floor two seconds later, and by place and time they are
        the same dot. They are going opposite ways, though, and a tracker that
        merely dropped somebody for an instant picks them up going the same
        way. So a pickup across a gap has to agree on direction, and when
        either side has no direction to compare, nothing is assumed.
        """
        if velocity is None or not fused.has_velocity:
            return False
        return math.hypot(velocity[0] - fused.vx, velocity[1] - fused.vy) <= MAX_VELOCITY_CHANGE

    def _absorb(self, fused: _Fused, obs: Observation) -> FusedPosition:
        if not fused.history:
            fused.history.append((obs.t_s, obs.x, obs.y))
        elif obs.t_s >= fused.t_s:
            dt = obs.t_s - fused.t_s
            px, py, _ = fused.at(obs.t_s)
            # Smoothed toward the observation, not replaced by it. Two cameras
            # never agree to the centimetre, and a track that jumps between
            # their two opinions would look like a shopper shaking. Only
            # lightly: the prediction is a straight line, so heavy smoothing
            # drags a track wide at every corner, and at 0.6 that lag alone
            # was splitting one shopper in five at the aisle ends.
            x = px + self.smoothing * (obs.x - px)
            y = py + self.smoothing * (obs.y - py)
            if dt > 0.2:
                vx, vy = (x - fused.x) / dt, (y - fused.y) / dt
                speed = math.hypot(vx, vy)
                if speed > MAX_SPEED:
                    vx, vy = vx * MAX_SPEED / speed, vy * MAX_SPEED / speed
                blend = 0.5 if fused.has_velocity else 1.0
                fused.vx += blend * (vx - fused.vx)
                fused.vy += blend * (vy - fused.vy)
                fused.has_velocity = True
                fused.history.append((obs.t_s, x, y))
            else:
                # Two cameras reporting within a moment of each other refine
                # the same point and do not start a new one.
                fused.history[-1] = (fused.t_s, x, y)
            while len(fused.history) > 2 and obs.t_s - fused.history[0][0] > self.history_s:
                fused.history.pop(0)
        # An observation older than the track's latest came out of a hold. It
        # confirms the membership and leaves the position where newer data put it.
        previous = fused.members.get(obs.camera)
        if previous is None or obs.t_s >= previous[1]:
            fused.members[obs.camera] = (obs.track_id, obs.t_s)
        x, y, _ = fused.at(obs.t_s)
        return FusedPosition(
            fused_id=fused.fused_id, t_s=obs.t_s, x=x, y=y,
            cameras=fused.watched_by(obs.t_s, self.lost_s),
        )

    def live(self, t_s: float) -> list[FusedPosition]:
        """Every fused track some camera is watching at `t_s`."""
        out = []
        for fused in self._live.values():
            cameras = fused.watched_by(t_s, self.lost_s)
            if cameras and fused.history:
                x, y, _ = fused.at(min(t_s, fused.t_s + self.lost_s))
                out.append(FusedPosition(fused.fused_id, t_s, x, y, cameras))
        return out


def _velocity(held: list[Observation]) -> tuple[float, float] | None:
    """Velocity over a held camera track, or None when it is one sample long."""
    first, last = held[0], held[-1]
    dt = last.t_s - first.t_s
    if dt < 0.2:
        return None
    return ((last.x - first.x) / dt, (last.y - first.y) / dt)


def fuse(
    observations: list[Observation],
    min_interval_s: float = 1.0,
    links: list[tuple[str, int, int]] | None = None,
    **settings: float,
) -> tuple[list[FusedPosition], FusionReport]:
    """Fuse a recorded set of observations, sampled at a bounded rate.

    The output is rate-limited per fused track for the same reason positions
    are: paths and speed are fine at 1Hz, and two cameras watching one shopper
    should not double the rows written about them.
    """
    fuser = FloorFuser(keep_links=links is not None, **settings)
    settled: list[FusedPosition] = []
    for obs in sorted(observations, key=lambda o: (o.t_s, o.camera, o.track_id)):
        settled.extend(fuser.update(obs))
    settled.extend(fuser.flush())
    if links is not None:
        links.extend(fuser.links or [])

    settled.sort(key=lambda p: (p.t_s, p.fused_id))
    last_emitted: dict[int, float] = {}
    out: list[FusedPosition] = []
    for position in settled:
        previous = last_emitted.get(position.fused_id)
        if previous is not None and position.t_s - previous < min_interval_s:
            continue
        last_emitted[position.fused_id] = position.t_s
        out.append(position)
    return out, fuser.report


class FusionRefused(ValueError):
    """The sessions cannot be fused, and the message says which rule stopped it."""


def session_offsets(
    sessions: list[dict], same_start: bool = False
) -> dict[int, float]:
    """Seconds to add to each session's own clock to put them on one clock.

    From `started_at` by default, which is right for cameras that were running
    live. Footage processed afterwards carries the time it was processed, not
    the time it was filmed, so `same_start` is the operator's statement that
    the recordings began together.
    """
    from datetime import datetime

    if same_start:
        return {int(s["id"]): 0.0 for s in sessions}
    starts = {int(s["id"]): datetime.fromisoformat(s["started_at"]) for s in sessions}
    first = min(starts.values())
    return {sid: (start - first).total_seconds() for sid, start in starts.items()}


def simultaneous_sessions(store) -> list[int]:
    """The newest camera session, and every camera session recorded alongside it.

    What `openv fuse` picks when no sessions are named. Only sessions with a
    floor calibration and positions count, fused sessions are never members,
    and "alongside" means their recorded spans overlap on the shared clock.
    """
    cameras = []
    for session in store.sessions():
        if store.fusion_members(session["id"]):
            continue
        if store.session_calibration(session["id"]) is None:
            continue
        span = store.position_span(session["id"])
        if span is not None:
            cameras.append((session, span))
    if not cameras:
        return []

    offsets = session_offsets([s for s, _ in cameras])
    walls = {
        int(s["id"]): (offsets[int(s["id"])] + span[0], offsets[int(s["id"])] + span[1])
        for s, span in cameras
    }
    newest = max(walls)
    start, end = walls[newest]
    return sorted(
        sid for sid, (a, b) in walls.items() if not (b < start or a > end)
    )


def fuse_sessions(
    store,
    session_ids: list[int],
    same_start: bool = False,
    **settings: float,
) -> tuple[int, FusionReport]:
    """Fuse camera sessions recorded at the same time into one fused session.

    The fused session holds the store-wide paths, and which camera track
    became which fused one. Visits and reaches stay with the camera that
    measured them, against the zones that camera was given, and `analyze`
    reads a fused session by gathering them from its members.

    Refuses sessions that do not overlap in time. Cameras running together
    describe one set of shoppers and joining them is geometry. Sessions from
    different times describe different visits, and joining those is the
    cross-session link the privacy posture rules out.
    """
    if len(set(session_ids)) < 2:
        raise FusionRefused("fusion needs at least two camera sessions")
    existing = store.fusion_of(session_ids)
    if existing is not None:
        raise FusionRefused(
            f"those sessions are already fused, as session {existing}"
        )

    sessions = []
    for session_id in sorted(set(session_ids)):
        session = store.session(session_id)
        if session is None:
            raise FusionRefused(f"no session {session_id}")
        if store.fusion_members(session_id):
            raise FusionRefused(
                f"session {session_id} is itself a fused session. Fuse the "
                "camera sessions it was built from"
            )
        sessions.append(session)

    units = set()
    for session in sessions:
        calibration = store.session_calibration(session["id"])
        if calibration is None:
            raise FusionRefused(
                f"session {session['id']} has no floor calibration, so its "
                "positions are not in store coordinates. Record it with --floor"
            )
        units.add(calibration.get("units", "m"))
    if len(units) > 1:
        raise FusionRefused(
            f"the sessions are calibrated in different units ({', '.join(sorted(units))}), "
            "so they do not share a floor"
        )

    offsets = session_offsets(sessions, same_start=same_start)
    rows = {int(s["id"]): store.position_rows(s["id"]) for s in sessions}
    empty = [sid for sid, r in rows.items() if not r]
    if empty:
        raise FusionRefused(f"session {empty[0]} recorded no floor positions")

    spans = {
        sid: (offsets[sid] + r[0][1], offsets[sid] + r[-1][1]) for sid, r in rows.items()
    }
    for sid, (start, end) in spans.items():
        alone = all(
            end < other_start or start > other_end
            for other, (other_start, other_end) in spans.items()
            if other != sid
        )
        if alone:
            raise FusionRefused(
                f"session {sid} does not overlap any of the others in time. "
                "Fusion joins cameras that were recording together. Joining "
                "sessions from different times would link one visit to another, "
                "which OpenV does not do. If the recordings did start together, "
                "say so with --same-start"
            )

    observations = [
        Observation(camera=str(sid), track_id=track, t_s=offsets[sid] + t, x=x, y=y)
        for sid, session_rows in rows.items()
        for track, t, x, y in session_rows
    ]
    origin = min(o.t_s for o in observations)
    links: list[tuple[str, int, int]] = []
    fused, report = fuse(observations, links=links, **settings)

    maps = [store.session_store_map(s["id"]) for s in sessions]
    maps = [m for m in maps if m is not None]
    shared_map = maps[0] if maps and all(m.site == maps[0].site for m in maps) else None

    from datetime import datetime, timedelta

    first_start = min(datetime.fromisoformat(s["started_at"]) for s in sessions)
    fused_session = store.start_session(
        source="fused: sessions " + ", ".join(str(s["id"]) for s in sessions),
        fps=1.0,
        width=0,
        height=0,
        store_map=shared_map,
        started_at=first_start + timedelta(seconds=origin if not same_start else 0.0),
    )
    from openv.floor import FloorPosition

    store.add_positions(
        fused_session,
        [
            FloorPosition(
                track_id=p.fused_id, frame=int(round(p.t_s - origin)),
                t_s=p.t_s - origin, x=p.x, y=p.y,
            )
            for p in fused
        ],
    )
    store.add_fusion_members(
        fused_session, [(sid, offsets[sid] - origin) for sid in sorted(offsets)]
    )
    store.add_fusion_links(
        fused_session, [(int(camera), track, fused_id) for camera, track, fused_id in links]
    )
    return fused_session, report
