"""The whole chain, from a frame stream to something a person would be sent.

Every other test here is a unit test, and almost every bug this codebase has had
lived in a seam rather than inside a component: shelf zones that no drawing tool
could create, an event store that only opened for one of its two callers, flags
that were set and did nothing. Each piece was individually correct.

So this wires the real trackers, the real store, the real analysis, the real
digest and the real payload builder together and pushes a known population of
shoppers through them. The detector is the only thing faked, because its output
is a `FrameResult` and that is the architecture seam everything downstream is
built on.

If the numbers at the end do not match the shoppers invented at the start,
something between them is lying.
"""

import json

import pytest

from patron.deliver import build_payload
from patron.digest import build_digest
from patron.events import ReachTracker, VisitTracker
from patron.floor import FloorMap, PositionRecorder
from patron.store import EventStore
from patron.types import Box, FrameResult, Pose, TrackedPerson
from patron.zones import Zone, ZoneSet

FPS = 10.0

# A shelf face on the left, walkable floor to its right. Foot points land in the
# aisle; a reaching wrist lands in the shelf.
ZONES = ZoneSet(
    zones=(
        Zone(name="shelf", polygon=((0, 0), (100, 0), (100, 400), (0, 400)), kind="shelf"),
        Zone(name="aisle", polygon=((100, 0), (500, 0), (500, 400), (100, 400)), kind="floor"),
    )
)

# Maps the aisle onto a 4m x 4m patch of floor, exactly, so a projected position
# can be checked against a distance rather than merely existing.
FLOOR = FloorMap(
    correspondences=(
        ((100.0, 400.0), (0.0, 0.0)),
        ((500.0, 400.0), (4.0, 0.0)),
        ((500.0, 0.0), (4.0, 4.0)),
        ((100.0, 0.0), (0.0, 4.0)),
    )
)


def standing_at(track_id: int, x: float, y: float = 300.0) -> TrackedPerson:
    return TrackedPerson(
        track_id=track_id,
        box=Box(x1=x - 20, y1=y - 160, x2=x + 20, y2=y),
        confidence=0.9,
    )


def reaching_pose(shoulder_x: float) -> Pose:
    """A wrist well inside the shelf, with the arm genuinely extended.

    Shoulder width 40px against a 160px body, so the denominator floor does not
    engage and the extension ratio is the real one.
    """
    return Pose(
        points={
            "left_wrist": (50.0, 200.0, 0.9),
            "left_shoulder": (shoulder_x, 200.0, 0.9),
            "right_shoulder": (shoulder_x + 40.0, 200.0, 0.9),
        }
    )


def walk_through(
    store: EventStore,
    session: int,
    shoppers: int,
    reachers: int,
    dwell_frames: int = 40,
):
    """Push a known population through the real trackers into the real store."""
    visits = VisitTracker(ZONES, fps=FPS, min_frames_inside=2, min_frames_outside=3)
    reaches = ReachTracker(ZONES, fps=FPS, min_frames_inside=2, min_frames_outside=3)
    positions = PositionRecorder(FLOOR, min_interval_s=1.0)

    frame = 0
    for track_id in range(1, shoppers + 1):
        is_reacher = track_id <= reachers
        for step in range(dwell_frames):
            person = standing_at(track_id, x=250.0)
            # Reach in the middle of the stay, so a visit and a reach overlap
            # the way they do in life.
            poses = (
                {track_id: reaching_pose(230.0)}
                if is_reacher and 10 <= step < 25
                else {}
            )
            result = FrameResult(
                frame_index=frame, timestamp_s=frame / FPS, people=(person,)
            )
            store.add_visits(session, visits.update(result))
            store.add_reaches(session, reaches.update(result, poses))
            store.add_positions(session, positions.update(result))
            frame += 1

        # Everyone leaves before the next arrives, so tracks never overlap and
        # the shopper count is unambiguous.
        for _ in range(6):
            empty = FrameResult(frame_index=frame, timestamp_s=frame / FPS, people=())
            store.add_visits(session, visits.update(empty))
            store.add_reaches(session, reaches.update(empty, {}))
            frame += 1

    store.add_visits(session, visits.flush())
    store.add_reaches(session, reaches.flush())


@pytest.fixture(scope="module")
def populated(tmp_path_factory):
    """Built once. Pushing 40 shoppers through the real trackers is the slowest
    thing in this suite, and every test using it only reads."""
    store = EventStore(tmp_path_factory.mktemp("e2e") / "e2e.db")
    session = store.start_session("synthetic", fps=FPS, width=500, height=400)
    walk_through(store, session, shoppers=40, reachers=10)
    yield store, session
    store.close()


def test_the_shoppers_invented_are_the_shoppers_counted(populated):
    store, session = populated
    rows = {r["zone"]: r for r in store.zone_summary(session)}

    assert rows["aisle"]["shoppers"] == 40
    # Each shopper visits once: the debounce must not have split a 4 second
    # stay into several, and the gaps must not have merged two people into one.
    assert rows["aisle"]["visits"] == 40


def test_only_the_reachers_reached(populated):
    store, session = populated
    reaches = {r["zone"]: r for r in store.reach_summary(session)}

    assert reaches["shelf"]["shoppers"] == 10


def test_the_funnel_arrives_intact_at_the_analysis_layer(populated):
    from patron.analysis import analyze

    store, session = populated
    [finding] = [f for f in analyze(store, session).findings if f.zone == "shelf"]

    assert finding.funnel.passed == 40
    assert finding.funnel.stopped == 40
    assert finding.funnel.reached == 10
    assert finding.funnel.reach_rate == pytest.approx(0.25)
    # Paired from where the reachers were standing, not from configuration.
    assert finding.funnel.floor_zone == "aisle"


def test_positions_land_on_the_floor_where_the_shoppers_stood(populated):
    store, session = populated
    paths = store.paths(session)

    assert len(paths) == 40
    # Everyone stood at image x=250. The aisle runs x=100 to x=500 and maps to
    # 0 to 4 metres, so that is (250-100)/400 * 4 = 1.5m from the shelf face.
    for path in paths.values():
        for _t, x, _y in path:
            assert x == pytest.approx(1.5, abs=0.01)


def test_a_healthy_store_produces_a_digest_worth_nobody_s_time(populated):
    store, session = populated
    digest = build_digest(store, session)

    # 25% reach against a one-zone median of 25% is not news.
    assert digest.worth_sending is False
    assert digest.render() == "Nothing to report."


def test_a_broken_shelf_travels_all_the_way_to_the_payload(tmp_path):
    """The end the whole product exists to reach."""
    with EventStore(tmp_path / "bad.db") as store:
        session = store.start_session("synthetic", fps=FPS, width=500, height=400)
        walk_through(store, session, shoppers=40, reachers=1)

        digest = build_digest(store, session)
        payload = build_payload(digest, fmt="json")

    assert digest.worth_sending is True
    assert payload["findings"][0]["zone"] == "shelf"
    assert payload["findings"][0]["passed"] == 40
    assert payload["findings"][0]["reached"] == 1
    assert json.dumps(payload)


def test_a_change_between_two_runs_is_measured_across_the_whole_chain(tmp_path):
    from patron.analysis import measure_change

    with EventStore(tmp_path / "change.db") as store:
        before = store.start_session("week1", fps=FPS, width=500, height=400)
        walk_through(store, before, shoppers=40, reachers=2)
        after = store.start_session("week2", fps=FPS, width=500, height=400)
        walk_through(store, after, shoppers=40, reachers=20)

        change = measure_change(store, "shelf", before, after)

    assert change.verdict == "improved"
    assert change.before.reached == 2
    assert change.after.reached == 20


def test_the_tool_layer_sees_the_same_numbers_as_the_report(populated):
    """Two paths out of one store must not disagree.

    The agent answers from the tool layer and a person reads the report. If they
    ever diverge, one of them is lying to somebody who is about to make a
    decision.
    """
    from patron.tools import call

    store, session = populated
    funnel = call("get_funnel", store, session_id=session, zone="shelf")["funnel"]
    rows = {r["zone"]: r for r in store.zone_summary(session)}

    assert funnel["passed"] == rows["aisle"]["shoppers"]
    assert funnel["reach_rate"]["value"] == pytest.approx(0.25)


def test_nothing_shopper_identifying_reaches_the_wire(populated):
    store, session = populated
    payload = build_payload(build_digest(store, session), fmt="json")

    text = json.dumps(payload)
    for forbidden in ("track_id", "frame", "image", "embedding"):
        assert forbidden not in text
