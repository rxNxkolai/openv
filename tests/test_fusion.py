"""Several cameras, one floor.

Fusion is the one place OpenV joins two observations and says "same shopper",
which makes it the one place it could drift into re-identification. So these
tests are about two things: that it joins on position and time and has no way
to join on anything else, and that when position and time are not enough it
splits a shopper in two instead of blending two shoppers into one.
"""

import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from openv.analysis import analyze
from openv.cli import main
from openv.demo import CAMERA_START_S, simulate_cameras
from openv.events import ZoneVisit
from openv.floor import FloorPosition
from openv.fusion import (
    FusionRefused,
    Observation,
    fuse,
    fuse_sessions,
    simultaneous_sessions,
)
from openv.products import analyze_products
from openv.store import EventStore
from openv.storemap import StoreMap

SRC = Path(__file__).resolve().parents[1] / "src" / "openv"
BASE = datetime(2026, 3, 2, 10, 0, 0, tzinfo=timezone.utc)
CALIBRATION = {
    "units": "m",
    "correspondences": [
        {"image": [0, 100], "floor": [0, 0]}, {"image": [100, 100], "floor": [10, 0]},
        {"image": [100, 0], "floor": [10, 10]}, {"image": [0, 0], "floor": [0, 10]},
    ],
}


def walk(camera, track, t0, t1, x0, x1, y=1.0, step=1.0):
    """One camera's samples of somebody walking in a straight line."""
    out, t = [], t0
    while t <= t1 + 1e-9:
        f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        out.append(Observation(camera, track, t, x0 + f * (x1 - x0), y))
        t += step
    return out


# --------------------------------------------------------------------------
# What it is allowed to know
# --------------------------------------------------------------------------

def test_an_observation_is_a_place_and_a_time_and_nothing_else():
    """Anything describing what a shopper looks like would have to be a field
    here. Adding one is the moment this stops being geometry."""
    names = {f.name for f in dataclasses.fields(Observation)}

    assert names == {"camera", "track_id", "t_s", "x", "y"}


def test_fusion_touches_no_pixels_and_no_model():
    text = (SRC / "fusion.py").read_text(encoding="utf-8")
    imports = [line for line in text.splitlines() if line.lstrip().startswith(("import ", "from "))]

    for banned in ("cv2", "numpy", "torch", "mediapipe", "rfdetr", "supervision"):
        assert not any(banned in line for line in imports), f"fusion imports {banned}"


# --------------------------------------------------------------------------
# Joining
# --------------------------------------------------------------------------

def test_two_cameras_watching_one_shopper_make_one_track():
    a = walk("a", 7, 0.0, 6.0, 1.0, 7.0)
    b = walk("b", 3, 3.4, 9.4, 4.4, 10.4)  # joins while camera a still has them

    fused, report = fuse(a + b)

    assert report.camera_tracks == 2
    assert report.fused_tracks == 1
    assert report.merged_in_overlap == 1
    assert {p.fused_id for p in fused} == {1}
    # One continuous path across the seam, start to finish.
    assert fused[0].x == pytest.approx(1.0, abs=0.3)
    assert fused[-1].x == pytest.approx(10.4, abs=0.3)


def test_two_shoppers_in_different_places_stay_two():
    a = walk("a", 1, 0.0, 6.0, 1.0, 7.0, y=1.0)
    b = walk("b", 1, 0.0, 6.0, 1.0, 7.0, y=6.0)  # same track id, other camera, 5m away

    _, report = fuse(a + b)

    assert report.fused_tracks == 2
    assert report.merged_in_overlap == 0


def test_a_shopper_is_handed_across_a_short_gap_in_coverage():
    a = walk("a", 1, 0.0, 4.0, 0.0, 4.8)       # walking at 1.2 m/s, then unseen
    b = walk("b", 9, 6.0, 9.0, 7.2, 10.8)      # reappears where that pace puts them

    _, report = fuse(a + b)

    assert report.fused_tracks == 1
    assert report.handed_off == 1


def test_somebody_walking_in_is_not_the_person_who_just_walked_out():
    """The doorway. One shopper leaves and another arrives through the same
    metre of floor two seconds later. By place and time they are one dot. They
    are walking opposite ways, and that is the only thing that says otherwise."""
    leaving = walk("c", 1, 0.0, 4.0, 9.0, 14.0)
    arriving = walk("c", 2, 6.0, 10.0, 14.0, 9.0)

    _, report = fuse(leaving + arriving)

    assert report.fused_tracks == 2
    assert report.rejoined == 0 and report.handed_off == 0


def test_a_dropped_track_is_picked_up_when_it_carries_on_the_same_way():
    """A tracker loses someone for an instant and starts a new id. Same camera,
    same direction, same place: that is one shopper, and healing it is what
    stops every hiccup adding a shopper to the count."""
    before = walk("a", 1, 0.0, 4.0, 0.0, 4.8)
    after = walk("a", 2, 5.0, 9.0, 6.0, 10.8)

    _, report = fuse(before + after)

    assert report.fused_tracks == 1
    assert report.rejoined == 1


def test_a_long_absence_ends_the_track():
    """Out of every view for longer than the handoff gap is somebody new.
    That is the session-scoped rule: an id dies when the shopper is gone."""
    a = walk("a", 1, 0.0, 4.0, 0.0, 4.8)
    b = walk("b", 9, 30.0, 34.0, 4.8, 9.6)

    _, report = fuse(a + b)

    assert report.fused_tracks == 2


def test_two_shoppers_too_close_to_tell_apart_are_not_guessed():
    """Side by side into a second camera's view. Either of that camera's new
    tracks could be either shopper, so neither is merged, and it is counted."""
    first = walk("a", 1, 0.0, 8.0, 0.0, 9.6, y=1.00)
    second = walk("a", 2, 0.0, 8.0, 0.0, 9.6, y=1.25)
    # Camera b places each of them between the two, where calibration error
    # could honestly put either one.
    late_one = walk("b", 5, 4.3, 8.3, 5.16, 9.96, y=1.10)
    late_two = walk("b", 6, 4.3, 8.3, 5.16, 9.96, y=1.15)

    _, report = fuse(first + second + late_one + late_two)

    assert report.ambiguous >= 1
    # Never fewer tracks than there were shoppers.
    assert report.fused_tracks >= 2


# --------------------------------------------------------------------------
# Against ground truth
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [21, 3])
def test_fusion_never_blends_two_shoppers_on_the_synthetic_store(seed):
    """Three overlapping cameras, 110 invented shoppers, one doorway.

    The simulation knows which shopper every camera track really was. Fusion
    is never told. Scored afterwards: no fused track may hold two shoppers,
    most shoppers must come out as a single track, and the count may only err
    upward. This is a synthetic store and proves nothing about a real one.
    """
    scene = simulate_cameras(seed=seed)
    observations = [
        Observation(camera, p.track_id, p.t_s + CAMERA_START_S[camera], p.x, p.y)
        for camera, recorded in scene["cameras"].items()
        for p in recorded["positions"]
    ]
    links: list = []

    _, report = fuse(observations, links=links)

    in_fused: dict[int, set] = {}
    of_walker: dict[int, set] = {}
    for camera, track, fused_id in links:
        walker = scene["cameras"][camera]["truth"][track]
        in_fused.setdefault(fused_id, set()).add(walker)
        of_walker.setdefault(walker, set()).add(fused_id)

    blended = [f for f, walkers in in_fused.items() if len(walkers) > 1]
    whole = sum(1 for tracks in of_walker.values() if len(tracks) == 1)

    assert blended == []
    assert whole >= 0.8 * scene["truth"]
    assert scene["truth"] <= report.fused_tracks <= 1.3 * scene["truth"]
    # And far fewer than the cameras reported between them.
    assert report.fused_tracks < 0.5 * report.camera_tracks


def test_cameras_on_different_clocks_have_to_be_lined_up():
    """The three synthetic cameras start up to 0.9s apart. Ignoring that moves
    every shopper over a metre between cameras and the joins fall apart, which
    is why a session's start time is kept to the millisecond."""
    scene = simulate_cameras(seed=21)

    def run(aligned: bool):
        return fuse(
            [
                Observation(
                    camera, p.track_id,
                    p.t_s + (CAMERA_START_S[camera] if aligned else 0.0), p.x, p.y,
                )
                for camera, recorded in scene["cameras"].items()
                for p in recorded["positions"]
            ]
        )[1]

    assert run(aligned=False).fused_tracks > run(aligned=True).fused_tracks * 1.15


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------

def camera_session(store, label, observations, started=BASE, store_map=None):
    session = store.start_session(
        label, fps=10.0, width=100, height=100, calibration=CALIBRATION, pose=True,
        store_map=store_map, started_at=started,
    )
    store.add_positions(
        session,
        [FloorPosition(o.track_id, int(o.t_s * 10), o.t_s, o.x, o.y) for o in observations],
    )
    return session


def test_fusing_sessions_writes_a_session_that_reads_as_the_whole_store(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        a = camera_session(store, "cam-a", walk("a", 7, 0.0, 6.0, 1.0, 7.0))
        b = camera_session(store, "cam-b", walk("b", 3, 3.4, 9.4, 4.4, 10.4))

        fused, report = fuse_sessions(store, [a, b])

        assert report.fused_tracks == 1
        assert [m for m, _ in store.fusion_members(fused)] == [a, b]
        assert len(store.paths(fused)) == 1
        # The one figure no single camera has.
        assert analyze(store, fused).total_shoppers == 1
        assert store.session(fused)["source"] == f"fused: sessions {a}, {b}"
        assert simultaneous_sessions(store) == [a, b]


def test_sessions_from_different_times_are_refused(tmp_path):
    """The rule that keeps fusion from becoming re-identification. Cameras
    running together describe one set of shoppers. Monday and Tuesday do not,
    and joining them would link one visit to another."""
    with EventStore(tmp_path / "e.db") as store:
        monday = camera_session(store, "cam-a", walk("a", 1, 0.0, 6.0, 1.0, 7.0))
        tuesday = camera_session(
            store, "cam-a", walk("a", 1, 0.0, 6.0, 1.0, 7.0),
            started=BASE + timedelta(days=1),
        )

        with pytest.raises(FusionRefused, match="does not overlap") as error:
            fuse_sessions(store, [monday, tuesday])

        assert "link one visit to another" in str(error.value)
        assert store.latest_fused_session() is None


def test_same_start_is_the_operators_statement_for_processed_footage(tmp_path):
    """Footage tracked afterwards carries the time it was processed. Two files
    filmed together but processed an hour apart do not overlap on that clock."""
    with EventStore(tmp_path / "e.db") as store:
        a = camera_session(store, "a.mp4", walk("a", 7, 0.0, 6.0, 1.0, 7.0))
        b = camera_session(
            store, "b.mp4", walk("b", 3, 3.4, 9.4, 4.4, 10.4),
            started=BASE + timedelta(hours=1),
        )

        with pytest.raises(FusionRefused):
            fuse_sessions(store, [a, b])
        _, report = fuse_sessions(store, [a, b], same_start=True)

    assert report.fused_tracks == 1


def test_the_other_refusals_say_what_is_missing(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        a = camera_session(store, "cam-a", walk("a", 7, 0.0, 6.0, 1.0, 7.0))
        b = camera_session(store, "cam-b", walk("b", 3, 3.4, 9.4, 4.4, 10.4))
        bare = store.start_session("no-floor", fps=10.0, width=1, height=1, started_at=BASE)

        with pytest.raises(FusionRefused, match="at least two"):
            fuse_sessions(store, [a])
        with pytest.raises(FusionRefused, match="no floor calibration"):
            fuse_sessions(store, [a, bare])

        fused, _ = fuse_sessions(store, [a, b])
        with pytest.raises(FusionRefused, match="already fused"):
            fuse_sessions(store, [a, b])
        with pytest.raises(FusionRefused, match="itself a fused session"):
            fuse_sessions(store, [a, fused])


def test_a_brand_seen_by_two_cameras_counts_the_shopper_once(tmp_path):
    """One shopper walks past the brand's shelf under camera a and its other
    shelf under camera b, and reaches at both. Each camera counted one shopper.
    The store had one shopper, and after fusion the brand has one too."""
    def span(track, zone, a, b):
        return ZoneVisit(track_id=track, zone=zone, entered_frame=int(a * 10),
                         entered_s=a, exited_frame=int(b * 10), exited_s=b)

    item = {"sku": "1", "brand": "Northfield", "category": "cereal"}
    store_map = StoreMap.from_dict(
        {"store": {"id": "store-014"},
         "fixtures": [{"zone": "shelf-1", "products": [item]},
                      {"zone": "shelf-2", "products": [dict(item, sku="2")]}]}
    )
    with EventStore(tmp_path / "e.db") as store:
        a = camera_session(store, "cam-a", walk("a", 7, 0.0, 6.0, 1.0, 7.0), store_map=store_map)
        b = camera_session(store, "cam-b", walk("b", 3, 3.4, 9.4, 4.4, 10.4), store_map=store_map)
        store.add_visits(a, [span(7, "aisle-1", 0.0, 6.0)])
        store.add_reaches(a, [span(7, "shelf-1", 2.0, 2.5)])
        store.add_visits(b, [span(3, "aisle-2", 3.4, 9.4)])
        store.add_reaches(b, [span(3, "shelf-2", 6.0, 6.5)])

        fused, _ = fuse_sessions(store, [a, b])
        brand = analyze_products(store, fused).brand("Northfield")

    assert set(brand.measured_zones) == {"shelf-1", "shelf-2"}
    assert (brand.funnel.passed, brand.funnel.reached) == (1, 1)


def test_the_commands_fuse_the_demo_cameras_and_refuse_a_second_time(tmp_path, capsys):
    out = tmp_path / "demo"
    assert main(["demo-store", "--out", str(out)]) == 0
    db = str(out / "cameras.db")
    capsys.readouterr()

    assert main(["fuse", "--db", db]) == 0
    printed = capsys.readouterr().out
    assert "fused sessions 1, 2, 3 into session 4" in printed
    assert "No appearance is used" in printed

    # The fused session is the latest, so the default analysis is the store.
    assert main(["analyze", "--db", db]) == 0
    assert "shelf-endcap" in capsys.readouterr().out

    assert main(["fuse", "--db", db]) == 1
    assert "already fused" in capsys.readouterr().err
