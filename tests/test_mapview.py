"""The 3D store map: what the page is given to draw, and how it is served.

The map shows shoppers as markers, which makes it the store's own view and not
an export. These tests hold it to that: the scene carries places and no stored
ids, the file it writes fetches nothing, and text somebody typed into a store
map cannot break out of the page it is written into.
"""

import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from openv.cli import main
from openv.demo import CAMERAS, _FRAME, camera_calibration, store_map, write_camera_demo
from openv.floor import FloorPosition
from openv.fusion import fuse_sessions
from openv.mapview import (
    LiveMap,
    build_scene,
    camera_footprint,
    create_map_app,
    render_html,
)
from openv.store import EventStore
from openv.web.server import console_page

from test_export import keys_in

BASE = datetime(2026, 3, 2, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def cameras_db(tmp_path_factory):
    """The three-camera demo, fused once. Every test here only reads it."""
    path = tmp_path_factory.mktemp("map") / "cameras.db"
    written = write_camera_demo(path)
    with EventStore(path) as store:
        fused, _ = fuse_sessions(store, list(written["sessions"].values()))
    return path, fused


# --------------------------------------------------------------------------
# The scene
# --------------------------------------------------------------------------

def test_a_fused_scene_has_every_camera_and_every_fixture(cameras_db):
    path, fused = cameras_db
    with EventStore(path) as store:
        scene = build_scene(store, fused)

    assert [c["label"] for c in scene["cameras"]] == ["cam-a", "cam-b", "cam-c"]
    assert scene["outline"] is not None and scene["extent"] == [0.0, 0.0, 16.0, 11.0]
    # Each fixture carries the finding from the camera that measured its zone.
    by_zone = {f["zone"]: f for f in scene["fixtures"]}
    assert set(by_zone) == {"shelf-cereal", "shelf-oats", "shelf-snacks", "shelf-endcap"}
    assert all(f["measured"] for f in by_zone.values())
    assert by_zone["shelf-endcap"]["severity"] == "high"
    assert by_zone["shelf-endcap"]["rate"] == "0%"
    assert scene["shoppers"] >= 110  # an upper bound on the 110 there were
    assert "No appearance is used" in scene["note"]


def test_the_scene_carries_places_and_no_stored_id(cameras_db):
    path, fused = cameras_db
    with EventStore(path) as store:
        scene = build_scene(store, fused)

    assert "track_id" not in keys_in(scene)
    assert "track_id" not in json.dumps(scene)
    # Markers are numbered in order of appearance within this scene.
    assert [t["id"] for t in scene["tracks"]] == list(range(1, len(scene["tracks"]) + 1))
    track = scene["tracks"][0]
    assert len(track["t"]) == len(track["x"]) == len(track["y"]) == len(track["c"])
    assert track["t"] == sorted(track["t"])
    # Somewhere in the overlaps, a shopper two cameras were both watching.
    assert any(count >= 2 for t in scene["tracks"] for count in t["c"])


def test_a_cameras_footprint_is_the_floor_it_was_calibrated_over():
    for camera, (x0, y0, x1, y1) in CAMERAS.items():
        footprint = camera_footprint(camera_calibration(camera), *_FRAME)
        xs = [p[0] for p in footprint]
        ys = [p[1] for p in footprint]

        assert (min(xs), min(ys)) == pytest.approx((x0, y0), abs=0.01)
        assert (max(xs), max(ys)) == pytest.approx((x1, y1), abs=0.01)


def test_one_camera_alone_is_still_a_scene(cameras_db):
    path, _ = cameras_db
    with EventStore(path) as store:
        scene = build_scene(store, 1)

    assert len(scene["cameras"]) == 1
    assert scene["fusion"] == []
    assert all(count == 1 for t in scene["tracks"] for count in t["c"])


def test_a_session_with_nobody_on_the_floor_says_so(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        session = store.start_session("t.mp4", fps=30.0, width=1, height=1)

        with pytest.raises(ValueError, match="no floor positions"):
            build_scene(store, session)


# --------------------------------------------------------------------------
# The file
# --------------------------------------------------------------------------

def test_the_replay_is_one_file_that_fetches_nothing(cameras_db):
    path, fused = cameras_db
    with EventStore(path) as store:
        html = render_html(build_scene(store, fused))

    assert "class StoreMap3D" in html and "window.OPENV_SCENE=" in html
    assert "--accent:#6fd6ff" in html  # the theme is inlined, not linked
    for remote in ("http://", "https://", 'src="/', 'href="/'):
        assert remote not in html, f"the replay reaches for {remote}"
    assert "<!--" not in html.split("<body>")[0].replace("<!doctype html>", "")


def test_text_from_a_store_map_cannot_end_the_script_it_is_written_into(cameras_db):
    path, fused = cameras_db
    with EventStore(path) as store:
        scene = build_scene(store, fused)
    scene["fixtures"][0]["label"] = "</script><script>alert(1)</script>"

    html = render_html(scene)

    assert "</script><script>alert(1)" not in html
    assert "<\\/script>" in html


def test_the_command_writes_the_replay_and_picks_the_fused_session(cameras_db, tmp_path, capsys):
    path, fused = cameras_db
    out = tmp_path / "map.html"

    assert main(["map3d", "--db", str(path), "--out", str(out)]) == 0
    printed = capsys.readouterr().out

    assert f"session {fused} / fused from 3 cameras" in printed
    assert "not an export" in printed
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_the_command_on_a_store_with_no_positions_exits_cleanly(tmp_path, capsys):
    db = tmp_path / "e.db"
    with EventStore(db) as store:
        store.start_session("t.mp4", fps=30.0, width=1, height=1)

    assert main(["map3d", "--db", str(db), "--out", str(tmp_path / "m.html")]) == 1
    assert "no session has floor positions" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Live
# --------------------------------------------------------------------------

def live_store(tmp_path):
    store = EventStore(tmp_path / "live.db")
    site = store_map("store-014", "Riverside")
    sessions = [
        store.start_session(
            f"synthetic: {camera}", fps=10.0, width=_FRAME[0], height=_FRAME[1],
            calibration=camera_calibration(camera), pose=True, store_map=site,
            started_at=BASE,
        )
        for camera in ("cam-a", "cam-c")
    ]
    return store, sessions


def test_the_live_map_starts_from_now_and_fuses_what_arrives(tmp_path):
    store, (a, c) = live_store(tmp_path)
    live = LiveMap(tmp_path / "live.db")
    with store:
        # Already in the store before the map started: history, not live.
        store.add_positions(a, [FloorPosition(99, 0, 0.0, 3.0, 9.0)])
        live.step(store)
        assert live.live_payload()["positions"] == []

        # One shopper in the overlap of two cameras, each with its own id.
        for second in range(1, 9):
            x = 1.0 + 0.5 * second
            store.add_positions(a, [FloorPosition(12, second * 10, float(second), x, 1.0)])
            store.add_positions(c, [FloorPosition(40, second * 10, second + 0.3, x + 0.15, 1.02)])
        live.step(store)
        payload = live.live_payload()
        scene = live.scene_payload()

    assert payload["cameras"] == 2 and payload["shoppers"] == 1
    assert len(payload["positions"]) == 1
    marker = payload["positions"][0]
    assert marker["cameras"] == 2
    assert marker["x"] == pytest.approx(4.5, abs=0.7)
    # Numbered by the map. Neither camera's track id leaves the store.
    assert marker["id"] == 1 and marker["tag"] == "S-001"
    assert "track_id" not in keys_in(payload)
    assert [cam["label"] for cam in scene["cameras"]] == ["cam-a", "cam-c"]
    assert len(scene["fixtures"]) == 4


def test_the_live_map_is_served_with_the_theme_and_nothing_from_outside(tmp_path):
    store, _ = live_store(tmp_path)
    store.close()
    live = LiveMap(tmp_path / "live.db", poll_s=0.05)

    with TestClient(create_map_app(live)) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert '/static/theme.css' in page.text and '/static/map3d.js' in page.text
        assert "https://" not in page.text and "<!--THEME-->" not in page.text

        theme = client.get("/static/theme.css")
        assert theme.status_code == 200 and theme.headers["content-type"].startswith("text/css")
        assert "--accent" in theme.text
        assert client.get("/static/map3d.js").status_code == 200

        assert client.get("/api/scene").json()["site"]["id"] == "store-014"
        assert client.get("/api/live").json()["positions"] == []

        # Only the theme, the renderer and fonts are served from that folder.
        assert client.get("/static/index.html").status_code == 404
        assert client.get("/static/..%2Fserver.py").status_code == 404
        assert client.get("/static/absent.css").status_code == 404


def test_the_console_page_links_the_theme_ahead_of_its_own_rules():
    page = console_page()

    assert "<!--THEME-->" not in page
    assert page.index("/static/theme.css") < page.index("<style>")
    assert "/static/map3d.js" in page
    # The colours are tokens now. A second palette in the page would drift.
    assert ":root{" not in page
