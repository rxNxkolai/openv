"""The store map: the retailer's shelf data, and how it is held to the zones.

A map is edited by hand or exported from somebody else's planogram system, so
the ways it can be wrong are the ordinary ones: a typo in a zone name, a floor
zone where a shelf was meant, an item listed twice. Each of those turns into a
product that nobody ever reaches for, which is the most damaging thing this
layer can say about an item, so each has to fail loudly at load time instead.
"""

import json
from pathlib import Path

import pytest

from openv.cli import main
from openv.store import EventStore
from openv.storemap import StoreMap
from openv.zones import Zone, ZoneSet

ROOT = Path(__file__).resolve().parents[1]

ZONES = ZoneSet(
    zones=(
        Zone(name="shelf-a", polygon=((0, 0), (100, 0), (100, 200), (0, 200)), kind="shelf"),
        Zone(name="shelf-b", polygon=((0, 200), (100, 200), (100, 400), (0, 400)), kind="shelf"),
        Zone(name="aisle", polygon=((100, 0), (300, 0), (300, 400), (100, 400)), kind="floor"),
    )
)


def raw_map(**overrides) -> dict:
    raw = {
        "store": {
            "id": "store-014",
            "name": "Riverside",
            "units": "m",
            "outline": [[0, 0], [10, 0], [10, 8], [0, 8]],
        },
        "fixtures": [
            {
                "zone": "shelf-a",
                "label": "Bay A",
                "plan": [[1, 1], [2, 1], [2, 4], [1, 4]],
                "products": [
                    {"sku": "1", "product": "Oat cereal", "brand": "Northfield",
                     "category": "cereal", "facings": 4, "level": "eye"},
                    {"sku": "2", "product": "Bran flakes", "brand": "Northfield",
                     "category": "cereal", "facings": 2},
                ],
            },
            {
                "zone": "shelf-b",
                "products": [
                    {"sku": "3", "brand": "Oatwright", "category": "cereal"},
                ],
            },
        ],
    }
    raw.update(overrides)
    return raw


def write(tmp_path, raw, name="s.map.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def test_a_map_round_trips_through_its_file(tmp_path):
    first = StoreMap.load(write(tmp_path, raw_map()))
    first.save(tmp_path / "again.map.json")

    assert StoreMap.load(tmp_path / "again.map.json") == first
    assert first.site == "store-014"
    assert first.fixture("shelf-a").facings == 6
    assert first.fixture("shelf-a").brands == ("Northfield",)


def test_a_product_with_no_name_is_called_by_its_sku(tmp_path):
    store_map = StoreMap.load(write(tmp_path, raw_map()))

    assert store_map.fixture("shelf-b").placements[0].product == "3"
    assert store_map.fixture("shelf-b").placements[0].facings == 1


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda r: r.pop("store"), "needs a 'store'"),
        (lambda r: r["store"].pop("id"), "needs an 'id'"),
        (lambda r: r["fixtures"][0].pop("zone"), "needs a 'zone'"),
        (lambda r: r["fixtures"][0]["products"][0].pop("brand"), "missing brand"),
        (lambda r: r["fixtures"][0]["products"][0].update(facings=0), "at least 1"),
        (lambda r: r["fixtures"][0]["products"][0].update(facings="lots"), "whole number"),
        (lambda r: r["fixtures"][0]["products"][1].update(sku="1"), "more than once"),
        (lambda r: r["fixtures"][1].update(zone="shelf-a"), "more than one fixture"),
        (lambda r: r["fixtures"][0].update(plan=[[0, 0], [1, 1]]), "at least 3 points"),
    ],
)
def test_a_broken_map_names_the_file_and_the_fault(tmp_path, mutate, expected):
    raw = raw_map()
    mutate(raw)
    path = write(tmp_path, raw)

    with pytest.raises(ValueError) as error:
        StoreMap.load(path)

    assert expected in str(error.value)
    assert str(path) in str(error.value)


def test_invalid_json_names_the_file_and_the_line(tmp_path):
    path = tmp_path / "bad.map.json"
    path.write_text('{"store": {"id": "x"},\n "fixtures": [}', encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON") as error:
        StoreMap.load(path)

    assert "line 2" in str(error.value)


# --------------------------------------------------------------------------
# Held to the zones
# --------------------------------------------------------------------------

def test_a_map_that_fits_has_no_problems(tmp_path):
    store_map = StoreMap.load(write(tmp_path, raw_map()))

    assert store_map.problems(ZONES) == []
    assert store_map.unmapped(ZONES) == []


def test_a_fixture_on_a_zone_that_does_not_exist_is_a_problem(tmp_path):
    raw = raw_map()
    raw["fixtures"][0]["zone"] = "shelf-z"

    problems = StoreMap.from_dict(raw).problems(ZONES)

    assert len(problems) == 1
    assert "shelf-z" in problems[0]
    # The fix is in the message: the names that do exist.
    assert "shelf-a" in problems[0]


def test_a_fixture_on_a_floor_zone_is_a_problem():
    """No reach can land in a floor zone, so its products would read as dead."""
    raw = raw_map()
    raw["fixtures"][0]["zone"] = "aisle"

    problems = StoreMap.from_dict(raw).problems(ZONES)

    assert "floor" in problems[0]


def test_a_shelf_with_nothing_mapped_is_noted_not_refused():
    raw = raw_map()
    raw["fixtures"].pop()

    store_map = StoreMap.from_dict(raw)

    assert store_map.problems(ZONES) == []
    assert store_map.unmapped(ZONES) == ["shelf-b"]


def test_the_bundled_sample_map_fits_its_zones():
    """The quickstart command in the README depends on this pair."""
    store_map = StoreMap.load(ROOT / "data" / "grocery.map.json")
    zones = ZoneSet.load(ROOT / "data" / "grocery.zones.json")

    assert store_map.problems(zones) == []
    assert store_map.unmapped(zones) == []


# --------------------------------------------------------------------------
# Recorded with the session
# --------------------------------------------------------------------------

def test_a_session_remembers_the_map_it_was_measured_against(tmp_path):
    store_map = StoreMap.from_dict(raw_map())

    with EventStore(tmp_path / "e.db") as store:
        session = store.start_session(
            "t.mp4", fps=30.0, width=300, height=400, zones=ZONES, pose=True,
            store_map=store_map,
        )
        bare = store.start_session("u.mp4", fps=30.0, width=300, height=400)

        assert store.session_store_map(session) == store_map
        # Unrecorded is None, not an empty map: the two mean different things.
        assert store.session_store_map(bare) is None
        sites = {row["id"]: row["site"] for row in store.sessions()}

    assert sites == {session: "store-014", bare: None}


def test_two_sessions_keep_their_own_maps(tmp_path):
    """A planogram reset must not rewrite what last month was measured against."""
    before = StoreMap.from_dict(raw_map())
    raw = raw_map()
    raw["fixtures"][1]["products"][0]["brand"] = "Harrow"
    after = StoreMap.from_dict(raw)

    with EventStore(tmp_path / "e.db") as store:
        first = store.start_session("a", fps=30.0, width=1, height=1, store_map=before)
        second = store.start_session("b", fps=30.0, width=1, height=1, store_map=after)

        assert store.session_store_map(first).fixture("shelf-b").brands == ("Oatwright",)
        assert store.session_store_map(second).fixture("shelf-b").brands == ("Harrow",)


def test_attaching_a_map_later_will_not_overwrite_one_silently(tmp_path):
    store_map = StoreMap.from_dict(raw_map())

    with EventStore(tmp_path / "e.db") as store:
        session = store.start_session("a", fps=30.0, width=1, height=1)
        store.attach_store_map(session, store_map)
        assert store.session_store_map(session) == store_map

        with pytest.raises(ValueError, match="already has a store map"):
            store.attach_store_map(session, store_map)

        raw = raw_map()
        raw["store"]["id"] = "store-015"
        store.attach_store_map(session, StoreMap.from_dict(raw), replace=True)
        assert store.session_store_map(session).site == "store-015"

        with pytest.raises(ValueError, match="no session"):
            store.attach_store_map(999, store_map)


# --------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------

def zones_file(tmp_path) -> Path:
    path = tmp_path / "z.zones.json"
    ZONES.save(path)
    return path


def test_the_command_says_what_each_fixture_can_measure(tmp_path, capsys):
    raw = raw_map()
    raw["fixtures"][1]["products"].append(
        {"sku": "4", "brand": "Harrow", "category": "cereal"}
    )

    code = main(["storemap", str(write(tmp_path, raw)), "--zones", str(zones_file(tmp_path))])
    out = capsys.readouterr().out

    assert code == 0
    assert "measures Northfield" in out
    assert "shared by 2 brands" in out
    assert "never" in out and "split by facings" in out
    assert "fits the zones" in out


def test_the_command_refuses_a_map_that_does_not_fit(tmp_path, capsys):
    raw = raw_map()
    raw["fixtures"][0]["zone"] = "shelf-z"

    code = main(["storemap", str(write(tmp_path, raw)), "--zones", str(zones_file(tmp_path))])

    assert code == 1
    assert "shelf-z" in capsys.readouterr().err


def test_the_command_reports_a_missing_file_without_a_traceback(tmp_path, capsys):
    code = main(["storemap", str(tmp_path / "absent.map.json")])
    captured = capsys.readouterr()

    assert code == 1
    assert "could not read" in captured.err
    assert "Traceback" not in captured.out + captured.err


def test_attach_checks_against_the_zones_the_session_recorded(tmp_path, capsys):
    """Not against a file passed alongside. The session's own zones are the
    ones its reaches were measured in."""
    db = tmp_path / "e.db"
    with EventStore(db) as store:
        session = store.start_session(
            "a", fps=30.0, width=300, height=400, zones=ZONES, pose=True
        )

    code = main(["storemap", str(write(tmp_path, raw_map())), "--attach", "--db", str(db)])
    out = capsys.readouterr().out

    assert code == 0
    assert f"recorded against session {session}" in out
    with EventStore(db) as store:
        assert store.session_store_map(session).site == "store-014"

    # A second attach needs the flag.
    code = main(["storemap", str(write(tmp_path, raw_map())), "--attach", "--db", str(db)])
    assert code == 1
    assert "--replace" in capsys.readouterr().err
