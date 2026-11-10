"""The export: the one file that goes to somebody outside the store.

Everything else here serves the retailer whose cameras these are. This is what
a brand receives, so "anonymised" has to be something a test can check about
the bytes and not something a sentence says about the intent.
"""

import json

import pytest

from openv.analysis import MIN_SHOPPERS_FOR_CONFIDENCE
from openv.cli import main
from openv.export import SCHEMA_ID, ExportUnavailable, build_export
from openv.floor import FloorPosition
from openv.planview import render_plan
from openv.products import analyze_products
from openv.rollup import build_rollup, load_export
from openv.store import EventStore
from openv.storemap import StoreMap

from test_products import STORE_MAP, ZONES, build, item, span


def keys_in(value) -> set[str]:
    """Every key anywhere in a nested structure."""
    if isinstance(value, dict):
        found = set(value)
        for child in value.values():
            found |= keys_in(child)
        return found
    if isinstance(value, list):
        return set().union(*(keys_in(v) for v in value)) if value else set()
    return set()


# --------------------------------------------------------------------------
# What is in the file, and what cannot be
# --------------------------------------------------------------------------

def test_the_export_carries_nothing_about_any_individual(tmp_path):
    """Walk every key. A track id, a position or a path anywhere in the output
    is a row about a person, however it got there."""
    store, session = build(tmp_path)
    with store:
        store.add_positions(
            session,
            [FloorPosition(track_id=3, frame=i * 30, t_s=float(i), x=1.0, y=float(i))
             for i in range(5)],
        )
        export = build_export(store, session)

    keys = keys_in(export)
    for forbidden in ("track_id", "track", "tracks", "positions", "paths", "path",
                      "x", "y", "frame", "frames", "t_s", "entered_s", "exited_s"):
        assert forbidden not in keys, f"the export has a {forbidden!r} key"

    assert export["schema"] == SCHEMA_ID
    assert export["privacy"]["min_shoppers_for_a_rate"] == MIN_SHOPPERS_FOR_CONFIDENCE
    assert any("track identifiers" in line for line in export["privacy"]["excludes"])
    # The source path of the footage is local detail and stays local.
    assert "t.mp4" not in json.dumps(export)


def test_the_retailer_export_is_the_whole_store(tmp_path):
    store, session = build(tmp_path)
    with store:
        export = build_export(store, session)

    assert export["scope"] == {"audience": "retailer"}
    assert export["site"] == {"id": "store-014", "name": "Riverside"}
    assert {z["zone"] for z in export["zones"]} == {"shelf-a", "shelf-b", "shelf-c"}
    assert {b["brand"] for b in export["brands"]} == {"Northfield", "Oatwright", "Harrow"}

    northfield = next(b for b in export["brands"] if b["brand"] == "Northfield")
    assert northfield["measured"] is True
    assert (northfield["passed"], northfield["reached"]) == (60, 12)
    assert northfield["reach_rate"]["value"] == pytest.approx(0.2)
    assert northfield["store_median_reach_rate"] == export["store_median_reach_rate"]


def test_a_brand_that_shares_every_zone_has_no_number_in_the_file(tmp_path):
    store, session = build(tmp_path)
    with store:
        export = build_export(store, session)

    oatwright = next(b for b in export["brands"] if b["brand"] == "Oatwright")

    assert oatwright["measured"] is False
    for number in ("passed", "stopped", "reached", "reach_rate", "stop_rate"):
        assert number not in oatwright
    assert oatwright["shared_zones"] == [
        {"zone": "shelf-b", "facings": 1, "zone_facings": 4,
         "share_of_facings": 0.25, "other_brands": 1}
    ]
    assert "not a share of the reaches" in oatwright["not_measured"]


def test_an_item_claims_numbers_only_when_it_is_alone_in_its_zone(tmp_path):
    store, session = build(tmp_path)
    with store:
        export = build_export(store, session)

    by_sku = {p["sku"]: p for p in export["products"]}

    assert by_sku["5"]["attribution"] == "exclusive"
    assert by_sku["5"]["passed"] == 40 and by_sku["5"]["reached"] == 0
    # Shares shelf-a with another Northfield item: no numbers of its own.
    assert by_sku["1"]["attribution"] == "shared"
    assert "reached" not in by_sku["1"] and "reach_rate" not in by_sku["1"]


def test_a_thin_zone_exports_its_counts_and_a_refusal(tmp_path):
    thin = range(101, 101 + MIN_SHOPPERS_FOR_CONFIDENCE - 1)
    store, session = build(tmp_path, aisle_2=thin)
    with store:
        export = build_export(store, session)

    zone = next(z for z in export["zones"] if z["zone"] == "shelf-c")

    assert zone["passed"] == MIN_SHOPPERS_FOR_CONFIDENCE - 1
    assert zone["reach_rate"]["value"] is None
    assert "withheld" in zone["reach_rate"]


# --------------------------------------------------------------------------
# Narrowed to one brand
# --------------------------------------------------------------------------

def test_a_brand_export_names_no_other_brand(tmp_path):
    store, session = build(tmp_path)
    with store:
        export = build_export(store, session, brand="northfield")

    text = json.dumps(export)

    assert export["scope"]["audience"] == "brand"
    assert export["scope"]["brand"] == "Northfield"
    assert "Oatwright" not in text and "Harrow" not in text
    assert [b["brand"] for b in export["brands"]] == ["Northfield"]
    # Only the zones it is in, and the co-tenant is a count.
    assert {z["zone"] for z in export["zones"]} == {"shelf-a", "shelf-b"}
    shared = next(z for z in export["zones"] if z["zone"] == "shelf-b")
    assert shared["other_brands_in_zone"] == 1
    assert {p["brand"] for p in shared["products"]} == {"Northfield"}
    assert {p["sku"] for p in export["products"]} == {"1", "2", "3"}
    # Its category is the benchmark; a category it is not in is not its business.
    assert [c["category"] for c in export["categories"]] == ["cereal"]


def test_an_unknown_brand_is_refused_with_the_brands_that_exist(tmp_path):
    store, session = build(tmp_path)
    with store, pytest.raises(ExportUnavailable) as error:
        build_export(store, session, brand="Acme")

    assert "Northfield" in str(error.value)


def test_no_store_map_is_refused_and_says_how_to_get_one(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        session = store.start_session("t.mp4", fps=30.0, width=1, height=1)

        with pytest.raises(ExportUnavailable, match="no store map") as error:
            build_export(store, session)

    assert "--map" in str(error.value)


# --------------------------------------------------------------------------
# Across stores
# --------------------------------------------------------------------------

def two_stores(tmp_path) -> list[dict]:
    first_store, first = build(tmp_path, name="a.db")
    second_map = StoreMap.from_dict(
        {
            "store": {"id": "store-022", "name": "Harbour"},
            "fixtures": [
                # Northfield alone here too. Oatwright still shares.
                {"zone": "shelf-a", "products": [item("1", "Northfield")]},
                {"zone": "shelf-b", "products": [item("4", "Oatwright"),
                                                 item("9", "Harrow")]},
            ],
        }
    )
    second_store, second = build(tmp_path, store_map=second_map, name="b.db")
    with first_store, second_store:
        return [build_export(first_store, first), build_export(second_store, second)]


def test_a_rollup_sums_the_stores_and_keeps_each_one_visible(tmp_path):
    rollup = build_rollup(two_stores(tmp_path))

    northfield = next(r for r in rollup["rows"] if r["brand"] == "Northfield")

    assert northfield["stores_measured"] == 2
    assert northfield["pooled"]["passed"] == 120
    assert northfield["pooled"]["reached"] == 24
    assert northfield["pooled"]["reach_rate"]["value"] == pytest.approx(0.2)
    assert [s["site"] for s in northfield["stores"]] == ["store-014", "store-022"]
    assert all(s["reached"] == 12 for s in northfield["stores"])


def test_a_store_with_no_number_adds_nothing_to_the_pool(tmp_path):
    rollup = build_rollup(two_stores(tmp_path))

    oatwright = next(r for r in rollup["rows"] if r["brand"] == "Oatwright")
    harrow = next(r for r in rollup["rows"] if r["brand"] == "Harrow")

    # Shared at both stores: listed at both, measured at neither, no pool.
    assert oatwright["stores_listed"] == 2 and oatwright["stores_measured"] == 0
    assert oatwright["pooled"] is None
    # Measured at one store, shared at the other. The pool is the one store.
    assert harrow["stores_measured"] == 1 and harrow["stores_listed"] == 2
    assert harrow["pooled"]["passed"] == 40


def test_the_same_session_twice_is_refused(tmp_path):
    """Two copies of one file would double the store and look like traffic."""
    exports = two_stores(tmp_path)

    with pytest.raises(ValueError, match="same session"):
        build_rollup([exports[0], exports[1], exports[0]])


def test_a_pooled_sample_that_is_still_thin_has_no_rate(tmp_path):
    thin = range(101, 111)
    first_store, first = build(tmp_path, aisle_2=thin, name="a.db")
    with first_store:
        export = build_export(first_store, first)

    harrow = next(
        r for r in build_rollup([export])["rows"] if r["brand"] == "Harrow"
    )

    assert harrow["pooled"]["passed"] == 10
    assert harrow["pooled"]["reach_rate"]["value"] is None
    assert "below the 30" in harrow["pooled"]["reach_rate"]["withheld"]


def test_a_file_that_is_not_an_export_is_named(tmp_path):
    path = tmp_path / "digest.json"
    path.write_text(json.dumps({"findings": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="not an OpenV export") as error:
        load_export(path)

    assert str(path) in str(error.value)


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------

PLAN_MAP = StoreMap.from_dict(
    {
        "store": {"id": "store-014", "name": "Riverside <North>", "units": "m",
                  "outline": [[0, 0], [10, 0], [10, 8], [0, 8]]},
        "fixtures": [
            {"zone": "shelf-a", "label": "Bay A", "plan": [[1, 1], [2, 1], [2, 6], [1, 6]],
             "products": [item("1", "Northfield")]},
            {"zone": "shelf-c", "label": "Bay C", "plan": [[7, 1], [8, 1], [8, 6], [7, 6]],
             "products": [item("5", "Harrow")]},
        ],
    }
)


def test_the_plan_draws_every_fixture_with_what_happened_at_it(tmp_path):
    store, session = build(tmp_path, store_map=PLAN_MAP)
    with store:
        svg = render_plan(analyze_products(store, session), store.paths(session))

    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "BAY A" in svg and "BAY C" in svg
    assert "12 of 60 reached (20%)" in svg
    assert "0 of 40 reached (0%)" in svg
    # Text from the map is escaped, not trusted.
    assert "Riverside &lt;North&gt;" in svg
    assert "foot traffic" not in svg


def test_recorded_positions_become_a_traffic_layer_with_no_track_in_it(tmp_path):
    store, session = build(tmp_path, store_map=PLAN_MAP)
    with store:
        store.add_positions(
            session,
            [FloorPosition(track_id=t, frame=i * 30, t_s=float(i), x=4.0, y=1.0 + i)
             for t in (1, 2) for i in range(5)],
        )
        svg = render_plan(analyze_products(store, session), store.paths(session))

    assert "foot traffic" in svg
    assert svg.count('fill="#6fd6ff"') >= 5
    assert "track" not in svg


def test_a_map_with_no_shapes_cannot_be_drawn_and_says_why(tmp_path):
    store, session = build(tmp_path)  # STORE_MAP has no outline and no plans
    with store, pytest.raises(ValueError, match="nothing to draw"):
        render_plan(analyze_products(store, session))


# --------------------------------------------------------------------------
# The commands
# --------------------------------------------------------------------------

def test_export_prints_a_table_and_writes_the_file(tmp_path, capsys):
    store, _ = build(tmp_path)
    store.close()
    db, out = tmp_path / "e.db", tmp_path / "exports" / "store-014.json"

    code = main(["export", "--db", str(db), "--out", str(out)])
    printed = capsys.readouterr()

    assert code == 0
    assert "Northfield" in printed.out and "shelf-a" in printed.out
    assert "no number" in printed.out  # Oatwright, shared
    assert "no track ids" in printed.err
    assert load_export(out)["site"]["id"] == "store-014"


def test_export_json_is_parseable_on_stdout(tmp_path, capsys):
    store, _ = build(tmp_path)
    store.close()

    code = main(["export", "--db", str(tmp_path / "e.db"), "--json", "--brand", "Harrow"])

    assert code == 0
    assert json.loads(capsys.readouterr().out)["scope"]["brand"] == "Harrow"


def test_export_without_a_map_exits_cleanly(tmp_path, capsys):
    with EventStore(tmp_path / "e.db") as store:
        store.start_session("t.mp4", fps=30.0, width=1, height=1, zones=ZONES)

    code = main(["export", "--db", str(tmp_path / "e.db")])
    captured = capsys.readouterr()

    assert code == 1
    assert "no store map" in captured.err
    assert "Traceback" not in captured.out + captured.err


def test_rollup_and_planview_commands(tmp_path, capsys):
    exports = two_stores(tmp_path)
    paths = []
    for index, export in enumerate(exports):
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(export), encoding="utf-8")
        paths.append(str(path))

    assert main(["rollup", *paths]) == 0
    out = capsys.readouterr().out
    assert "Northfield: passed 120" in out
    assert "2 sessions from 2 stores" in out

    assert main(["rollup", paths[0], paths[0]]) == 1
    assert "same session" in capsys.readouterr().err

    # The first store's map has no shapes, so planview refuses with the reason.
    assert main(["planview", "--db", str(tmp_path / "a.db"), "--out", str(tmp_path / "p.svg")]) == 1
    assert "nothing to draw" in capsys.readouterr().err

    with EventStore(tmp_path / "a.db") as store:
        store.attach_store_map(store.latest_session_id(), PLAN_MAP, replace=True)
    assert main(["planview", "--db", str(tmp_path / "a.db"), "--out", str(tmp_path / "p.svg")]) == 0
    assert (tmp_path / "p.svg").read_text(encoding="utf-8").startswith("<svg")


def test_a_reach_in_a_session_is_read_against_that_sessions_map(tmp_path):
    """A planogram reset between two runs must not move last month's reaches
    onto this month's brand."""
    store, first = build(tmp_path)
    with store:
        reset = StoreMap.from_dict(
            {"store": {"id": "store-014"},
             "fixtures": [{"zone": "shelf-a", "products": [item("8", "Harrow")]}]}
        )
        second = store.start_session(
            "later.mp4", fps=30.0, width=700, height=400, zones=ZONES, pose=True,
            store_map=reset,
        )
        store.add_visits(second, [span(i, "aisle-1", 0.0, 5.0) for i in range(1, 41)])
        store.add_reaches(second, [span(i, "shelf-a", 1.0, 1.5) for i in range(1, 5)])

        before = build_export(store, first)
        after = build_export(store, second)

    assert {b["brand"] for b in before["brands"]} == set(
        p.brand for p in STORE_MAP.placements
    )
    harrow_after = next(b for b in after["brands"] if b["brand"] == "Harrow")
    assert harrow_after["measured_zones"] == ["shelf-a"]
    assert (harrow_after["passed"], harrow_after["reached"]) == (40, 4)
    northfield_before = next(b for b in before["brands"] if b["brand"] == "Northfield")
    assert northfield_before["reached"] == 12


# --------------------------------------------------------------------------
# The synthetic store
# --------------------------------------------------------------------------

def test_the_demo_store_runs_the_whole_product_layer(tmp_path, capsys):
    """`demo-store` is the first thing a new person runs, with no camera and no
    GPU. Every command the README points them at next has to work on it."""
    out = tmp_path / "demo"

    assert main(["demo-store", "--out", str(out)]) == 0
    db = str(out / "store-014.db")
    capsys.readouterr()

    assert main(["analyze", "--db", db]) == 0
    assert main(["measure", "shelf-endcap", "--db", db]) == 0
    measured = capsys.readouterr().out
    assert "shelf-endcap: improved" in measured

    assert main(["export", "--db", db, "--session", "1"]) == 0
    table = capsys.readouterr().out
    # One of each: a dead brand, a measured one, and one that shares its zone.
    assert "Tallow Lane" in table and "no number" in table

    assert main(["planview", "--db", db, "--out", str(out / "plan.svg")]) == 0
    assert "foot traffic" in (out / "plan.svg").read_text(encoding="utf-8")


def test_the_demo_store_says_it_is_synthetic_and_will_not_overwrite(tmp_path, capsys):
    out = tmp_path / "demo"
    assert main(["demo-store", "--out", str(out)]) == 0
    assert "invented" in capsys.readouterr().out

    with EventStore(out / "store-014.db") as store:
        # Labelled in the store itself, so it cannot pass for a measurement.
        assert all(s["source"].startswith("synthetic") for s in store.sessions())

    assert main(["demo-store", "--out", str(out)]) == 1
    assert "already exist" in capsys.readouterr().err
