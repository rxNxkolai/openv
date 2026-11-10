"""Product, brand and category numbers, and the one rule they are built on.

A reach lands in a zone. Whether it can be called a reach for a brand depends
entirely on whether that brand had the zone to itself, and every way of getting
a number out of a shared zone is an assumption about shoppers presented as a
measurement of them. These tests hold the join to the things it can actually
know.
"""

import pytest

from openv.agent import build_prompt
from openv.analysis import MIN_SHOPPERS_FOR_CONFIDENCE, analyze
from openv.events import ZoneVisit
from openv.products import EXCLUSIVE, SHARED, analyze_products
from openv.store import EventStore
from openv.storemap import StoreMap
from openv.tools import TOOL_SPECS, TOOLS, call
from openv.zones import Zone, ZoneSet

# Two aisles. shelf-a and shelf-b face aisle-1, shelf-c faces aisle-2.
ZONES = ZoneSet(
    zones=(
        Zone(name="shelf-a", polygon=((0, 0), (100, 0), (100, 200), (0, 200)), kind="shelf"),
        Zone(name="shelf-b", polygon=((0, 200), (100, 200), (100, 400), (0, 400)), kind="shelf"),
        Zone(name="aisle-1", polygon=((100, 0), (300, 0), (300, 400), (100, 400)), kind="floor"),
        Zone(name="shelf-c", polygon=((400, 0), (500, 0), (500, 400), (400, 400)), kind="shelf"),
        Zone(name="aisle-2", polygon=((500, 0), (700, 0), (700, 400), (500, 400)), kind="floor"),
    )
)


def item(sku, brand, category="cereal", facings=1):
    return {"sku": sku, "product": f"item {sku}", "brand": brand,
            "category": category, "facings": facings}


#: Northfield has shelf-a to itself and shares shelf-b with Oatwright. Harrow
#: has shelf-c to itself, with a single item on it.
STORE_MAP = StoreMap.from_dict(
    {
        "store": {"id": "store-014", "name": "Riverside"},
        "fixtures": [
            {"zone": "shelf-a", "products": [item("1", "Northfield", facings=4),
                                             item("2", "Northfield", facings=2)]},
            {"zone": "shelf-b", "products": [item("3", "Northfield", facings=3),
                                             item("4", "Oatwright", facings=1)]},
            {"zone": "shelf-c", "products": [item("5", "Harrow", category="snacks")]},
        ],
    }
)


def span(track_id: int, zone: str, entered_s: float, exited_s: float) -> ZoneVisit:
    return ZoneVisit(
        track_id=track_id,
        zone=zone,
        entered_frame=int(entered_s * 30),
        entered_s=entered_s,
        exited_frame=int(exited_s * 30),
        exited_s=exited_s,
    )


def build(tmp_path, store_map=STORE_MAP, aisle_2=range(101, 141), name="e.db"):
    """60 shoppers in aisle-1, 12 reach shelf-a and 6 reach shelf-b.

    40 shoppers in aisle-2 and nobody reaches shelf-c, which was watched with
    pose on, so that is a dead fixture and not a missing row.
    """
    store = EventStore(tmp_path / name)
    session = store.start_session(
        "t.mp4", fps=30.0, width=700, height=400, zones=ZONES, pose=True,
        store_map=store_map,
    )
    store.add_visits(session, [span(i, "aisle-1", 0.0, 5.0) for i in range(1, 61)])
    store.add_visits(session, [span(i, "aisle-2", 10.0, 15.0) for i in aisle_2])
    store.add_reaches(session, [span(i, "shelf-a", 1.0, 1.5) for i in range(1, 13)])
    store.add_reaches(session, [span(i, "shelf-b", 2.0, 2.5) for i in range(7, 13)])
    return store, session


# --------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------

def test_a_brand_alone_in_a_zone_owns_that_zones_funnel(tmp_path):
    store, session = build(tmp_path)
    with store:
        products = analyze_products(store, session)
        zone = next(
            f.funnel for f in analyze(store, session).findings if f.zone == "shelf-a"
        )

    brand = products.brand("Northfield")

    assert brand.measured_zones == ("shelf-a",)
    # Number for number the zone's own funnel, so anybody can reproduce a
    # brand figure by running `openv analyze` by hand.
    assert (brand.funnel.passed, brand.funnel.stopped, brand.funnel.reached) == (
        zone.passed, zone.stopped, zone.reached,
    )
    assert brand.funnel.reached == 12
    assert brand.funnel.passed == 60


def test_a_shared_zone_gives_no_brand_a_number(tmp_path):
    """The whole point. Six reaches landed in shelf-b, which holds two brands,
    and nothing in the data says whose they were."""
    store, session = build(tmp_path)
    with store:
        products = analyze_products(store, session)

    oatwright = products.brand("Oatwright")

    assert oatwright.measured is False
    assert oatwright.funnel is None
    assert oatwright.measured_zones == ()
    shared = oatwright.shared_zones[0]
    assert (shared.zone, shared.facings, shared.zone_facings, shared.others) == (
        "shelf-b", 1, 4, 1,
    )
    assert shared.share_of_facings == pytest.approx(0.25)


def test_a_shared_zone_adds_nothing_to_the_brand_that_is_also_measured_elsewhere(tmp_path):
    """Northfield is in shelf-b too. Its six reaches there must not leak into
    the brand figure, by facings share or any other route."""
    store, session = build(tmp_path)
    with store:
        products = analyze_products(store, session)

    northfield = products.brand("Northfield")

    assert northfield.funnel.reached == 12  # shelf-a only, not 12 + any of the 6
    assert [s.zone for s in northfield.shared_zones] == ["shelf-b"]
    assert northfield.facings == 9  # a fact about the shelf, all three items


def test_shelves_and_items_carry_their_attribution(tmp_path):
    store, session = build(tmp_path)
    with store:
        products = analyze_products(store, session)

    by_zone = {s.zone: s for s in products.shelves}
    assert by_zone["shelf-a"].brand_attribution == EXCLUSIVE
    assert by_zone["shelf-b"].brand_attribution == SHARED
    assert by_zone["shelf-c"].brand_attribution == EXCLUSIVE

    by_sku = {i.placement.sku: i for i in products.items}
    # Two Northfield items share shelf-a: the brand is measured, the item is not.
    assert by_sku["1"].attribution == SHARED
    assert by_sku["1"].shares_zone_with == 1
    # One item alone on shelf-c: the zone's numbers are the item's numbers.
    assert by_sku["5"].attribution == EXCLUSIVE


def test_a_brand_nobody_reached_for_is_a_finding_worded_for_a_brand(tmp_path):
    store, session = build(tmp_path)
    with store:
        products = analyze_products(store, session)

    harrow = products.brand("harrow")  # case does not matter to a person typing

    assert harrow.finding.kind == "dead"
    assert harrow.funnel.passed == 40
    assert "Harrow: 40 shoppers walked past its shelf and not one reached" == (
        harrow.finding.headline
    )
    # A brand with no number sorts last, never ahead of one that has a finding.
    assert products.brands[-1].name == "Oatwright"


def test_categories_roll_up_by_the_same_rule(tmp_path):
    store, session = build(tmp_path)
    with store:
        products = analyze_products(store, session)

    by_name = {c.name: c for c in products.categories}

    # shelf-a and shelf-b both hold only cereal, so cereal is measured over
    # both even though two brands share shelf-b. Shared for brands is not
    # shared for the category.
    assert set(by_name["cereal"].measured_zones) == {"shelf-a", "shelf-b"}
    assert by_name["cereal"].funnel.reached == 12  # tracks 1-12, each once
    assert by_name["cereal"].funnel.passed == 60
    assert by_name["snacks"].finding.kind == "dead"


# --------------------------------------------------------------------------
# Each shopper once
# --------------------------------------------------------------------------

def test_a_shopper_passing_two_of_a_brands_fixtures_is_one_shopper(tmp_path):
    """Ten shoppers walk both aisles. A brand with a fixture in each saw 90
    people, not 100, and two of them reached at both and count once."""
    both = StoreMap.from_dict(
        {
            "store": {"id": "store-014"},
            "fixtures": [
                {"zone": "shelf-a", "products": [item("1", "Northfield")]},
                {"zone": "shelf-c", "products": [item("2", "Northfield")]},
            ],
        }
    )
    store, session = build(tmp_path, store_map=both, aisle_2=range(51, 91))
    with store:
        # 51 and 52 already reached nothing at shelf-a; give them a reach
        # there, and eight shoppers a reach at shelf-c including those two.
        store.add_reaches(session, [span(i, "shelf-a", 1.0, 1.5) for i in (51, 52)])
        store.add_reaches(session, [span(i, "shelf-c", 11.0, 11.5) for i in range(51, 59)])
        products = analyze_products(store, session)

    brand = products.brand("Northfield")

    assert set(brand.measured_zones) == {"shelf-a", "shelf-c"}
    assert brand.funnel.passed == 90
    assert brand.funnel.reached == 12 + 2 + 8 - 2


# --------------------------------------------------------------------------
# What it will not say
# --------------------------------------------------------------------------

def test_no_map_means_no_product_analysis_at_all(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        session = store.start_session("t.mp4", fps=30.0, width=1, height=1)

        assert analyze_products(store, session) is None


def test_a_mapped_fixture_that_was_not_watched_is_not_called_dead(tmp_path):
    """Pose off means no reach could have been seen. That is a setting, and
    reporting its products as untouched would be a false accusation."""
    with EventStore(tmp_path / "e.db") as store:
        session = store.start_session(
            "t.mp4", fps=30.0, width=700, height=400, zones=ZONES, pose=False,
            store_map=STORE_MAP,
        )
        store.add_visits(session, [span(i, "aisle-2", 0.0, 5.0) for i in range(1, 41)])
        products = analyze_products(store, session)

    harrow = products.brand("Harrow")
    assert harrow.measured is False
    assert harrow.unwatched_zones == ("shelf-c",)
    assert "shelf-c" in products.unwatched_zones
    assert all(i.attribution == "unwatched" for i in products.items)


def test_a_thin_brand_has_counts_and_no_rate(tmp_path):
    thin = range(101, 101 + MIN_SHOPPERS_FOR_CONFIDENCE - 1)
    store, session = build(tmp_path, aisle_2=thin)
    with store:
        out = call("get_brand", store, session_id=session, brand="Harrow")

    assert out["found"] is True
    assert out["reach_rate"]["value"] is None
    assert "below the 30" in out["reach_rate"]["withheld"]
    assert out["finding"]["kind"] == "insufficient_data"


# --------------------------------------------------------------------------
# The agent's route to a product
# --------------------------------------------------------------------------

def test_every_advertised_tool_exists_and_the_product_tools_are_among_them():
    assert {spec["name"] for spec in TOOL_SPECS} == set(TOOLS)
    assert {"list_products", "get_brand"} <= set(TOOLS)


def test_the_product_tools_say_so_when_there_is_no_map(tmp_path):
    with EventStore(tmp_path / "e.db") as store:
        session = store.start_session("t.mp4", fps=30.0, width=1, height=1)

        listed = call("list_products", store, session_id=session)
        brand = call("get_brand", store, session_id=session, brand="Northfield")

    assert listed["mapped"] is False and brand["mapped"] is False
    assert "no store map" in listed["reason"]


def test_list_products_puts_attribution_beside_every_zone(tmp_path):
    store, session = build(tmp_path)
    with store:
        out = call("list_products", store, session_id=session)
        one = call("list_products", store, session_id=session, zone="shelf-b")
        missing = call("list_products", store, session_id=session, zone="shelf-z")

    attribution = {z["zone"]: z["brand_attribution"] for z in out["zones"]}
    assert attribution == {"shelf-a": EXCLUSIVE, "shelf-b": SHARED, "shelf-c": EXCLUSIVE}
    assert [p["brand"] for p in one["zones"][0]["products"]] == ["Northfield", "Oatwright"]
    assert missing["found"] is False and "shelf-a" in missing["available"]


def test_an_unmeasured_brand_comes_back_with_its_reason_and_no_numbers(tmp_path):
    store, session = build(tmp_path)
    with store:
        out = call("get_brand", store, session_id=session, brand="Oatwright")
        unknown = call("get_brand", store, session_id=session, brand="Acme")

    assert out["measured"] is False
    assert "reach_rate" not in out and "passed" not in out and "reached" not in out
    assert "not a share of the reaches" in out["not_measured"]
    assert unknown["found"] is False
    assert "Northfield" in unknown["available"]


def test_the_advisor_is_handed_the_shelf_contents_and_only_when_they_exist(tmp_path):
    import json

    store, session = build(tmp_path)
    with store:
        analysis = analyze(store, session)

    contents = {"shelf-c": [item("5", "Harrow", category="snacks")]}
    with_map = json.loads(build_prompt(analysis, contents).split("\n\n")[1])
    without = json.loads(build_prompt(analysis).split("\n\n")[1])

    by_zone = {f["shelf_zone"]: f for f in with_map["findings"]}
    assert by_zone["shelf-c"]["on_this_shelf"][0]["brand"] == "Harrow"
    assert "on_this_shelf" not in by_zone["shelf-a"]
    assert all("on_this_shelf" not in f for f in without["findings"])
