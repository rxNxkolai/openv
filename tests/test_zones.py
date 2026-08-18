import json

import pytest

from openv.zones import Zone, ZoneSet

SQUARE = ((0, 0), (100, 0), (100, 100), (0, 100))


def test_contains_inside_outside_and_boundary():
    zone = Zone(name="endcap", polygon=SQUARE)

    assert zone.contains((50, 50)) is True
    assert zone.contains((150, 50)) is False
    assert zone.contains((0, 50)) is True  # on the edge counts as inside


def test_concave_polygon_excludes_the_notch():
    # An L-shaped zone, e.g. an aisle that wraps a corner. A bounding-box test
    # would wrongly include the notch.
    l_shape = ((0, 0), (100, 0), (100, 40), (40, 40), (40, 100), (0, 100))
    zone = Zone(name="corner", polygon=l_shape)

    assert zone.contains((20, 20)) is True
    assert zone.contains((80, 20)) is True
    assert zone.contains((80, 80)) is False  # the notch


def test_zone_needs_at_least_three_points():
    with pytest.raises(ValueError, match="at least 3 points"):
        Zone(name="bad", polygon=((0, 0), (10, 10)))


def test_containing_returns_every_overlapping_zone():
    zones = ZoneSet(
        zones=(
            Zone(name="aisle-6", polygon=((0, 0), (200, 0), (200, 200), (0, 200))),
            Zone(name="endcap", polygon=SQUARE),
        )
    )

    assert set(zones.containing((50, 50))) == {"aisle-6", "endcap"}
    assert zones.containing((150, 150)) == ("aisle-6",)
    assert zones.containing((500, 500)) == ()


def test_round_trip_through_json(tmp_path):
    original = ZoneSet(
        zones=(
            Zone(name="endcap", polygon=SQUARE, kind="shelf"),
            Zone(name="queue", polygon=((5, 5), (10, 5), (10, 10)), kind="checkout"),
        )
    )
    path = tmp_path / "zones.json"
    original.save(path)

    loaded = ZoneSet.load(path)
    assert loaded.names == ("endcap", "queue")
    assert loaded.zones[1].kind == "checkout"
    assert loaded.zones[0].polygon == SQUARE


def test_zone_with_no_declared_kind_loads_as_floor(tmp_path):
    path = tmp_path / "zones.json"
    path.write_text(
        json.dumps({"zones": [{"name": "aisle-6", "polygon": [[0, 0], [1, 0], [1, 1]]}]}),
        encoding="utf-8",
    )

    # Defaulting the other way would test an unlabelled polygon against wrists,
    # so it would report no visits at all, and no reaches either unless pose is
    # running. Floor is the kind that fails visibly rather than silently.
    loaded = ZoneSet.load(path)

    assert loaded.zones[0].kind == "floor"
    assert len(loaded.floor) == 1
    assert len(loaded.shelf) == 0


def test_duplicate_zone_names_are_rejected(tmp_path):
    path = tmp_path / "zones.json"
    path.write_text(
        json.dumps(
            {
                "zones": [
                    {"name": "endcap", "polygon": [[0, 0], [1, 0], [1, 1]]},
                    {"name": "endcap", "polygon": [[5, 5], [6, 5], [6, 6]]},
                ]
            }
        ),
        encoding="utf-8",
    )

    # Duplicate names would silently merge two different areas into one row in
    # every report, which is worse than failing loudly.
    with pytest.raises(ValueError, match="duplicate zone names"):
        ZoneSet.load(path)
