"""What the commands do when the input is wrong.

Most of these paths have only ever been walked by hand. A command that dies with
a traceback is telling the user their install is broken when actually they typed
a filename wrong, and on an edge box nobody is watching it will just be a
non-zero exit with a stack trace in a log.

Every case here should exit non-zero with something a person can act on, or
succeed quietly when the input is merely empty rather than wrong.
"""

import json

import pytest

from openv.cli import main
from openv.store import EventStore

LIGHT_COMMANDS = [
    ["analyze"],
    ["report"],
    ["digest"],
    ["sessions"],
    ["threads"],
    ["measure", "shelf"],
]


@pytest.mark.parametrize("command", LIGHT_COMMANDS, ids=lambda c: c[0])
def test_a_missing_database_is_reported_not_raised(command, tmp_path, capsys):
    code = main([*command, "--db", str(tmp_path / "absent.db")])

    assert code == 1
    assert "no database" in capsys.readouterr().err


@pytest.mark.parametrize("command", LIGHT_COMMANDS, ids=lambda c: c[0])
def test_an_empty_database_is_not_an_error(command, tmp_path, capsys):
    """Empty is a state, not a fault.

    A store that exists and holds nothing is what every deployment looks like on
    its first morning, and a crash there reads as a broken install.
    """
    db = tmp_path / "empty.db"
    EventStore(db).close()

    code = main([*command, "--db", str(db)])
    output = capsys.readouterr()

    # digest exits 2 for "nothing worth sending", measure exits 1 for "nothing
    # to compare". Neither is a crash, and neither should produce a traceback.
    assert code in (0, 1, 2)
    assert "Traceback" not in output.out + output.err


def test_measure_on_a_zone_that_was_never_seen(tmp_path, capsys):
    db = tmp_path / "e.db"
    EventStore(db).close()

    code = main(["measure", "no-such-zone", "--db", str(db)])
    error = capsys.readouterr().err

    assert code == 1
    assert "reach data in 0 sessions" in error
    # Points at the command that would show what does exist.
    assert "openv sessions" in error


def test_threads_on_a_thread_that_does_not_exist(tmp_path, capsys):
    db = tmp_path / "e.db"
    EventStore(db).close()

    code = main(["threads", "--db", str(db), "--thread", "42"])

    assert code == 0
    assert "no messages in thread 42" in capsys.readouterr().out


def test_ask_checks_the_database_before_reaching_for_credentials(tmp_path, capsys):
    """Order matters: a missing file is a better error than a missing key when
    both are true, because the file is the one the user got wrong."""
    code = main(["ask", "anything", "--db", str(tmp_path / "absent.db")])

    assert code == 1
    assert "no database" in capsys.readouterr().err


def test_digest_rejects_an_unknown_format_before_touching_anything(tmp_path):
    db = tmp_path / "e.db"
    EventStore(db).close()

    # argparse choices, so this exits rather than returning.
    with pytest.raises(SystemExit) as exit_info:
        main(["digest", "--db", str(db), "--format", "xml"])

    assert exit_info.value.code != 0


def test_a_malformed_zones_file_names_the_file_and_the_line(tmp_path):
    """A raw JSONDecodeError reads as a broken install.

    What actually happened is that somebody hand-edited a zones file and
    dropped a comma, and the message should say so.
    """
    from openv.zones import ZoneSet

    path = tmp_path / "bad.zones.json"
    path.write_text('{\n  "zones": [\n', encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON") as error:
        ZoneSet.load(path)

    assert "bad.zones.json" in str(error.value)
    assert "line" in str(error.value)


def test_a_zones_file_missing_its_polygon_key(tmp_path):
    from openv.zones import ZoneSet

    path = tmp_path / "bad.zones.json"
    path.write_text(json.dumps({"zones": [{"name": "a"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing") as error:
        ZoneSet.load(path)

    assert "polygon" in str(error.value)


def test_a_zones_file_with_no_zones_key(tmp_path):
    from openv.zones import ZoneSet

    path = tmp_path / "empty.zones.json"
    path.write_text(json.dumps({"areas": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="no 'zones' key"):
        ZoneSet.load(path)


def test_a_two_point_polygon_names_its_file(tmp_path):
    from openv.zones import ZoneSet

    path = tmp_path / "thin.zones.json"
    path.write_text(
        json.dumps({"zones": [{"name": "a", "polygon": [[0, 0], [1, 1]]}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least 3 points") as error:
        ZoneSet.load(path)

    assert "thin.zones.json" in str(error.value)


def test_a_malformed_floor_file_names_the_file(tmp_path):
    from openv.floor import FloorMap

    path = tmp_path / "bad.floor.json"
    path.write_text("not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON") as error:
        FloorMap.load(path)

    assert "bad.floor.json" in str(error.value)


def test_a_floor_file_with_a_malformed_correspondence(tmp_path):
    from openv.floor import FloorMap

    path = tmp_path / "bad.floor.json"
    path.write_text(
        json.dumps({"correspondences": [{"image": [0, 0]}]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="'image' and a 'floor'"):
        FloorMap.load(path)


def test_a_floor_file_with_too_few_points_is_refused_on_load(tmp_path):
    from openv.floor import FloorMap

    path = tmp_path / "bad.floor.json"
    path.write_text(
        json.dumps(
            {
                "units": "m",
                "correspondences": [
                    {"image": [0, 0], "floor": [0, 0]},
                    {"image": [1, 0], "floor": [1, 0]},
                ],
            }
        ),
        encoding="utf-8",
    )

    # Better here than three hours into a recording session.
    with pytest.raises(ValueError, match="at least 4 correspondences") as error:
        FloorMap.load(path)

    assert "bad.floor.json" in str(error.value)


def test_a_floor_file_with_degenerate_points_is_refused_on_load(tmp_path):
    from openv.floor import FloorMap

    path = tmp_path / "flat.floor.json"
    path.write_text(
        json.dumps(
            {
                "units": "m",
                "correspondences": [
                    {"image": [x, 100], "floor": [x / 100, 0]} for x in (0, 100, 200, 300)
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        FloorMap.load(path)


def test_no_subcommand_exits_rather_than_doing_something_arbitrary():
    with pytest.raises(SystemExit) as exit_info:
        main([])

    assert exit_info.value.code != 0
