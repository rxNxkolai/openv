"""Flags that were set but cannot do anything.

This codebase has twice shipped a setting that looked like it worked and
silently did nothing: shelf zones were uncreatable from either drawing tool, and
`--floor` alone never opened the event store. Neither was visible at runtime,
which is what made them survive. A flag that cannot take effect has to say so,
because the alternative is someone trusting an empty result.
"""

import argparse

import pytest

from patron.cli import _inert_flag_warnings


def args(**overrides):
    base = dict(
        pose=False,
        zones=None,
        floor=None,
        out=None,
        show=False,
        no_trace=False,
        min_arm_extension=2.5,
        position_interval=1.0,
        enter_seconds=0.2,
        exit_seconds=0.5,
        lost_seconds=1.5,
        db="out/patron.db",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_a_plain_run_warns_about_nothing():
    assert _inert_flag_warnings(args(source="clip.mp4")) == []


def test_pose_without_zones_is_paid_for_and_unused():
    [warning] = _inert_flag_warnings(args(pose=True))

    assert "--pose without --zones" in warning
    # The cost is the point: this is not a stylistic nit, it is wasted compute.
    assert "14ms" in warning


def test_pose_with_zones_is_fine():
    assert _inert_flag_warnings(args(pose=True, zones="z.json")) == []


def test_arm_extension_without_pose_does_nothing():
    [warning] = _inert_flag_warnings(args(zones="z.json", min_arm_extension=3.0))

    assert "--min-arm-extension without --pose" in warning


def test_the_default_arm_extension_is_not_reported_as_set():
    # Only a deliberate value is worth a warning; the default is not a choice.
    assert _inert_flag_warnings(args(zones="z.json", min_arm_extension=2.5)) == []


def test_position_interval_without_floor_does_nothing():
    [warning] = _inert_flag_warnings(args(position_interval=5.0))

    assert "--position-interval without --floor" in warning


@pytest.mark.parametrize(
    "field,flag",
    [
        ("enter_seconds", "--enter-seconds"),
        ("exit_seconds", "--exit-seconds"),
        ("lost_seconds", "--lost-seconds"),
    ],
)
def test_debounce_settings_without_zones_do_nothing(field, flag):
    [warning] = _inert_flag_warnings(args(**{field: 9.0}))

    assert flag in warning
    assert "no visits are tracked" in warning


def test_a_database_nothing_writes_to_is_flagged():
    """The trap that shipped: --floor alone never opened the store at all."""
    [warning] = _inert_flag_warnings(args(db="out/mine.db"))

    assert "--db without --zones or --floor" in warning


def test_a_database_with_floor_alone_is_fine():
    assert _inert_flag_warnings(args(db="out/mine.db", floor="f.json")) == []


def test_no_trace_without_anything_rendered():
    [warning] = _inert_flag_warnings(args(no_trace=True))

    assert "--no-trace" in warning


def test_no_trace_is_fine_when_something_is_rendered():
    assert _inert_flag_warnings(args(no_trace=True, out="o.mp4")) == []
    assert _inert_flag_warnings(args(no_trace=True, show=True)) == []


def test_pose_with_zones_that_contain_no_shelf():
    warnings = _inert_flag_warnings(
        args(pose=True, zones="z.json"), shelf_zone_count=0
    )

    assert any("kind 'shelf'" in w for w in warnings)


def test_pose_with_a_shelf_zone_present_is_silent():
    assert _inert_flag_warnings(args(pose=True, zones="z.json"), shelf_zone_count=2) == []


def test_arm_extension_is_not_inert_when_pose_is_on():
    assert _inert_flag_warnings(args(pose=True, zones="z.json", min_arm_extension=3.0)) == []


def test_several_inert_flags_are_all_reported():
    warnings = _inert_flag_warnings(
        args(pose=True, position_interval=4.0, no_trace=True)
    )

    # One message per problem, so a user fixing them does not play whack-a-mole.
    assert len(warnings) == 3
    assert len({w.split(":")[0] for w in warnings}) == 3
