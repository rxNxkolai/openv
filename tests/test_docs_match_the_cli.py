"""Does the README describe the program that exists?

It has been wrong before. `--min-arm-extension` was documented as a flag and
tuning advice was written around it while it existed only as a constructor
argument, so anybody following the README got an argparse error and no way to
know the docs were the broken part.

Documentation drift is invisible to every other test in this suite, and a
README is the first thing a new person trusts.
"""

import re
from pathlib import Path

import pytest

from openv.cli import main

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")

# Commands that need a camera, a GUI or a model, so their help is checked but
# they are never invoked for real here.
FENCE = re.compile(r"```bash\n(.*?)```", re.DOTALL)


def documented_invocations() -> list[tuple[str, list[str]]]:
    """Every `openv <command> ... --flag` written in a bash block."""
    found = []
    for block in FENCE.findall(README):
        # Join continuation lines so a wrapped command is read as one.
        joined = block.replace("\\\n", " ")
        for line in joined.splitlines():
            line = line.strip()
            if not line.startswith("uv run openv "):
                continue
            parts = line.split()
            command = parts[3] if len(parts) > 3 else None
            if command is None or command.startswith("-"):
                continue
            flags = [p for p in parts[4:] if p.startswith("--")]
            found.append((command, flags))
    return found


def help_text(command: str, capsys) -> str:
    """Help for a command, or a failure that says the command is unknown.

    The exit code matters. argparse exits for a good `--help` and for a bad
    command alike, so merely catching SystemExit would call a nonexistent
    command present.
    """
    with pytest.raises(SystemExit) as exit_info:
        main([command, "--help"])
    if exit_info.value.code not in (0, None):
        raise LookupError(f"openv has no command {command!r}")
    return capsys.readouterr().out


def test_the_readme_actually_contains_commands():
    """A guard on the guard: a parsing change that silently matched nothing
    would make every check below vacuously pass."""
    invocations = documented_invocations()

    assert len(invocations) >= 8
    assert {c for c, _ in invocations} >= {"track", "analyze", "digest"}


def test_every_documented_command_exists(capsys):
    unknown = []
    for command, _flags in documented_invocations():
        try:
            help_text(command, capsys)
        except LookupError:
            unknown.append(command)

    assert unknown == [], f"README documents commands that do not exist: {unknown}"


def test_the_command_check_can_actually_fail(capsys):
    """A guard on the guard.

    argparse exits for a good --help and a bad command alike, so a check that
    only caught SystemExit would pass for anything at all.
    """
    with pytest.raises(LookupError, match="no command"):
        help_text("summarise", capsys)


def test_every_documented_flag_exists_on_its_command(capsys):
    """The exact failure that shipped once already."""
    missing = []
    for command, flags in documented_invocations():
        text = help_text(command, capsys)
        for flag in flags:
            name = flag.split("=")[0]
            if name not in text:
                missing.append(f"{command} {name}")

    assert missing == [], f"README documents flags that do not exist: {missing}"


def test_flags_named_in_prose_exist_somewhere(capsys):
    """Backticked flags outside code blocks are documentation too.

    The one that broke was described in a sentence, not shown in a block, so
    checking only the blocks would have missed it.
    """
    commands = sorted({c for c, _ in documented_invocations()})
    every_help = " ".join(help_text(c, capsys) for c in commands)

    prose_flags = set(re.findall(r"`(--[a-z][a-z0-9-]+)`", README))
    unknown = sorted(f for f in prose_flags if f not in every_help)

    assert unknown == [], (
        f"README describes flags no command accepts: {unknown}"
    )
