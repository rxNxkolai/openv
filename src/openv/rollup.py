"""One brand or category across several stores, from their export files.

A retailer reads one store. A brand buys coverage: the same question asked of
every store its product is in. This is that view, and it is built from export
files and nothing else, so it runs wherever those files are and needs no access
to any store's event store, footage or cameras. What crossed the wall to get
here was already aggregate.

Three things it will not do:

- **Count a file twice.** Two copies of the same session would double every
  number in it, and the result would look like a busier store. Duplicates are
  refused by name.
- **Pool what was not measured.** A brand that shares its zones at one store
  has no number there. That store is listed as contributing nothing, and is
  not quietly dropped or quietly estimated.
- **Hide the stores behind the total.** A pooled rate is one number standing
  in for several stores that may disagree, so every store's own row stays
  beside it.

Nothing links a shopper between two files. Someone who shops at two stores, or
at one store on two days, is counted in each, and that is the design rather
than a gap in it.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

from openv.analysis import MIN_SHOPPERS_FOR_CONFIDENCE
from openv.export import SCHEMA_ID
from openv.tools import Rate


def load_export(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc.msg} at line {exc.lineno}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA_ID:
        raise ValueError(
            f"{path} is not an OpenV export (expected schema {SCHEMA_ID!r}). "
            "`openv export --out` writes one"
        )
    return raw


def _pooled(count: int, of: int) -> dict[str, Any]:
    if of >= MIN_SHOPPERS_FOR_CONFIDENCE:
        return Rate(value=count / of, count=count, of=of).as_dict()
    return Rate(
        value=None,
        count=count,
        of=of,
        withheld=(
            f"{of} shopper{'' if of == 1 else 's'} observed across these stores, "
            f"below the {MIN_SHOPPERS_FOR_CONFIDENCE} needed to state a rate"
        ),
    ).as_dict()


def build_rollup(exports: list[dict[str, Any]], by: str = "brand") -> dict[str, Any]:
    """Combine exports into one row per brand (or category), with every store kept."""
    if by not in ("brand", "category"):
        raise ValueError(f"cannot roll up by {by!r}, expected brand or category")

    seen: dict[tuple, int] = {}
    for index, export in enumerate(exports):
        key = (export["site"]["id"], export["session"]["id"], export["session"]["started_at"])
        if key in seen:
            raise ValueError(
                f"file {index + 1} and file {seen[key] + 1} are the same session "
                f"(store {key[0]}, session {key[1]}). Counting it twice would "
                "double every number in it"
            )
        seen[key] = index

    section = "brands" if by == "brand" else "categories"
    grouped: dict[str, dict[str, Any]] = {}
    for export in exports:
        for row in export.get(section, []):
            name = row[by]
            entry = grouped.setdefault(
                name.casefold(), {by: name, "stores": [], "_p": 0, "_s": 0, "_r": 0}
            )
            store_row: dict[str, Any] = {
                "site": export["site"]["id"],
                "site_name": export["site"].get("name"),
                "session": export["session"]["id"],
                "started_at": export["session"]["started_at"],
                "facings": row["facings"],
                "measured": row["measured"],
            }
            if row["measured"]:
                store_row.update(
                    passed=row["passed"],
                    stopped=row["stopped"],
                    reached=row["reached"],
                    reach_rate=row["reach_rate"],
                    finding=row["finding"]["kind"],
                )
                entry["_p"] += row["passed"]
                entry["_s"] += row["stopped"]
                entry["_r"] += row["reached"]
            else:
                store_row["not_measured"] = row.get("not_measured")
            entry["stores"].append(store_row)

    rows = []
    for entry in grouped.values():
        passed, stopped, reached = entry.pop("_p"), entry.pop("_s"), entry.pop("_r")
        measured_in = sum(1 for s in entry["stores"] if s["measured"])
        entry["stores_measured"] = measured_in
        entry["stores_listed"] = len(entry["stores"])
        entry["pooled"] = (
            {
                "passed": passed,
                "stopped": stopped,
                "reached": reached,
                "reach_rate": _pooled(reached, passed),
            }
            if measured_in
            else None
        )
        rows.append(entry)

    rows.sort(key=lambda r: (-(r["pooled"]["passed"] if r["pooled"] else -1), r[by]))

    return {
        "schema": "openv.rollup/1",
        "by": by,
        "sessions": [
            {
                "site": e["site"]["id"],
                "site_name": e["site"].get("name"),
                "session": e["session"]["id"],
                "started_at": e["session"]["started_at"],
            }
            for e in exports
        ],
        "note": (
            "Pooled counts are sums over stores. Nothing links a shopper between "
            "files, so a shopper seen at two stores or on two days is counted in "
            "each."
        ),
        "rows": rows,
    }


def render(rollup: dict[str, Any]) -> str:
    by = rollup["by"]
    sites = sorted({s["site"] for s in rollup["sessions"]})
    lines = [
        f"{len(rollup['sessions'])} session"
        f"{'' if len(rollup['sessions']) == 1 else 's'} from {len(sites)} "
        f"store{'' if len(sites) == 1 else 's'}: {', '.join(sites)}",
    ]

    def rate(value: dict[str, Any]) -> str:
        return "not enough" if value["value"] is None else f"{value['value'] * 100:.0f}%"

    for row in rollup["rows"]:
        lines.append("")
        pooled = row["pooled"]
        if pooled is None:
            lines.append(f"{row[by]}: not measured at any of these stores")
        else:
            lines.append(
                f"{row[by]}: passed {pooled['passed']} -> stopped {pooled['stopped']}"
                f" -> reached {pooled['reached']}, reach {rate(pooled['reach_rate'])}"
                f"  (measured at {row['stores_measured']} of {row['stores_listed']})"
            )
        for store in row["stores"]:
            label = f"{store['site']} #{store['session']}"
            if store["measured"]:
                lines.append(
                    f"    {label:<24}passed {store['passed']:>5}  reached "
                    f"{store['reached']:>4}  reach {rate(store['reach_rate']):>10}"
                    f"  {store['finding']}"
                )
            else:
                lines.append(f"    {label:<24}no number: shares its zones or was not watched")

    lines.append("")
    lines.extend(textwrap.wrap(rollup["note"], width=78))
    return "\n".join(lines)
