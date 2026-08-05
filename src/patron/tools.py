"""The tool surface an agent is allowed to ask questions through.

`agent.py` already writes prose over findings it was handed. A chat surface needs
something looser than that, and the loose version is where this product can
quietly stop being defensible: a model with database access, or with a
text-to-SQL escape hatch, is a model doing arithmetic on raw rows, and a retailer
argues with the numbers long before they argue with the advice.

So the contract is the same as everywhere else in Patron. **The agent never
computes.** Every number it can cite came out of `analysis.py`, and these
functions are the only way it can reach them.

## Tools return verdicts, not ingredients

The failure mode worth designing against explicitly.

`analysis.py` refuses to state a rate below `MIN_SHOPPERS_FOR_CONFIDENCE`
shoppers, because "0% reach" off three shoppers is a lie dressed as data. If a
tool answered `{"passed": 12, "reached": 1}` a model could divide those and
assert "8% reach rate", which is precisely the number the deterministic layer
declined to state. The guard would live in the analysis layer and be defeated by
the transport.

Every rate therefore crosses this boundary as a `Rate`: a value that is either a
number or `None`, and when it is `None` it carries the reason in the same object.
The counts remain visible, because hiding them would be its own dishonesty, but
they arrive already labelled as not rate-able.

Nothing here computes a rate itself. `Rate.of` reads the guard off the funnel
that `analysis.py` already decided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from patron.analysis import (
    MIN_SHOPPERS_FOR_CONFIDENCE,
    Finding,
    StoreAnalysis,
    ZoneFunnel,
    analyze,
)
from patron.store import EventStore


@dataclass(frozen=True)
class Rate:
    """A proportion, or a stated refusal to assert one.

    Serialised with the refusal alongside the counts so a reader cannot see the
    ingredients without also seeing that the verdict was withheld.
    """

    value: float | None
    count: int
    of: int
    withheld: str | None = None

    @classmethod
    def from_funnel(cls, funnel: ZoneFunnel, count: int, value: float | None) -> Rate:
        if funnel.has_confidence:
            return cls(value=value, count=count, of=funnel.passed)
        return cls(
            value=None,
            count=count,
            of=funnel.passed,
            withheld=(
                f"{funnel.passed} shopper{'' if funnel.passed == 1 else 's'} observed, "
                f"below the {MIN_SHOPPERS_FOR_CONFIDENCE} needed to state a rate"
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "value": self.value,
            "count": self.count,
            "of": self.of,
        }
        if self.withheld is not None:
            out["withheld"] = self.withheld
        return out


def _funnel_dict(funnel: ZoneFunnel) -> dict[str, Any]:
    return {
        "shelf_zone": funnel.shelf_zone,
        "floor_zone": funnel.floor_zone,
        "passed": funnel.passed,
        "stopped": funnel.stopped,
        "reached": funnel.reached,
        "mean_dwell_s": round(funnel.mean_dwell_s, 2),
        "stop_rate": Rate.from_funnel(funnel, funnel.stopped, funnel.stop_rate).as_dict(),
        "reach_rate": Rate.from_funnel(funnel, funnel.reached, funnel.reach_rate).as_dict(),
    }


def _finding_dict(finding: Finding) -> dict[str, Any]:
    return {
        "zone": finding.zone,
        "kind": finding.kind,
        "severity": finding.severity,
        "headline": finding.headline,
        "confident": finding.confident,
        "benchmark_reach_rate": finding.benchmark_reach_rate,
        "funnel": _funnel_dict(finding.funnel),
    }


def _analysis(store: EventStore, session_id: int | None) -> StoreAnalysis:
    return analyze(store, session_id)


# --------------------------------------------------------------------------
# The tools themselves. Each returns plain JSON-serialisable data.
# --------------------------------------------------------------------------


def list_zones(store: EventStore, session_id: int | None = None) -> dict[str, Any]:
    """Zones that have data, and which floor each shelf was paired to.

    The pairing comes from where reachers were standing, not from configuration,
    so it is worth surfacing: it is a measured claim rather than a setting.
    """
    analysis = _analysis(store, session_id)
    return {
        "zones": [
            {
                "shelf_zone": f.funnel.shelf_zone,
                "floor_zone": f.funnel.floor_zone,
                "shoppers_observed": f.funnel.passed,
                "has_enough_data_for_rates": f.funnel.has_confidence,
            }
            for f in analysis.findings
        ],
        "total_shoppers": analysis.total_shoppers,
    }


def get_findings(
    store: EventStore,
    session_id: int | None = None,
    zone: str | None = None,
    severity: str | None = None,
) -> dict[str, Any]:
    """Ranked findings, already severity-ordered by the analysis layer."""
    analysis = _analysis(store, session_id)
    findings = [
        f
        for f in analysis.findings
        if (zone is None or f.zone == zone) and (severity is None or f.severity == severity)
    ]
    return {
        "findings": [_finding_dict(f) for f in findings],
        "store_median_reach_rate": analysis.median_reach_rate,
        "total_shoppers": analysis.total_shoppers,
        "benchmark_note": (
            "The benchmark is this store's own median, not an industry figure."
        ),
    }


def get_funnel(
    store: EventStore, zone: str, session_id: int | None = None
) -> dict[str, Any]:
    """The pass-by to engagement funnel for one shelf zone."""
    analysis = _analysis(store, session_id)
    for finding in analysis.findings:
        if finding.funnel.shelf_zone == zone:
            return {"found": True, "funnel": _funnel_dict(finding.funnel)}
    return {
        "found": False,
        "zone": zone,
        "reason": "no shelf zone by that name has reach data in this session",
        "available": [f.funnel.shelf_zone for f in analysis.findings],
    }


def compare_zones(
    store: EventStore, zone_a: str, zone_b: str, session_id: int | None = None
) -> dict[str, Any]:
    """Compare two shelf zones on reach rate.

    The comparison is computed here rather than left to the caller, so that two
    withheld rates cannot be silently subtracted into a confident difference.
    """
    analysis = _analysis(store, session_id)
    by_zone = {f.funnel.shelf_zone: f.funnel for f in analysis.findings}

    missing = [z for z in (zone_a, zone_b) if z not in by_zone]
    if missing:
        return {
            "comparable": False,
            "reason": f"no reach data for {', '.join(missing)}",
            "available": sorted(by_zone),
        }

    a, b = by_zone[zone_a], by_zone[zone_b]
    if not (a.has_confidence and b.has_confidence):
        thin = [f.shelf_zone for f in (a, b) if not f.has_confidence]
        return {
            "comparable": False,
            "reason": (
                f"{', '.join(thin)} {'has' if len(thin) == 1 else 'have'} fewer than "
                f"{MIN_SHOPPERS_FOR_CONFIDENCE} shoppers, so no rate exists to compare"
            ),
            "observed": {
                zone_a: {"passed": a.passed, "reached": a.reached},
                zone_b: {"passed": b.passed, "reached": b.reached},
            },
        }

    return {
        "comparable": True,
        "zone_a": {"zone": zone_a, "reach_rate": a.reach_rate, "passed": a.passed},
        "zone_b": {"zone": zone_b, "reach_rate": b.reach_rate, "passed": b.passed},
        "difference": a.reach_rate - b.reach_rate,
        "better": zone_a if a.reach_rate > b.reach_rate else zone_b,
    }


def get_recommendations(
    store: EventStore, session_id: int | None = None, status: str | None = None
) -> dict[str, Any]:
    """Stored recommendations and their state.

    Read only, deliberately. Nothing in this module changes a status, because
    approval is a human decision and that gate is the liability boundary.
    """
    rows = store.recommendations(session_id=session_id, status=status)
    return {
        "recommendations": [dict(r) for r in rows],
        "note": (
            "Recommendations are proposals. Approval is a human decision and no "
            "tool here can grant it."
        ),
    }


#: What the model is told it can call. Kept beside the implementations so a tool
#: cannot be advertised without existing, or exist without being described.
TOOL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "list_zones",
        "description": (
            "List shelf zones that have data, how many shoppers each was seen by, "
            "and which floor zone it was paired to from the data."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_findings",
        "description": (
            "Ranked findings about what is going wrong at the shelf, already "
            "ordered by severity. Optionally filtered by zone or severity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "zone": {"type": "string"},
                "severity": {"type": "string", "enum": ["high", "medium", "low", "none"]},
            },
            "required": [],
        },
    },
    {
        "name": "get_funnel",
        "description": (
            "The pass-by to stop to reach funnel for one shelf zone. Rates come "
            "back withheld, with a reason, when too few shoppers were observed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"zone": {"type": "string"}},
            "required": ["zone"],
        },
    },
    {
        "name": "compare_zones",
        "description": (
            "Compare two shelf zones on reach rate. Refuses to compare when "
            "either zone has too few shoppers for a rate to exist."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "zone_a": {"type": "string"},
                "zone_b": {"type": "string"},
            },
            "required": ["zone_a", "zone_b"],
        },
    },
    {
        "name": "get_recommendations",
        "description": (
            "Stored recommendations and their status. Read only: approval is a "
            "human decision and cannot be granted through a tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": [],
        },
    },
)

TOOLS = {
    "list_zones": list_zones,
    "get_findings": get_findings,
    "get_funnel": get_funnel,
    "compare_zones": compare_zones,
    "get_recommendations": get_recommendations,
}


def call(
    name: str, store: EventStore, session_id: int | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Dispatch a tool call by name.

    Unknown names are refused rather than guessed at, because a model inventing
    a plausible tool name is a model about to invent a plausible answer.
    """
    if name not in TOOLS:
        return {
            "error": f"no tool named {name!r}",
            "available": sorted(TOOLS),
        }
    return TOOLS[name](store, session_id=session_id, **kwargs)
