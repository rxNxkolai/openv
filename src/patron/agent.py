"""The agent layer.

Reads the deterministic findings from `analysis.py` and writes the recommendation:
what to change at the shelf, why, and a draft of the change itself. This is the
layer the venture plan calls the product, and the one no verified competitor ships.

Two design rules it exists under:

*The agent never computes.* Every number it cites was computed in `analysis.py`
and handed to it. A model doing arithmetic over raw rows would be unauditable, and
a retailer will argue with the numbers before they argue with the advice.

*The agent never executes.* Output is a proposal with status `proposed`. Nothing
touches a merchandising system. Human-in-the-loop is the liability boundary the
plan's risk register calls for, and it is structural here, not a setting.

What it cannot do, honestly: Patron does not know SKUs. It knows zones. So the
recommendation is about placement, facing, and shelf position, not "move SKU-204".
That is a hardware limitation (see CLAUDE.md constraint 3), not a prompt problem,
and the system prompt forbids inventing product names to paper over it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from patron.analysis import Finding, StoreAnalysis

MODEL = "claude-opus-5"

SYSTEM = """\
You are the analysis layer of Patron, a shopper-behavior system that watches a \
store's security cameras and advises on merchandising.

You receive computed funnel measurements for zones of a store. Your job is to turn \
each finding into one concrete, physically actionable change, and to draft that \
change so a merchandiser could hand it to staff.

Rules you do not break:

1. Every number you cite must come from the data given to you. Do not compute new \
numbers, do not estimate, and do not introduce figures that are not present.
2. Patron measures zones, not products. You do not know what is on the shelf, what \
it costs, or what brand it is. Never invent a SKU, product name, brand, or price. \
Recommend changes to placement, shelf height, facing count, signage position, and \
adjacency, which are the things zone data can support.
3. If a finding is marked as low confidence or insufficient data, say so plainly in \
the recommendation and recommend collecting more observation rather than acting.
4. One action per finding. A list of five things to try is not a recommendation, it \
is a way of avoiding one. Pick the change with the best evidence behind it.
5. State the expected effect in terms of the measured funnel stage you expect to \
move, and be honest that it is an expectation, not a promise.

You are advising a human who will approve or reject. Write for someone who will \
push back."""

SCHEMA = {
    "type": "object",
    "properties": {
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "zone": {"type": "string"},
                    "diagnosis": {
                        "type": "string",
                        "description": "What the numbers say is wrong, in one sentence.",
                    },
                    "action": {
                        "type": "string",
                        "description": "The single concrete change to make.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why this action follows from this evidence.",
                    },
                    "expected_effect": {
                        "type": "string",
                        "description": "Which funnel stage should move, and roughly how much.",
                    },
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "drafted_change": {
                        "type": "string",
                        "description": "The change written out as an instruction staff could execute.",
                    },
                },
                "required": [
                    "zone",
                    "diagnosis",
                    "action",
                    "rationale",
                    "expected_effect",
                    "confidence",
                    "drafted_change",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["recommendations"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Recommendation:
    zone: str
    diagnosis: str
    action: str
    rationale: str
    expected_effect: str
    confidence: str
    drafted_change: str


class AdvisorUnavailable(RuntimeError):
    """No usable credentials, or the model declined. Analysis still works."""


def _finding_payload(finding: Finding) -> dict:
    f = finding.funnel
    return {
        "shelf_zone": f.shelf_zone,
        "serves_aisle": f.floor_zone,
        "finding_kind": finding.kind,
        "severity": finding.severity,
        "headline": finding.headline,
        "confident": finding.confident,
        "shoppers_who_passed": f.passed,
        "shoppers_who_stopped": f.stopped,
        "shoppers_who_reached": f.reached,
        "pass_by_rate": f.pass_by_rate,
        "stop_rate": f.stop_rate,
        "reach_rate": f.reach_rate,
        "mean_dwell_seconds": round(f.mean_dwell_s, 2),
        "store_median_reach_rate": finding.benchmark_reach_rate,
    }


def build_prompt(analysis: StoreAnalysis) -> str:
    payload = {
        "store_median_reach_rate": analysis.median_reach_rate,
        "total_shoppers_observed": analysis.total_shoppers,
        "findings": [_finding_payload(f) for f in analysis.findings],
        "measurement_caveats": [
            "Shopper counts are an upper bound: tracking can assign a new id to a "
            "shopper who was occluded and reacquired.",
            "A reach means a hand entered the shelf zone with the arm extended. It "
            "does not confirm a product was picked up or purchased.",
        ],
    }
    return (
        "Here are the measured funnels for this store session.\n\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        "Produce one recommendation per finding that has severity high or medium. "
        "Skip findings marked healthy. For findings marked insufficient_data, "
        "recommend further observation rather than a merchandising change."
    )


def advise(
    analysis: StoreAnalysis,
    model: str = MODEL,
    effort: str = "high",
) -> list[Recommendation]:
    """Ask the model for recommendations over already-computed findings.

    Raises AdvisorUnavailable rather than returning junk when there are no
    credentials or the request is declined. The deterministic analysis is the
    product's floor, so failing loudly here costs nothing.
    """
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise AdvisorUnavailable("the anthropic package is not installed") from exc

    if not analysis.findings:
        return []

    # Zero-arg constructor on purpose: it resolves ANTHROPIC_API_KEY, then
    # ANTHROPIC_AUTH_TOKEN, then an `ant auth login` profile. Hardcoding a key
    # here would break the profile path for no gain.
    client = anthropic.Anthropic()

    request = {
        "model": model,
        "max_tokens": 16000,
        "system": SYSTEM,
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": SCHEMA},
        },
        "messages": [{"role": "user", "content": build_prompt(analysis)}],
    }

    # Note the TypeError: with no credentials at all the SDK raises it from header
    # validation at request time, not AuthenticationError, and not at construction.
    # Catching only AuthenticationError leaves the commonest first-run failure
    # surfacing as a stack trace.
    no_credentials = (anthropic.AuthenticationError, TypeError)
    credentials_help = (
        "no usable credentials. Set ANTHROPIC_API_KEY, or run `ant auth login`"
    )

    try:
        # Server-side fallbacks: this model's safety classifiers can decline a
        # request, and a decline is a 200 with stop_reason "refusal", not an
        # error. "default" routes by refusal category rather than pinning a model.
        response = client.beta.messages.create(
            **request, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
        )
    except no_credentials as exc:
        raise AdvisorUnavailable(credentials_help) from exc
    except anthropic.BadRequestError:
        # The fallbacks beta may not be enabled for this account. The advice is
        # worth more than the fallback, so retry on the plain endpoint.
        try:
            response = client.messages.create(**request)
        except no_credentials as exc:
            raise AdvisorUnavailable(credentials_help) from exc

    # Check stop_reason before touching content: on a refusal, content is empty
    # or partial and indexing into it raises.
    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None)
        raise AdvisorUnavailable(f"the model declined this request (category: {category})")

    text = next((b.text for b in response.content if b.type == "text"), None)
    if not text:
        raise AdvisorUnavailable("the model returned no text content")

    return parse_recommendations(text)


def parse_recommendations(text: str) -> list[Recommendation]:
    """Parse the model's JSON into typed records."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AdvisorUnavailable(f"the model returned unparseable JSON: {exc}") from exc

    return [
        Recommendation(
            zone=item["zone"],
            diagnosis=item["diagnosis"],
            action=item["action"],
            rationale=item["rationale"],
            expected_effect=item["expected_effect"],
            confidence=item["confidence"],
            drafted_change=item["drafted_change"],
        )
        for item in data.get("recommendations", [])
    ]
