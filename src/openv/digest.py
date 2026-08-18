"""What is worth telling someone, and when to say nothing.

A finding nobody reads is worth nothing, so findings have to reach the person who
can act on them. The hard part is not the sending, it is deciding what deserves
to be sent.

Two failure modes, and they pull against each other:

**Reporting everything.** A digest listing every zone is a wall of numbers. The
one line that mattered is in there somewhere, which is the same as it not being
there.

**Reporting nothing, loudly.** A daily message saying "nothing to report" teaches
people that the channel is noise, and by the time something matters they have
muted it. So a digest that has nothing to say says nothing at all, and
`worth_sending` is how the caller knows.

This module decides and formats. It does not send: delivery is a connector's job,
and keeping the decision separate means Slack, email and a webhook all inherit
the same judgement rather than each inventing their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openv.analysis import Change, Finding, StoreAnalysis, analyze, measure_change
from openv.store import EventStore

# Only these reach a person unprompted. A "low" finding is context for someone
# already looking, not a reason to interrupt.
NOTIFY_SEVERITIES = ("high", "medium")


@dataclass
class Digest:
    session_id: int | None
    total_shoppers: int
    findings: list[Finding] = field(default_factory=list)
    changes: list[Change] = field(default_factory=list)
    compared_with: int | None = None

    @property
    def worth_sending(self) -> bool:
        """Is there anything here a person needs to see?

        A conclusive change counts even when no finding is outstanding: "the
        thing you changed worked" is the most useful message this system sends.
        """
        return bool(self.findings) or any(c.conclusive for c in self.changes)

    @property
    def headline(self) -> str:
        parts = []
        if self.findings:
            n = len(self.findings)
            parts.append(f"{n} zone{'' if n == 1 else 's'} worth a look")
        conclusive = [c for c in self.changes if c.conclusive]
        if conclusive:
            improved = sum(1 for c in conclusive if c.verdict == "improved")
            worsened = len(conclusive) - improved
            if improved:
                parts.append(f"{improved} improved")
            if worsened:
                parts.append(f"{worsened} got worse")
        return ", ".join(parts) if parts else "nothing to report"

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "compared_with": self.compared_with,
            "total_shoppers": self.total_shoppers,
            "worth_sending": self.worth_sending,
            "headline": self.headline,
            "findings": [
                {
                    "zone": f.zone,
                    "severity": f.severity,
                    "kind": f.kind,
                    "headline": f.headline,
                    "passed": f.funnel.passed,
                    "stopped": f.funnel.stopped,
                    "reached": f.funnel.reached,
                }
                for f in self.findings
            ],
            "changes": [
                {
                    "zone": c.zone,
                    "verdict": c.verdict,
                    "reason": c.reason,
                    "conclusive": c.conclusive,
                    "delta": c.delta,
                }
                for c in self.changes
            ],
        }

    def render(self) -> str:
        """Plain text, wide enough for a terminal and narrow enough for a chat
        message. No markdown, because every destination mangles it differently."""
        if not self.worth_sending:
            return "Nothing to report."

        lines = [self.headline, ""]

        mark = {"high": "!!", "medium": " !"}
        for finding in self.findings:
            lines.append(f"{mark.get(finding.severity, '  ')} {finding.headline}")
            lines.append(
                f"     passed {finding.funnel.passed} -> stopped "
                f"{finding.funnel.stopped} -> reached {finding.funnel.reached}"
            )
            lines.append("")

        conclusive = [c for c in self.changes if c.conclusive]
        if conclusive:
            lines.append(f"since session {self.compared_with}:")
            for change in conclusive:
                arrow = "up" if change.verdict == "improved" else "down"
                lines.append(
                    f"  {change.zone}: {arrow} "
                    f"{abs(change.delta) * 100:.1f} points, {change.verdict}"
                )
            lines.append("")

        lines.append(
            f"{self.total_shoppers} shoppers observed. "
            "Numbers are measured, not estimated."
        )
        return "\n".join(lines).rstrip()


def build_digest(
    store: EventStore,
    session_id: int | None = None,
    compare_with: int | None = None,
) -> Digest:
    """Decide what is worth saying about a session.

    `compare_with` names an earlier session to measure against.

    Zones are re-measured if they are worrying **now or were worrying then**.
    Only checking the present would lose the single most useful message this
    system sends: a zone that was a problem, was changed, and is no longer a
    problem has by definition dropped out of the current findings, so looking
    only at those would announce every failure and no success.

    A shelf that was fine before and is fine now is left alone, because a verdict
    nobody asked for is just a longer message.
    """
    analysis: StoreAnalysis = analyze(store, session_id)

    findings = [f for f in analysis.findings if f.severity in NOTIFY_SEVERITIES]

    changes: list[Change] = []
    if compare_with is not None:
        worrying_now = {f.funnel.shelf_zone for f in findings}
        worrying_before = {
            f.funnel.shelf_zone
            for f in analyze(store, compare_with).findings
            if f.severity in NOTIFY_SEVERITIES
        }
        for zone in sorted(worrying_now | worrying_before):
            change = measure_change(store, zone, compare_with, session_id)
            if change is not None:
                changes.append(change)

    return Digest(
        session_id=session_id,
        total_shoppers=analysis.total_shoppers,
        findings=findings,
        changes=changes,
        compared_with=compare_with,
    )
