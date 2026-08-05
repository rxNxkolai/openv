"""Sending a digest somewhere, over a plain HTTP POST.

`digest.py` decides what is worth saying. This puts it somewhere a person will
see it. The split matters: Slack, Teams, a webhook and an email all inherit one
judgement about what deserves sending rather than each inventing their own.

**What leaves the machine.** Zone names, shopper counts, findings and verdicts.
No images, no frames, no track identifiers, nothing about any individual. This
is the same anonymised event data the store already holds, and the privacy
posture in CLAUDE.md is unaffected by shipping it to a channel. Worth saying out
loud, because "we post analytics to Slack" is the kind of sentence that alarms a
legal review until someone reads what is actually in the payload.

**A failed send is reported, never swallowed.** A digest that silently fails to
post is worse than one that was never scheduled: the numbers look attended to and
nobody is reading them.

stdlib only. A delivery mechanism is not worth a dependency, and the licence
discipline in CLAUDE.md applies to convenience just as much as to detectors.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

DEFAULT_TIMEOUT_S = 10.0

#: Slack, Discord, Teams and Google Chat incoming webhooks all accept a plain
#: `{"text": ...}` body, so one format covers the destinations people actually
#: use without a per-vendor adapter.
FORMATS = ("text", "json")


@dataclass(frozen=True)
class Delivery:
    ok: bool
    status: int | None
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "status": self.status, "detail": self.detail}


def build_payload(digest, fmt: str = "text") -> dict[str, Any]:
    """The exact body that would be posted.

    Separated from sending so `--dry-run` shows the real thing rather than an
    approximation of it. A preview that differs from what ships is worse than no
    preview.
    """
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}, expected one of {FORMATS}")
    if fmt == "text":
        return {"text": digest.render()}
    return digest.as_dict()


def _post(url: str, body: bytes, timeout: float) -> Delivery:
    request = urllib.request.Request(  # noqa: S310 - the URL is operator-supplied
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return Delivery(ok=True, status=response.status, detail="sent")
    except urllib.error.HTTPError as exc:
        # The body of an error is where Slack explains itself, so it is worth
        # more than the status code alone.
        detail = exc.read().decode("utf-8", "replace")[:300] or exc.reason
        return Delivery(ok=False, status=exc.code, detail=str(detail))
    except urllib.error.URLError as exc:
        return Delivery(ok=False, status=None, detail=f"could not reach it: {exc.reason}")
    except TimeoutError:
        return Delivery(ok=False, status=None, detail=f"timed out after {timeout:g}s")


def send(
    digest,
    url: str,
    fmt: str = "text",
    timeout: float = DEFAULT_TIMEOUT_S,
    transport: Callable[[str, bytes, float], Delivery] | None = None,
) -> Delivery:
    """Post a digest, unless it has nothing to say.

    A digest that is not worth sending is not sent, and that is reported as a
    success with nothing done rather than an error. Scheduled jobs run on quiet
    days too, and a quiet day is not a failure.
    """
    if not digest.worth_sending:
        return Delivery(ok=True, status=None, detail="nothing worth sending, not sent")

    body = json.dumps(build_payload(digest, fmt)).encode("utf-8")
    return (transport or _post)(url, body, timeout)
