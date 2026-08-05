"""The conversation loop behind the agent tab.

`advise` hands the model a fixed set of findings and asks for prose. A chat
surface cannot work that way, because the question decides which numbers matter.
So the model gets tools instead of a payload, and `tools.py` is the only route
to a number.

Two things this module is responsible for, beyond turning a question into an
answer:

**The citation trail.** Every tool call and its result is kept, in order, and
returned alongside the answer. That is what makes a claim checkable: a category
manager who disagrees with a number can be shown the call that produced it. An
answer without its trail is an opinion.

**Refusing to loop forever.** A model that keeps asking for tools and never
writes an answer would spend money indefinitely, so the cycle is bounded and
says so when it runs out rather than returning a half-answer as if it were whole.

The deterministic layer stays the floor. If there are no credentials this raises,
and `patron analyze` is unaffected, because it needs no model at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from patron.store import EventStore
from patron.tools import TOOL_SPECS, call

MODEL = "claude-opus-5"

# Enough for a question that needs a funnel, a comparison and a check of the
# store median. Past this the model is not converging and should say so.
MAX_TOOL_ROUNDS = 8

SYSTEM = """\
You answer questions about shopper behaviour in a retail store, for the staff who
work in it. You are looking at measurements from ceiling cameras.

**You never compute.** Every number you state must have come back from a tool
call in this conversation. Do not add, divide, average or estimate. If you want a
comparison, call the comparison tool rather than subtracting two numbers
yourself.

**Withheld rates stay withheld.** A rate arrives as an object with a `value` and
sometimes a `withheld` reason. When `value` is null there is no rate, and the
counts beside it are not raw material for you to divide. Say what was observed
and say why a rate cannot be stated yet. Never turn "1 of 12 shoppers" into a
percentage.

**You know zones, not products.** The cameras see areas of floor and shelf, and
the store's own names for them. You cannot see products, brands, prices or SKUs,
and you must never invent one to make an answer feel more useful. If someone asks
which product is underperforming, say plainly that Patron measures zones and
offer the zone-level answer.

**You never act.** You can describe what a change would be. You cannot approve
one, schedule one, or claim one has been made. Approval is a human decision.

**You do not identify people.** Track identifiers exist only within one session,
carry no identity, and questions about a specific individual cannot and will not
be answered.

The benchmark is always this store's own median, never an industry figure.

Be brief and concrete. Lead with the answer. If the data does not support an
answer, say so first rather than last.
"""


class ChatUnavailable(RuntimeError):
    """No model reachable. The deterministic layer is unaffected."""


@dataclass(frozen=True)
class Citation:
    """One tool call and what it returned, kept so a claim can be checked."""

    tool: str
    params: dict[str, Any]
    result: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "params": self.params, "result": self.result}


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    rounds: int = 0
    truncated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "citations": [c.as_dict() for c in self.citations],
            "rounds": self.rounds,
            "truncated": self.truncated,
        }


def _client():
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ChatUnavailable("the anthropic package is not installed") from exc
    # Zero-arg on purpose: resolves ANTHROPIC_API_KEY, then ANTHROPIC_AUTH_TOKEN,
    # then an `ant auth login` profile.
    return anthropic.Anthropic()


class ChatSession:
    """One conversation against one event store.

    History is held here rather than rebuilt per question, so a follow-up like
    "and the aisle next to it" has something to refer to.
    """

    def __init__(
        self,
        store: EventStore,
        session_id: int | None = None,
        model: str = MODEL,
        client: Any | None = None,
        max_rounds: int = MAX_TOOL_ROUNDS,
        conversation_id: int | None = None,
    ) -> None:
        self.store = store
        self.session_id = session_id
        self.model = model
        self.max_rounds = max_rounds
        self._client = client
        self.messages: list[dict[str, Any]] = []
        # Persistence is opt-in. A one-off question from the CLI does not need a
        # thread, and creating one for every invocation would fill the list with
        # single-question stubs nobody returns to.
        self.conversation_id = conversation_id

    def _ensure_client(self):
        if self._client is None:
            self._client = _client()
        return self._client

    def _run_tool(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        # Tool errors are returned to the model rather than raised, so a bad
        # argument becomes something it can correct instead of a dead turn.
        try:
            return call(name, self.store, session_id=self.session_id, **params)
        except TypeError as exc:
            return {"error": f"bad arguments for {name}: {exc}"}

    def resume(self, conversation_id: int) -> ChatSession:
        """Reopen a stored thread so a follow-up has its history.

        Only the prose is replayed, not the tool-call blocks. Those referred to
        specific tool_use ids from a finished exchange, and reviving them would
        hand the model dangling references to calls it can no longer complete.
        The citations stay on the stored message for the reader.
        """
        self.conversation_id = conversation_id
        self.messages = [
            {"role": row["role"], "content": row["text"]}
            for row in self.store.conversation(conversation_id)
            if row["text"]
        ]
        return self

    def ask(self, question: str) -> Answer:
        client = self._ensure_client()
        self.messages.append({"role": "user", "content": question})
        if self.conversation_id is not None:
            self.store.add_message(self.conversation_id, "user", question)

        answer = Answer(text="")

        for round_index in range(self.max_rounds):
            response = self._create(client)
            answer.rounds = round_index + 1

            if getattr(response, "stop_reason", None) == "refusal":
                category = getattr(
                    getattr(response, "stop_details", None), "category", None
                )
                raise ChatUnavailable(
                    f"the model declined this request (category: {category})"
                )

            blocks = list(response.content)
            self.messages.append({"role": "assistant", "content": blocks})

            tool_uses = [b for b in blocks if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                answer.text = "".join(
                    b.text for b in blocks if getattr(b, "type", None) == "text"
                )
                return self._finish(answer)

            results = []
            for block in tool_uses:
                params = dict(block.input or {})
                result = self._run_tool(block.name, params)
                answer.citations.append(
                    Citation(tool=block.name, params=params, result=result)
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": _as_text(result),
                    }
                )
            self.messages.append({"role": "user", "content": results})

        # Out of rounds with no answer written. Returning the partial text as if
        # it were finished would be the worst option available.
        answer.truncated = True
        answer.text = (
            f"I could not settle this within {self.max_rounds} rounds of looking "
            f"things up. The calls I made are attached."
        )
        return self._finish(answer)

    def _finish(self, answer: Answer) -> Answer:
        """Store the answer with its evidence, if this thread is being kept."""
        if self.conversation_id is not None:
            self.store.add_message(
                self.conversation_id,
                "assistant",
                answer.text,
                citations=[c.as_dict() for c in answer.citations],
                truncated=answer.truncated,
            )
        return answer

    def _create(self, client):
        request = {
            "model": self.model,
            "max_tokens": 8000,
            "system": SYSTEM,
            "tools": list(TOOL_SPECS),
            # A snapshot, not the live list. Passing the reference means anything
            # holding the request sees later turns appear inside it, which makes
            # the conversation unauditable after the fact.
            "messages": list(self.messages),
        }
        try:
            return client.messages.create(**request)
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed error below
            if _is_credentials_problem(exc):
                raise ChatUnavailable(
                    "no usable credentials. Set ANTHROPIC_API_KEY, or run "
                    "`ant auth login`"
                ) from exc
            raise


def _is_credentials_problem(exc: Exception) -> bool:
    """With no credentials at all the SDK raises TypeError from header
    validation at request time, not AuthenticationError, so the commonest
    first-run failure is the one a narrow check would miss."""
    if isinstance(exc, TypeError):
        return True
    return type(exc).__name__ in {"AuthenticationError", "PermissionDeniedError"}


def _as_text(result: dict[str, Any]) -> str:
    import json

    return json.dumps(result, default=str)
