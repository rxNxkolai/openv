"""The conversation loop behind the agent tab.

The model is faked. What is under test is the machinery around it: that
requested tools actually run, that their results go back, that every call is
kept as a citation, and that a model which never settles is cut off rather than
allowed to spend forever.

Faking the client also means this suite exercises the agent path with no
credentials, which matters because the real one has still never run.
"""

import json

import pytest

from patron.chat import ChatSession, ChatUnavailable
from patron.events import ZoneVisit
from patron.store import EventStore


# --------------------------------------------------------------------------
# A fake that speaks the shape the SDK returns
# --------------------------------------------------------------------------

class Block:
    def __init__(self, type, text=None, name=None, input=None, id=None):
        self.type = type
        self.text = text
        self.name = name
        self.input = input
        self.id = id


class Response:
    def __init__(self, content, stop_reason="end_turn", stop_details=None):
        self.content = content
        self.stop_reason = stop_reason
        self.stop_details = stop_details


class FakeClient:
    """Replays a script of responses and records what it was sent."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.messages = self

    def create(self, **request):
        self.requests.append(request)
        if not self.script:
            raise AssertionError("the loop asked for more turns than were scripted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def text(s):
    return Response([Block("text", text=s)])


def uses(name, params, id="tu_1", say=None):
    blocks = []
    if say:
        blocks.append(Block("text", text=say))
    blocks.append(Block("tool_use", name=name, input=params, id=id))
    return Response(blocks)


def build(tmp_path, shoppers=40, reachers=10):
    store = EventStore(tmp_path / "e.db")
    session = store.start_session("t.mp4", fps=30.0, width=1920, height=1080)

    def span(i, zone, a, b):
        return ZoneVisit(
            track_id=i, zone=zone, entered_frame=int(a * 30), entered_s=a,
            exited_frame=int(b * 30), exited_s=b,
        )

    store.add_visits(session, [span(i, "aisle", 0.0, 5.0) for i in range(1, shoppers + 1)])
    store.add_reaches(session, [span(i, "shelf", 1.0, 1.5) for i in range(1, reachers + 1)])
    return store, session


# --------------------------------------------------------------------------

def test_a_plain_answer_needs_no_tools(tmp_path):
    store, session = build(tmp_path)
    with store:
        chat = ChatSession(store, session, client=FakeClient([text("Nothing to look up.")]))
        answer = chat.ask("hello")

    assert answer.text == "Nothing to look up."
    assert answer.citations == []
    assert answer.rounds == 1


def test_a_requested_tool_actually_runs_and_its_result_goes_back(tmp_path):
    store, session = build(tmp_path)
    client = FakeClient([
        uses("get_funnel", {"zone": "shelf"}),
        text("Reach rate is 25%."),
    ])
    with store:
        chat = ChatSession(store, session, client=client)
        answer = chat.ask("how is the shelf doing")

    assert answer.text == "Reach rate is 25%."
    assert [c.tool for c in answer.citations] == ["get_funnel"]

    # The tool result must reach the model, not just be recorded for the panel.
    sent = client.requests[-1]["messages"]
    tool_result = sent[-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert json.loads(tool_result["content"])["funnel"]["reached"] == 10


def test_every_call_is_kept_in_order_as_the_citation_trail(tmp_path):
    store, session = build(tmp_path)
    client = FakeClient([
        uses("list_zones", {}, id="a"),
        uses("get_funnel", {"zone": "shelf"}, id="b"),
        text("done"),
    ])
    with store:
        chat = ChatSession(store, session, client=client)
        answer = chat.ask("walk me through it")

    # An answer without its trail is an opinion, so the order matters as much
    # as the presence: it is the reasoning a reader retraces.
    assert [c.tool for c in answer.citations] == ["list_zones", "get_funnel"]
    assert answer.citations[1].params == {"zone": "shelf"}
    assert answer.citations[1].result["found"] is True


def test_several_tools_in_one_turn_all_run(tmp_path):
    store, session = build(tmp_path)
    both = Response([
        Block("tool_use", name="list_zones", input={}, id="a"),
        Block("tool_use", name="get_funnel", input={"zone": "shelf"}, id="b"),
    ])
    client = FakeClient([both, text("done")])
    with store:
        chat = ChatSession(store, session, client=client)
        answer = chat.ask("everything please")

    assert len(answer.citations) == 2
    assert len(client.requests[-1]["messages"][-1]["content"]) == 2


def test_a_withheld_rate_reaches_the_model_still_withheld(tmp_path):
    """The guard has to survive the transport, which is the whole point."""
    store, session = build(tmp_path, shoppers=12, reachers=1)
    client = FakeClient([uses("get_funnel", {"zone": "shelf"}), text("Not enough data.")])
    with store:
        chat = ChatSession(store, session, client=client)
        chat.ask("what is the reach rate")

    payload = json.loads(client.requests[-1]["messages"][-1]["content"][0]["content"])
    reach = payload["funnel"]["reach_rate"]
    assert reach["value"] is None
    assert "below the 30 needed" in reach["withheld"]


def test_a_bad_argument_comes_back_as_something_correctable(tmp_path):
    store, session = build(tmp_path)
    client = FakeClient([
        uses("get_funnel", {"wrong_arg": 1}),
        text("Let me try that differently."),
    ])
    with store:
        chat = ChatSession(store, session, client=client)
        answer = chat.ask("funnel please")

    # A dead turn teaches the model nothing; an error it can read does.
    assert "error" in answer.citations[0].result
    assert answer.text == "Let me try that differently."


def test_an_invented_tool_name_is_refused(tmp_path):
    store, session = build(tmp_path)
    client = FakeClient([uses("get_everything", {}), text("Understood.")])
    with store:
        chat = ChatSession(store, session, client=client)
        answer = chat.ask("give me everything")

    assert "error" in answer.citations[0].result
    assert "get_findings" in answer.citations[0].result["available"]


def test_a_model_that_never_settles_is_cut_off(tmp_path):
    store, session = build(tmp_path)
    client = FakeClient([uses("list_zones", {}) for _ in range(10)])
    with store:
        chat = ChatSession(store, session, client=client, max_rounds=3)
        answer = chat.ask("loop forever")

    assert answer.truncated is True
    assert answer.rounds == 3
    assert "could not settle" in answer.text
    # The work it did do is still attached rather than thrown away.
    assert len(answer.citations) == 3


def test_history_survives_between_questions(tmp_path):
    store, session = build(tmp_path)
    client = FakeClient([text("first"), text("second")])
    with store:
        chat = ChatSession(store, session, client=client)
        chat.ask("one")
        chat.ask("two")

    roles = [m["role"] for m in client.requests[-1]["messages"]]
    # user, assistant, user: a follow-up has something to refer back to.
    assert roles == ["user", "assistant", "user"]


def test_a_refusal_is_raised_not_returned_as_an_answer(tmp_path):
    store, session = build(tmp_path)

    class Details:
        category = "policy"

    client = FakeClient([Response([], stop_reason="refusal", stop_details=Details())])
    with store:
        chat = ChatSession(store, session, client=client)
        with pytest.raises(ChatUnavailable, match="declined"):
            chat.ask("something disallowed")


def test_missing_credentials_raise_the_typed_error(tmp_path):
    store, session = build(tmp_path)
    # With no credentials at all the SDK raises TypeError from header validation
    # at request time, which a narrow AuthenticationError check would miss.
    client = FakeClient([TypeError("Could not resolve authentication method")])
    with store:
        chat = ChatSession(store, session, client=client)
        with pytest.raises(ChatUnavailable, match="credentials"):
            chat.ask("anything")


def test_the_tools_are_advertised_to_the_model(tmp_path):
    store, session = build(tmp_path)
    client = FakeClient([text("hi")])
    with store:
        ChatSession(store, session, client=client).ask("hi")

    names = {t["name"] for t in client.requests[0]["tools"]}
    assert "get_findings" in names
    assert not any("approve" in n for n in names)


def test_the_system_prompt_carries_the_constraints(tmp_path):
    store, session = build(tmp_path)
    client = FakeClient([text("hi")])
    with store:
        ChatSession(store, session, client=client).ask("hi")

    system = client.requests[0]["system"]
    for rule in ("never compute", "zones, not products", "never act", "own median"):
        assert rule in system.lower() or rule in system


def test_the_answer_serialises_for_a_ui(tmp_path):
    store, session = build(tmp_path)
    client = FakeClient([uses("list_zones", {}), text("ok")])
    with store:
        chat = ChatSession(store, session, client=client)
        payload = chat.ask("zones?").as_dict()

    assert json.dumps(payload)  # the evidence panel reads exactly this
    assert payload["citations"][0]["tool"] == "list_zones"
