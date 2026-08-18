"""Threads that survive a restart.

A question asked last week should be reopenable rather than retyped, and the
evidence behind its answer has to still be there. That second part is the
subtle one: a tool result is a claim about a moment, so it is stored rather than
recomputed. Re-running the same call next month answers a different question.
"""

import json

from test_chat import FakeClient, build, text, uses

from openv.chat import ChatSession


def test_a_thread_records_both_sides(tmp_path):
    store, session = build(tmp_path)
    with store:
        thread = store.start_conversation("endcap question", session)
        chat = ChatSession(
            store, session, client=FakeClient([text("Traffic is fine.")]),
            conversation_id=thread,
        )
        chat.ask("how is the endcap")

        messages = store.conversation(thread)

    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["text"] == "how is the endcap"
    assert messages[1]["text"] == "Traffic is fine."


def test_the_evidence_is_stored_with_the_answer(tmp_path):
    store, session = build(tmp_path)
    with store:
        thread = store.start_conversation("funnel", session)
        chat = ChatSession(
            store, session,
            client=FakeClient([uses("get_funnel", {"zone": "shelf"}), text("25%.")]),
            conversation_id=thread,
        )
        chat.ask("reach rate?")

        answer = store.conversation(thread)[-1]

    # Stored, not recomputed: a tool result is a claim about a moment, and an
    # answer whose evidence has silently moved underneath it is worse than one
    # with no evidence at all.
    assert answer["citations"][0]["tool"] == "get_funnel"
    assert answer["citations"][0]["result"]["funnel"]["reached"] == 10


def test_a_question_without_a_thread_stores_nothing(tmp_path):
    store, session = build(tmp_path)
    with store:
        chat = ChatSession(store, session, client=FakeClient([text("ok")]))
        chat.ask("just asking")

        # One-off questions should not fill the thread list with stubs.
        assert store.conversations() == []


def test_resuming_replays_the_prose_but_not_the_tool_blocks(tmp_path):
    store, session = build(tmp_path)
    with store:
        thread = store.start_conversation("ongoing", session)
        first = ChatSession(
            store, session,
            client=FakeClient([uses("list_zones", {}), text("Two zones.")]),
            conversation_id=thread,
        )
        first.ask("which zones")

        client = FakeClient([text("The second one.")])
        second = ChatSession(store, session, client=client).resume(thread)
        second.ask("and the busier of them?")

    sent = client.requests[-1]["messages"]
    # Reviving the tool_use blocks would hand the model dangling references to
    # calls from a finished exchange that it can no longer complete.
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert all(isinstance(m["content"], str) for m in sent)
    assert sent[1]["content"] == "Two zones."


def test_threads_list_newest_first_with_counts(tmp_path):
    store, _session = build(tmp_path)
    with store:
        older = store.start_conversation("older", None)
        newer = store.start_conversation("newer", None)
        store.add_message(older, "user", "a")
        store.add_message(older, "assistant", "b")
        store.add_message(newer, "user", "c")

        threads = store.conversations()

    assert [t["id"] for t in threads] == [newer, older]
    assert {t["title"]: t["message_count"] for t in threads} == {"newer": 1, "older": 2}


def test_an_empty_thread_still_lists(tmp_path):
    store, _session = build(tmp_path)
    with store:
        store.start_conversation("nothing said yet", None)
        threads = store.conversations()

    # A LEFT JOIN rather than an inner one, so a thread opened and abandoned is
    # visible and can be cleaned up rather than invisibly occupying an id.
    assert threads[0]["message_count"] == 0


def test_threads_can_be_scoped_to_a_session(tmp_path):
    store, session = build(tmp_path)
    with store:
        store.start_conversation("scoped", session)
        store.start_conversation("unscoped", None)

        assert [t["title"] for t in store.conversations(session)] == ["scoped"]
        assert len(store.conversations()) == 2


def test_a_truncated_answer_is_marked_as_such(tmp_path):
    store, session = build(tmp_path)
    with store:
        thread = store.start_conversation("looping", session)
        chat = ChatSession(
            store, session,
            client=FakeClient([uses("list_zones", {}) for _ in range(5)]),
            conversation_id=thread, max_rounds=2,
        )
        chat.ask("go forever")

        stored = store.conversation(thread)[-1]

    # Reading a cut-off answer later as if it were complete is exactly the
    # confusion the flag exists to prevent.
    assert stored["truncated"] is True
    assert "could not settle" in stored["text"]


def test_citations_round_trip_as_json(tmp_path):
    store, _session = build(tmp_path)
    with store:
        thread = store.start_conversation("t", None)
        payload = [{"tool": "get_funnel", "params": {"zone": "a"}, "result": {"x": 1}}]
        store.add_message(thread, "assistant", "text", citations=payload)

        stored = store.conversation(thread)[0]

    assert stored["citations"] == payload
    assert json.dumps(stored["citations"])
