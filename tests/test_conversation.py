"""ConversationStore: bounded history, atomic turns, per-session busy lock."""

import threading

import pytest

from app.conversation import ConversationBusyError, ConversationStore, Message
from app.tools import ToolAuditRecord


def test_recent_messages_are_bounded_and_start_with_the_customer():
    store = ConversationStore(history_window=3)
    store.commit_turn("s", "q1", "a1")
    store.commit_turn("s", "q2", "a2")

    # The last 3 messages start with an assistant message, which is dropped.
    assert store.recent_messages("s") == [
        Message("user", "q2"),
        Message("assistant", "a2"),
    ]


def test_history_window_must_be_positive():
    with pytest.raises(ValueError):
        ConversationStore(history_window=0)


def test_returned_history_cannot_modify_the_store():
    store = ConversationStore()
    store.commit_turn("s", "q", "a")

    store.recent_messages("s").clear()
    store.messages("s").clear()

    assert len(store.messages("s")) == 2


def test_second_turn_on_a_busy_session_fails_fast():
    store = ConversationStore()

    with store.turn("s"):
        with pytest.raises(ConversationBusyError):
            with store.turn("s"):
                pass

        # Other sessions are not blocked.
        with store.turn("other"):
            pass


def test_session_is_released_when_the_turn_raises():
    store = ConversationStore()

    with pytest.raises(RuntimeError):
        with store.turn("s"):
            raise RuntimeError("failure")

    with store.turn("s"):
        pass


def test_tool_audit_is_kept_separately_from_history():
    store = ConversationStore()
    record = ToolAuditRecord("r", "get_my_account_balance", {}, "ok")

    store.record_tool_calls("s", [])
    assert store.session_ids() == set()

    store.record_tool_calls("s", [record])
    assert store.tool_calls("s") == [record]
    assert store.messages("s") == []
    assert store.session_ids() == set()


def test_concurrent_commits_keep_each_turn_together():
    store = ConversationStore()

    def commit(worker):
        for i in range(50):
            store.commit_turn("s", f"q-{worker}-{i}", f"a-{worker}-{i}")

    threads = [threading.Thread(target=commit, args=(w,)) for w in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    messages = store.messages("s")
    assert len(messages) == 2000
    for user, assistant in zip(messages[::2], messages[1::2]):
        assert user.role == "user" and assistant.role == "assistant"
        assert user.content[1:] == assistant.content[1:]
