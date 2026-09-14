"""Successful /chat behaviour, conversation memory, and app regressions."""

import os

import pytest

from app import llm
from app.conversation import Message, store
from tests.fakes import CUSTOMER_MESSAGE, SESSION_ID, TEST_API_KEY, everything_sent

REPLY = "Your balance is 25,430.50 INR."


def test_app_uses_dummy_key_not_real_env_key():
    assert os.environ["GEMINI_API_KEY"] == TEST_API_KEY
    assert llm.api_key == TEST_API_KEY


def test_home_returns_running_message(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Banking AI Assistant is running!"}


def test_chat_success_returns_reply(client, fake_gemini):
    fake_gemini.returns(REPLY)

    response = client.post(
        "/chat",
        json={"session_id": SESSION_ID, "message": CUSTOMER_MESSAGE},
    )

    assert response.status_code == 200
    assert response.json() == {"reply": REPLY}
    assert len(fake_gemini.calls) == 1


def test_sessions_start_empty_after_previous_test():
    # The previous test saved messages; the clean_sessions fixture must
    # have discarded them.
    assert store.session_ids() == set()


def test_success_saves_user_then_assistant(client, fake_gemini):
    fake_gemini.returns(REPLY)

    client.post(
        "/chat",
        json={"session_id": SESSION_ID, "message": CUSTOMER_MESSAGE},
    )

    assert store.messages(SESSION_ID) == [
        Message("user", CUSTOMER_MESSAGE),
        Message("assistant", REPLY),
    ]


def test_second_turn_sends_history_and_current_message_once(client, fake_gemini):
    first_message = "first-turn-message-a1"
    second_message = "second-turn-message-b2"
    fake_gemini.returns(REPLY)

    client.post("/chat", json={"session_id": SESSION_ID, "message": first_message})
    client.post("/chat", json={"session_id": SESSION_ID, "message": second_message})

    second_request = fake_gemini.calls[1]["contents"]
    assert [content.role for content in second_request] == ["user", "model", "user"]
    assert [content.parts[0].text for content in second_request] == [
        first_message,
        REPLY,
        second_message,
    ]

    sent = everything_sent(fake_gemini.calls[1])
    assert sent.count(first_message) == 1
    assert sent.count(second_message) == 1

    assert [m.content for m in store.messages(SESSION_ID)] == [
        first_message,
        REPLY,
        second_message,
        REPLY,
    ]


def test_sessions_do_not_share_history(client, fake_gemini):
    fake_gemini.returns(REPLY)

    client.post("/chat", json={"session_id": "session-a", "message": "message-a"})
    client.post("/chat", json={"session_id": "session-b", "message": "message-b"})

    assert "message-a" not in everything_sent(fake_gemini.calls[1])
    assert store.messages("session-a")[0].content == "message-a"
    assert store.messages("session-b")[0].content == "message-b"


def test_model_is_never_told_customer_identity_or_api_key(client, fake_gemini):
    fake_gemini.returns(REPLY)

    client.post(
        "/chat",
        json={"session_id": SESSION_ID, "message": CUSTOMER_MESSAGE},
    )

    sent = everything_sent(fake_gemini.calls[0])
    for customer_id in ("CUST001", "CUST002", "CUST003"):
        assert customer_id not in sent
    assert TEST_API_KEY not in sent


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="empty-body"),
        pytest.param({"session_id": SESSION_ID}, id="missing-message"),
        pytest.param({"message": CUSTOMER_MESSAGE}, id="missing-session-id"),
        pytest.param({"session_id": SESSION_ID, "message": 123}, id="message-not-string"),
    ],
)
def test_malformed_chat_request_returns_422(client, fake_gemini, body):
    response = client.post("/chat", json=body)

    assert response.status_code == 422
    assert fake_gemini.calls == []
    assert store.session_ids() == set()


def test_invalid_json_returns_422(client, fake_gemini):
    response = client.post(
        "/chat",
        content="not json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert fake_gemini.calls == []


def test_openapi_documents_chat_error_responses(client):
    spec = client.get("/openapi.json").json()
    responses = spec["paths"]["/chat"]["post"]["responses"]

    assert {"200", "409", "422", "502", "503"} <= set(responses)

    error_ref = "#/components/schemas/ErrorResponse"
    for status in ("409", "502", "503"):
        schema = responses[status]["content"]["application/json"]["schema"]
        assert schema == {"$ref": error_ref}

    assert "CONVERSATION_BUSY" in responses["409"]["description"]
    assert "AI_SERVICE_ERROR" in responses["502"]["description"]
    for code in ("AI_SERVICE_BUSY", "AI_SERVICE_UNAVAILABLE", "AI_SERVICE_TIMEOUT"):
        assert code in responses["503"]["description"]

    assert "Retry-After" in responses["503"]["headers"]
    assert "headers" not in responses["502"]
    assert "headers" not in responses["409"]

    schemas = spec["components"]["schemas"]
    assert schemas["ErrorResponse"]["required"] == ["error"]
    assert sorted(schemas["ErrorDetail"]["required"]) == ["code", "message"]
