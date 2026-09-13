"""Gemini failures must map to stable, safe HTTP errors (STEP 2 contract)."""

import copy
import logging
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors

from app import main, memory
from tests.fakes import (
    CUSTOMER_MESSAGE,
    PROVIDER_SECRET,
    SENSITIVE_VALUES,
    SESSION_ID,
    provider_error,
)

EXPECTED_MESSAGES = {
    "AI_SERVICE_BUSY": "The assistant is busy right now. Please try again in a minute.",
    "AI_SERVICE_ERROR": "The assistant could not process your request. Please try again later.",
    "AI_SERVICE_UNAVAILABLE": "The assistant is temporarily unavailable. Please try again shortly.",
    "AI_SERVICE_TIMEOUT": "The assistant took too long to respond. Please try again.",
}

# (exception factory, HTTP status, error code, Retry-After)
FAILURE_CASES = [
    pytest.param(
        lambda: provider_error(errors.ClientError, 429),
        503, "AI_SERVICE_BUSY", "60",
        id="gemini-429",
    ),
    *[
        pytest.param(
            lambda code=code: provider_error(errors.ClientError, code),
            502, "AI_SERVICE_ERROR", None,
            id=f"gemini-{code}",
        )
        for code in (400, 401, 403, 404)
    ],
    *[
        pytest.param(
            lambda code=code: provider_error(errors.ServerError, code),
            503, "AI_SERVICE_UNAVAILABLE", "30",
            id=f"gemini-{code}",
        )
        for code in (500, 503)
    ],
    pytest.param(
        lambda: provider_error(errors.APIError, 302),
        502, "AI_SERVICE_ERROR", None,
        id="gemini-other-api-error",
    ),
    pytest.param(
        lambda: httpx.ReadTimeout(PROVIDER_SECRET),
        503, "AI_SERVICE_TIMEOUT", "30",
        id="read-timeout",
    ),
    # ConnectTimeout is both a TimeoutException and a TransportError;
    # it must be reported as a timeout.
    pytest.param(
        lambda: httpx.ConnectTimeout(PROVIDER_SECRET),
        503, "AI_SERVICE_TIMEOUT", "30",
        id="connect-timeout",
    ),
    pytest.param(
        lambda: httpx.ConnectError(PROVIDER_SECRET),
        503, "AI_SERVICE_UNAVAILABLE", "30",
        id="network-error",
    ),
]

EMPTY_REPLY_CASES = [
    pytest.param(None, id="text-none"),
    pytest.param("", id="text-empty"),
]


def seed_history():
    memory.add_message(SESSION_ID, "user", "earlier question")
    memory.add_message(SESSION_ID, "assistant", "earlier answer")
    return copy.deepcopy(memory.get_history(SESSION_ID))


def post_chat(client):
    return client.post(
        "/chat",
        json={"session_id": SESSION_ID, "message": CUSTOMER_MESSAGE},
    )


def assert_safe_error(response, all_logs, status, code, retry_after):
    assert response.status_code == status
    assert response.json() == {
        "error": {"code": code, "message": EXPECTED_MESSAGES[code]}
    }
    assert response.headers.get("retry-after") == retry_after

    # The documented OpenAPI model must match what is really returned.
    main.ErrorResponse.model_validate(response.json())

    for value in SENSITIVE_VALUES:
        assert value not in response.text
        assert value not in all_logs.text

    # The failure is still logged for operators, at warning level or above.
    assert any(
        record.name == "app.main" and record.levelno >= logging.WARNING
        for record in all_logs.records
    )


@pytest.mark.parametrize("make_exc, status, code, retry_after", FAILURE_CASES)
def test_gemini_failure_returns_safe_error(
    client, fake_gemini, all_logs, make_exc, status, code, retry_after
):
    history_before = seed_history()
    fake_gemini.raises(make_exc())

    response = post_chat(client)

    assert_safe_error(response, all_logs, status, code, retry_after)
    assert memory.get_history(SESSION_ID) == history_before


@pytest.mark.parametrize("text", EMPTY_REPLY_CASES)
def test_empty_reply_returns_safe_error(client, fake_gemini, all_logs, text):
    history_before = seed_history()
    fake_gemini.runs(lambda **_: SimpleNamespace(text=text))

    response = post_chat(client)

    assert_safe_error(response, all_logs, 502, "AI_SERVICE_ERROR", None)
    assert memory.get_history(SESSION_ID) == history_before


@pytest.mark.parametrize("make_exc, status, code, retry_after", FAILURE_CASES)
def test_failure_on_new_session_saves_nothing(
    client, fake_gemini, make_exc, status, code, retry_after
):
    fake_gemini.raises(make_exc())

    response = post_chat(client)

    assert response.status_code == status
    assert memory.sessions == {}


def test_unexpected_application_error_returns_generic_500(
    client_no_raise, fake_gemini
):
    history_before = seed_history()
    fake_gemini.raises(RuntimeError(PROVIDER_SECRET))

    response = post_chat(client_no_raise)

    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert PROVIDER_SECRET not in response.text
    assert CUSTOMER_MESSAGE not in response.text
    assert memory.get_history(SESSION_ID) == history_before
