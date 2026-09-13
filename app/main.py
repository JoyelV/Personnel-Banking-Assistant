import logging
import os

import httpx
from google.genai import errors, types
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from google import genai

from app.banking import get_account_balance
from app.auth import get_current_customer_id
from app.memory import get_history, add_message
from app.transactions import (
    get_recent_transactions,
    get_transaction_details,
)

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not configured")

# HttpOptions.timeout is in MILLISECONDS. httpx applies it per phase
# (connect / write / each read), not as a total deadline, and automatic
# function calling may make several model calls per /chat request.
GEMINI_TIMEOUT_MS = 30_000

QUOTA_RETRY_AFTER_SECONDS = 60
UNAVAILABLE_RETRY_AFTER_SECONDS = 30

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
)


def ai_error_response(
    status_code: int,
    error_code: str,
    message: str,
    retry_after_seconds: int | None = None,
) -> JSONResponse:
    headers = {}

    if retry_after_seconds is not None:
        headers["Retry-After"] = str(retry_after_seconds)

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": error_code,
                "message": message,
            }
        },
        headers=headers,
    )


class ChatRequest(BaseModel):
    session_id: str
    message: str


# OpenAPI documentation for the errors returned by ai_error_response().
# These models describe the response shape only; they are not used at runtime.
class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


RETRY_AFTER_HEADER = {
    "Retry-After": {
        "description": "Seconds to wait before retrying.",
        "schema": {"type": "integer"},
    }
}

CHAT_ERROR_RESPONSES = {
    502: {
        "model": ErrorResponse,
        "description": (
            "The AI provider rejected the request or returned no reply. "
            "Code: AI_SERVICE_ERROR."
        ),
    },
    503: {
        "model": ErrorResponse,
        "description": (
            "The AI provider is busy, unavailable, or timed out. Codes: "
            "AI_SERVICE_BUSY, AI_SERVICE_UNAVAILABLE, AI_SERVICE_TIMEOUT."
        ),
        "headers": RETRY_AFTER_HEADER,
    },
}


def get_my_account_balance():
    customer_id = get_current_customer_id()

    return get_account_balance(customer_id)

def get_my_recent_transactions():
    customer_id = get_current_customer_id()

    return get_recent_transactions(customer_id)

def get_my_transaction_details(transaction_id: str):
    customer_id = get_current_customer_id()

    return get_transaction_details(
        customer_id,
        transaction_id,
    )

@app.get("/")
def home():
    return {
        "message": "Banking AI Assistant is running!"
    }


@app.post("/chat", responses=CHAT_ERROR_RESPONSES)
def chat(request: ChatRequest):

    customer_id = get_current_customer_id()

    history = get_history(request.session_id)

    try:
        response = client.models.generate_content(
            model="gemini-3.7-flash",
            contents=(
                f"The authenticated customer ID is {customer_id}.\n"
                f"Conversation history: {history}\n"
                f"Customer message: {request.message}"
            ),
            config={
                "tools": [
                    get_my_account_balance,
                    get_my_recent_transactions,
                    get_my_transaction_details,
                ]
            }
        )

    # Log only the provider's code/status: never the prompt, the customer's
    # message, the session_id, or the raw provider error text.
    except errors.ClientError as exc:
        if exc.code == 429:
            logger.warning(
                "Gemini quota/rate limit: code=%s status=%s",
                exc.code,
                exc.status,
            )
            return ai_error_response(
                503,
                "AI_SERVICE_BUSY",
                "The assistant is busy right now. Please try again in a minute.",
                QUOTA_RETRY_AFTER_SECONDS,
            )

        # 400/401/403/404 mean our request or configuration is wrong.
        # Retrying will not help, so no Retry-After.
        logger.error(
            "Gemini rejected request: code=%s status=%s",
            exc.code,
            exc.status,
        )
        return ai_error_response(
            502,
            "AI_SERVICE_ERROR",
            "The assistant could not process your request. Please try again later.",
        )

    except errors.ServerError as exc:
        logger.warning(
            "Gemini server error: code=%s status=%s",
            exc.code,
            exc.status,
        )
        return ai_error_response(
            503,
            "AI_SERVICE_UNAVAILABLE",
            "The assistant is temporarily unavailable. Please try again shortly.",
            UNAVAILABLE_RETRY_AFTER_SECONDS,
        )

    except errors.APIError as exc:
        logger.error(
            "Gemini unexpected API error: code=%s status=%s",
            exc.code,
            exc.status,
        )
        return ai_error_response(
            502,
            "AI_SERVICE_ERROR",
            "The assistant could not process your request. Please try again later.",
        )

    except httpx.TimeoutException as exc:
        logger.warning("Gemini request timed out: %s", type(exc).__name__)
        return ai_error_response(
            503,
            "AI_SERVICE_TIMEOUT",
            "The assistant took too long to respond. Please try again.",
            UNAVAILABLE_RETRY_AFTER_SECONDS,
        )

    except httpx.TransportError as exc:
        logger.warning("Gemini network error: %s", type(exc).__name__)
        return ai_error_response(
            503,
            "AI_SERVICE_UNAVAILABLE",
            "The assistant is temporarily unavailable. Please try again shortly.",
            UNAVAILABLE_RETRY_AFTER_SECONDS,
        )

    reply = response.text

    if not reply:
        logger.error("Gemini returned an empty response")
        return ai_error_response(
            502,
            "AI_SERVICE_ERROR",
            "The assistant could not process your request. Please try again later.",
        )

    # Save the exchange only after success, so a failed request never
    # leaves an unanswered user message in the session.
    add_message(
        request.session_id,
        "user",
        request.message,
    )

    add_message(
        request.session_id,
        "assistant",
        reply,
    )

    return {
        "reply": reply
    }