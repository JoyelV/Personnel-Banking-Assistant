import logging
import uuid

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from google.genai import errors
from pydantic import BaseModel

from app.auth import get_current_customer_id
from app.banking import BankingService
from app.context import RequestContext
from app.conversation import ConversationBusyError, store
from app.llm import EmptyModelReply
from app.mock_banking import MockBankingService
from app.orchestrator import run_turn
from app.tools import ToolExecutor

logger = logging.getLogger(__name__)

app = FastAPI()

# The banking backend. A real adapter replaces the mock here; the tools,
# agent loop and Gemini integration do not change.
banking_service: BankingService = MockBankingService()

QUOTA_RETRY_AFTER_SECONDS = 60
UNAVAILABLE_RETRY_AFTER_SECONDS = 30


def error_response(
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


# OpenAPI documentation for the errors returned by error_response().
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
    409: {
        "model": ErrorResponse,
        "description": (
            "Another message in this conversation is still being processed. "
            "Code: CONVERSATION_BUSY."
        ),
    },
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


@app.get("/")
def home():
    return {
        "message": "Banking AI Assistant is running!"
    }


@app.post("/chat", responses=CHAT_ERROR_RESPONSES)
def chat(request: ChatRequest):

    # Identity comes from the backend only. It goes to the tool executor
    # and is never sent to the model.
    ctx = RequestContext(
        customer_id=get_current_customer_id(),
        session_id=request.session_id,
        request_id=uuid.uuid4().hex,
    )

    try:
        with store.turn(ctx.session_id):
            executor = ToolExecutor(ctx, banking_service)

            try:
                reply = run_turn(
                    store.recent_messages(ctx.session_id),
                    request.message,
                    executor,
                )
            finally:
                # Executed tool calls are audited even if the turn fails.
                store.record_tool_calls(ctx.session_id, executor.records)

            # Save the exchange only after success, so a failed request never
            # leaves an unanswered user message in the session.
            store.commit_turn(ctx.session_id, request.message, reply)

    except ConversationBusyError:
        logger.warning("Conversation busy: request_id=%s", ctx.request_id)
        return error_response(
            409,
            "CONVERSATION_BUSY",
            "Your previous message is still being processed. "
            "Please wait for the reply and try again.",
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
            return error_response(
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
        return error_response(
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
        return error_response(
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
        return error_response(
            502,
            "AI_SERVICE_ERROR",
            "The assistant could not process your request. Please try again later.",
        )

    except httpx.TimeoutException as exc:
        logger.warning("Gemini request timed out: %s", type(exc).__name__)
        return error_response(
            503,
            "AI_SERVICE_TIMEOUT",
            "The assistant took too long to respond. Please try again.",
            UNAVAILABLE_RETRY_AFTER_SECONDS,
        )

    except httpx.TransportError as exc:
        logger.warning("Gemini network error: %s", type(exc).__name__)
        return error_response(
            503,
            "AI_SERVICE_UNAVAILABLE",
            "The assistant is temporarily unavailable. Please try again shortly.",
            UNAVAILABLE_RETRY_AFTER_SECONDS,
        )

    except EmptyModelReply:
        logger.error("Gemini returned an empty response")
        return error_response(
            502,
            "AI_SERVICE_ERROR",
            "The assistant could not process your request. Please try again later.",
        )

    return {
        "reply": reply
    }
