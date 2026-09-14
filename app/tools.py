"""Banking tools offered to the model, and the backend executor that runs them.

The model can only *request* a tool call. ToolExecutor decides what happens:
it checks the tool name, validates the arguments, takes the customer identity
from the RequestContext, authorizes, executes, and returns a controlled result.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable

from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.banking import get_account_balance
from app.context import RequestContext
from app.transactions import get_recent_transactions, get_transaction_details

logger = logging.getLogger(__name__)


class ToolArgs(BaseModel):
    # Unknown fields (e.g. an injected customer_id) and type coercion are
    # rejected, not ignored.
    model_config = ConfigDict(extra="forbid", strict=True)


class NoArgs(ToolArgs):
    pass


class TransactionDetailsArgs(ToolArgs):
    # Matches the mock data's ID format; revisit with the real banking API.
    transaction_id: str = Field(pattern=r"^TXN[0-9]{4,12}$")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[ToolArgs]
    parameters: types.Schema | None
    handler: Callable[[RequestContext, Any], dict]

    def declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


# Handlers shape their results explicitly: only fields the model needs,
# and never the customer identity.
def _account_balance(ctx: RequestContext, args: NoArgs) -> dict:
    account = get_account_balance(ctx.customer_id)

    return {
        "found": account["found"],
        "balance": account["balance"],
        "currency": account["currency"],
    }


def _recent_transactions(ctx: RequestContext, args: NoArgs) -> dict:
    return {"transactions": get_recent_transactions(ctx.customer_id)}


def _transaction_details(ctx: RequestContext, args: TransactionDetailsArgs) -> dict:
    return get_transaction_details(ctx.customer_id, args.transaction_id)


TOOL_REGISTRY: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            name="get_my_account_balance",
            description="Get the current balance of the customer's account.",
            args_model=NoArgs,
            parameters=None,
            handler=_account_balance,
        ),
        ToolSpec(
            name="get_my_recent_transactions",
            description="List the customer's recent transactions.",
            args_model=NoArgs,
            parameters=None,
            handler=_recent_transactions,
        ),
        ToolSpec(
            name="get_my_transaction_details",
            description=(
                "Get the details of one of the customer's transactions "
                "by its transaction ID."
            ),
            args_model=TransactionDetailsArgs,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "transaction_id": types.Schema(
                        type=types.Type.STRING,
                        description="Transaction ID, for example TXN1001.",
                    )
                },
                required=["transaction_id"],
            ),
            handler=_transaction_details,
        ),
    )
}


def declarations() -> list[types.FunctionDeclaration]:
    return [spec.declaration() for spec in TOOL_REGISTRY.values()]


class ToolNotAuthorized(Exception):
    pass


def authorize(ctx: RequestContext, spec: ToolSpec) -> None:
    # Every current tool reads the requesting customer's own data, and the
    # handlers look data up by ctx.customer_id, so ownership is enforced by
    # construction. Real authorization rules arrive with STEP 6.
    if not ctx.customer_id:
        raise ToolNotAuthorized(spec.name)


@dataclass(frozen=True)
class ToolAuditRecord:
    request_id: str
    tool: str
    # Validated arguments only; None when the call never passed validation.
    arguments: dict | None
    status: str


# Fixed messages sent back to the model. Internal error details never are.
TOOL_ERROR_MESSAGES = {
    "unknown_tool": "This tool does not exist.",
    "invalid_arguments": "The tool arguments were invalid.",
    "unauthorized": "This tool call is not authorized.",
    "failed": "The tool is temporarily unavailable.",
    "budget_exceeded": "The tool call limit for this request was reached.",
}


class ToolExecutor:
    """Runs model-requested tool calls for one request."""

    def __init__(
        self,
        ctx: RequestContext,
        registry: dict[str, ToolSpec] = TOOL_REGISTRY,
    ):
        self._ctx = ctx
        self._registry = registry
        self.calls_used = 0
        self.records: list[ToolAuditRecord] = []

    def execute(self, call: types.FunctionCall) -> types.Part:
        self.calls_used += 1

        spec = self._registry.get(call.name or "")
        if spec is None:
            return self._error(call, "unknown_tool")

        try:
            # The SDK may send args=None for a tool without parameters.
            args = spec.args_model.model_validate(call.args or {})
        except ValidationError:
            return self._error(call, "invalid_arguments")

        try:
            authorize(self._ctx, spec)
        except ToolNotAuthorized:
            return self._error(call, "unauthorized", args)

        try:
            result = spec.handler(self._ctx, args)
        except Exception as exc:
            logger.error(
                "Tool failed: tool=%s error=%s request_id=%s",
                spec.name,
                type(exc).__name__,
                self._ctx.request_id,
            )
            return self._error(call, "failed", args)

        self._record(call, "ok", args)
        return _response_part(call, {"output": result})

    def reject_over_budget(self, call: types.FunctionCall) -> types.Part:
        return self._error(call, "budget_exceeded")

    def _error(
        self,
        call: types.FunctionCall,
        status: str,
        args: ToolArgs | None = None,
    ) -> types.Part:
        self._record(call, status, args)
        return _response_part(
            call,
            {"error": {"code": status, "message": TOOL_ERROR_MESSAGES[status]}},
        )

    def _record(
        self,
        call: types.FunctionCall,
        status: str,
        args: ToolArgs | None,
    ) -> None:
        known = (call.name or "") in self._registry
        self.records.append(
            ToolAuditRecord(
                request_id=self._ctx.request_id,
                tool=(call.name or "")[:64],
                arguments=args.model_dump() if args is not None else None,
                status=status,
            )
        )
        # The model chose the tool name, so unknown names are not logged.
        logger.info(
            "Tool call: tool=%s status=%s request_id=%s",
            call.name if known else "<unknown>",
            status,
            self._ctx.request_id,
        )


def _response_part(call: types.FunctionCall, response: dict) -> types.Part:
    return types.Part(
        function_response=types.FunctionResponse(
            id=call.id,
            name=call.name,
            response=response,
        )
    )
