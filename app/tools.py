"""Banking tools offered to the model, and the backend executor that runs them.

The model can only *request* a tool call. ToolExecutor decides what happens:
it checks the tool name, validates the arguments, authorizes, binds a banking
view to the backend's customer, executes, and returns a controlled result.
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable

from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.banking import (
    TRANSACTION_ID_PATTERN,
    BankingError,
    BankingService,
    BankingUnavailable,
    CustomerBanking,
    CustomerNotFound,
    InvalidTransactionReference,
    Transaction,
    TransactionNotFound,
)
from app.context import RequestContext

logger = logging.getLogger(__name__)


class ToolArgs(BaseModel):
    # Unknown fields (e.g. an injected customer_id) and type coercion are
    # rejected, not ignored.
    model_config = ConfigDict(extra="forbid", strict=True)


class NoArgs(ToolArgs):
    pass


class TransactionDetailsArgs(ToolArgs):
    transaction_id: str = Field(pattern=TRANSACTION_ID_PATTERN)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[ToolArgs]
    parameters: types.Schema | None
    handler: Callable[[CustomerBanking, Any], dict]

    def declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


# Handlers receive a view already bound to the backend's customer. They never
# see the RequestContext or any customer ID, and they choose exactly which
# fields reach the model. Money is sent as an exact decimal string.
def _money(amount: Decimal) -> str:
    return format(amount, "f")


def _transaction_json(transaction: Transaction) -> dict:
    return {
        "transaction_id": transaction.transaction_id,
        "date": transaction.date.isoformat(),
        "type": transaction.type.value,
        "amount": _money(transaction.amount),
        "currency": transaction.currency,
        "description": transaction.description,
    }


def _account_balance(banking: CustomerBanking, args: NoArgs) -> dict:
    balance = banking.get_balance()

    return {"balance": _money(balance.amount), "currency": balance.currency}


def _recent_transactions(banking: CustomerBanking, args: NoArgs) -> dict:
    return {
        "transactions": [
            _transaction_json(transaction)
            for transaction in banking.get_recent_transactions()
        ]
    }


def _transaction_details(banking: CustomerBanking, args: TransactionDetailsArgs) -> dict:
    return _transaction_json(banking.get_transaction(args.transaction_id))


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
            description="List the customer's most recent transactions, newest first.",
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
    # Every current tool reads the requesting customer's own data through a
    # view bound to ctx.customer_id, so ownership is enforced by construction.
    # Real authorization rules arrive with STEP 6.
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
    "not_found": "No matching transaction was found for this customer.",
    "account_unavailable": "The customer's account information is unavailable.",
    "banking_unavailable": "The banking service is temporarily unavailable.",
    "failed": "The tool is temporarily unavailable.",
    "budget_exceeded": "The tool call limit for this request was reached.",
}


def _banking_error_status(exc: BankingError) -> str:
    if isinstance(exc, TransactionNotFound):
        return "not_found"
    if isinstance(exc, InvalidTransactionReference):
        return "invalid_arguments"
    if isinstance(exc, CustomerNotFound):
        return "account_unavailable"
    if isinstance(exc, BankingUnavailable):
        return "banking_unavailable"
    return "failed"


class ToolExecutor:
    """Runs model-requested tool calls for one request."""

    def __init__(
        self,
        ctx: RequestContext,
        banking_service: BankingService,
        registry: dict[str, ToolSpec] = TOOL_REGISTRY,
    ):
        self._ctx = ctx
        self._banking_service = banking_service
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
            # Banking access is bound to the backend's customer, and only
            # after the call has been validated and authorized.
            banking = self._banking_service.for_customer(self._ctx.customer_id)
            result = spec.handler(banking, args)
        except BankingError as exc:
            status = _banking_error_status(exc)
            if status in ("account_unavailable", "banking_unavailable", "failed"):
                logger.warning(
                    "Banking error: tool=%s error=%s request_id=%s",
                    spec.name,
                    type(exc).__name__,
                    self._ctx.request_id,
                )
            return self._error(call, status, args)
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
