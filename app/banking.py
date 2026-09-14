"""Banking domain contract: models, errors and the service interface.

Read-only by design. Tools reach banking data only through a
CustomerBanking view that BankingService.for_customer() binds to one
customer, so no method a tool can call accepts a customer ID.
"""

import datetime
import re
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Protocol

# The most recent transactions a customer view ever returns.
RECENT_TRANSACTIONS_LIMIT = 10

# Matches the mock data's ID format; revisit with the real banking API.
TRANSACTION_ID_PATTERN = r"^TXN[0-9]{4,12}$"
_TRANSACTION_ID = re.compile(TRANSACTION_ID_PATTERN)


class TransactionType(str, Enum):
    DEBIT = "DEBIT"
    CREDIT = "CREDIT"


@dataclass(frozen=True)
class Balance:
    amount: Decimal
    currency: str


@dataclass(frozen=True)
class Transaction:
    transaction_id: str
    date: datetime.date
    type: TransactionType
    # Always positive; the direction is given by type.
    amount: Decimal
    currency: str
    description: str


class BankingError(Exception):
    """Base for banking domain errors. They never carry internal details."""


class CustomerNotFound(BankingError):
    """The backend identity has no banking record."""


class TransactionNotFound(BankingError):
    """No such transaction for this customer.

    Also raised when the transaction belongs to someone else, so callers
    cannot tell whether another customer's transaction exists.
    """


class InvalidTransactionReference(BankingError):
    """The transaction ID is not in a valid format."""


class BankingUnavailable(BankingError):
    """The banking backend could not be reached or did not respond."""


def validate_transaction_id(transaction_id: object) -> str:
    if not isinstance(transaction_id, str) or not _TRANSACTION_ID.fullmatch(transaction_id):
        raise InvalidTransactionReference

    return transaction_id


class CustomerBanking(Protocol):
    """Read-only banking operations for ONE customer."""

    def get_balance(self) -> Balance: ...

    def get_recent_transactions(self) -> list[Transaction]:
        """Newest first, at most RECENT_TRANSACTIONS_LIMIT."""
        ...

    def get_transaction(self, transaction_id: str) -> Transaction:
        """Raises InvalidTransactionReference or TransactionNotFound."""
        ...


class BankingService(Protocol):
    def for_customer(self, customer_id: str) -> CustomerBanking:
        """Bind to one customer. Raises CustomerNotFound.

        customer_id must come from the backend's authenticated context,
        never from the model.
        """
        ...
