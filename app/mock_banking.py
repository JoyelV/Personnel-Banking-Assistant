"""In-memory BankingService for development and tests.

One immutable dataset. A customer view is given only that customer's
record, so it has no way to reach anyone else's data.
"""

import datetime
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Mapping

from app.banking import (
    RECENT_TRANSACTIONS_LIMIT,
    Balance,
    CustomerNotFound,
    Transaction,
    TransactionNotFound,
    TransactionType,
    validate_transaction_id,
)


@dataclass(frozen=True)
class MockCustomer:
    balance: Balance
    transactions: tuple[Transaction, ...] = ()


def _transaction(
    transaction_id: str,
    day: str,
    type_: TransactionType,
    amount: str,
    description: str,
) -> Transaction:
    return Transaction(
        transaction_id=transaction_id,
        date=datetime.date.fromisoformat(day),
        type=type_,
        amount=Decimal(amount),
        currency="INR",
        description=description,
    )


SEED_CUSTOMERS: Mapping[str, MockCustomer] = MappingProxyType({
    "CUST001": MockCustomer(
        balance=Balance(Decimal("25430.50"), "INR"),
        transactions=(
            _transaction("TXN1001", "2026-09-05", TransactionType.DEBIT, "2500.00", "Amazon"),
            _transaction("TXN1002", "2026-09-01", TransactionType.CREDIT, "50000.00", "Salary"),
            _transaction("TXN1003", "2026-08-30", TransactionType.DEBIT, "850.00", "Electricity Bill"),
        ),
    ),
    "CUST002": MockCustomer(
        balance=Balance(Decimal("7820.00"), "INR"),
        transactions=(
            _transaction("TXN2001", "2026-09-04", TransactionType.DEBIT, "1200.00", "Swiggy"),
        ),
    ),
    "CUST003": MockCustomer(
        balance=Balance(Decimal("150000.75"), "INR"),
    ),
})


class MockBankingService:
    def __init__(self, customers: Mapping[str, MockCustomer] = SEED_CUSTOMERS):
        self._customers = dict(customers)

    def for_customer(self, customer_id: str) -> "MockCustomerBanking":
        customer = self._customers.get(customer_id)

        if customer is None:
            raise CustomerNotFound

        return MockCustomerBanking(customer)


class MockCustomerBanking:
    """CustomerBanking view holding a single customer's record."""

    def __init__(self, customer: MockCustomer):
        self._customer = customer

    def get_balance(self) -> Balance:
        return self._customer.balance

    def get_recent_transactions(self) -> list[Transaction]:
        newest_first = sorted(
            self._customer.transactions,
            key=lambda transaction: transaction.date,
            reverse=True,
        )
        return newest_first[:RECENT_TRANSACTIONS_LIMIT]

    def get_transaction(self, transaction_id: str) -> Transaction:
        validate_transaction_id(transaction_id)

        for transaction in self._customer.transactions:
            if transaction.transaction_id == transaction_id:
                return transaction

        raise TransactionNotFound
