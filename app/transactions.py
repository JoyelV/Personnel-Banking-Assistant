def get_recent_transactions(customer_id: str):
    transactions = {
        "CUST001": [
            {
                "transaction_id": "TXN1001",
                "type": "DEBIT",
                "amount": 2500.00,
                "currency": "INR",
                "description": "Amazon",
                "date": "2026-09-05",
            },
            {
                "transaction_id": "TXN1002",
                "type": "CREDIT",
                "amount": 50000.00,
                "currency": "INR",
                "description": "Salary",
                "date": "2026-09-01",
            },
            {
                "transaction_id": "TXN1003",
                "type": "DEBIT",
                "amount": 850.00,
                "currency": "INR",
                "description": "Electricity Bill",
                "date": "2026-08-30",
            },
        ],
        "CUST002": [
            {
                "transaction_id": "TXN2001",
                "type": "DEBIT",
                "amount": 1200.00,
                "currency": "INR",
                "description": "Swiggy",
                "date": "2026-09-04",
            }
        ],
    }

    return transactions.get(customer_id, [])

def get_transaction_details(customer_id: str, transaction_id: str):
    transactions = {
        "CUST001": [
            {
                "transaction_id": "TXN1001",
                "type": "DEBIT",
                "amount": 2500.00,
                "currency": "INR",
                "description": "Amazon",
                "date": "2026-09-05",
            },
            {
                "transaction_id": "TXN1002",
                "type": "CREDIT",
                "amount": 50000.00,
                "currency": "INR",
                "description": "Salary",
                "date": "2026-09-01",
            },
            {
                "transaction_id": "TXN1003",
                "type": "DEBIT",
                "amount": 850.00,
                "currency": "INR",
                "description": "Electricity Bill",
                "date": "2026-08-30",
            },
        ],
        "CUST002": [
            {
                "transaction_id": "TXN2001",
                "type": "DEBIT",
                "amount": 1200.00,
                "currency": "INR",
                "description": "Swiggy",
                "date": "2026-09-04",
            }
        ],
    }

    customer_transactions = transactions.get(customer_id, [])

    for transaction in customer_transactions:
        if transaction["transaction_id"] == transaction_id:
            return {
                "found": True,
                **transaction,
            }

    return {
        "found": False,
        "transaction_id": transaction_id,
    }

    