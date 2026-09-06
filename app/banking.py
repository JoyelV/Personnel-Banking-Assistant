def get_account_balance(customer_id: str):
    accounts = {
        "CUST001": 25430.50,
        "CUST002": 7820.00,
        "CUST003": 150000.75,
    }

    balance = accounts.get(customer_id)

    if balance is None:
        return {
            "found": False,
            "customer_id": customer_id,
            "balance": None,
            "currency": "INR",
        }

    return {
        "found": True,
        "customer_id": customer_id,
        "balance": balance,
        "currency": "INR",
    }