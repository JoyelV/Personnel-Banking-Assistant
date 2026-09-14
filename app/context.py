from dataclasses import dataclass


@dataclass(frozen=True)
class RequestContext:
    """Facts about one /chat request, established by the backend.

    customer_id comes from the auth layer only. It is handed to the tool
    executor and is never sent to, or chosen by, the model.
    """

    customer_id: str
    session_id: str
    request_id: str
