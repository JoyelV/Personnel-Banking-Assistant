"""Test doubles and sentinel values shared by the test suite.

Nothing in this module may reach the real Gemini API.
"""

from types import SimpleNamespace

from google.genai import errors

# Fake credentials installed by conftest.py before app.main is imported.
TEST_API_KEY = "test-dummy-key-not-real-7c1e"

# Distinctive values that must never appear in responses or logs.
SESSION_ID = "session-SENTINEL-7f3a"
CUSTOMER_MESSAGE = "customer-message-SENTINEL-91bc"
PROVIDER_SECRET = "PROVIDER-SECRET-DETAIL-4d2e"

SENSITIVE_VALUES = (TEST_API_KEY, SESSION_ID, CUSTOMER_MESSAGE, PROVIDER_SECRET)


def provider_error(error_class: type[errors.APIError], code: int) -> errors.APIError:
    """Build a google-genai error shaped like a real provider response."""
    return error_class(
        code,
        {
            "error": {
                "code": code,
                "message": PROVIDER_SECRET,
                "status": "PROVIDER_STATUS",
            }
        },
    )


class FakeModels:
    """Stands in for client.models. Fails loudly unless configured."""

    def __init__(self):
        self.calls = []
        self._behaviour = None

    def returns(self, text):
        self._behaviour = lambda **_: SimpleNamespace(text=text)

    def raises(self, exc: BaseException):
        def behaviour(**_):
            raise exc

        self._behaviour = behaviour

    def runs(self, behaviour):
        """behaviour(model=, contents=, config=) -> object with a .text."""
        self._behaviour = behaviour

    def generate_content(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})

        if self._behaviour is None:
            raise AssertionError(
                "Unexpected Gemini call: configure the fake_gemini fixture first"
            )

        return self._behaviour(model=model, contents=contents, config=config)


class FakeClient:
    def __init__(self):
        self.models = FakeModels()
