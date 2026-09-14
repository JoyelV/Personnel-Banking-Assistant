"""Test doubles and sentinel values shared by the test suite.

Nothing in this module may reach the real Gemini API.
"""

from google.genai import errors, types

# Fake credentials installed by conftest.py before app.main is imported.
TEST_API_KEY = "test-dummy-key-not-real-7c1e"

# Distinctive values that must never appear in responses or logs.
SESSION_ID = "session-SENTINEL-7f3a"
CUSTOMER_MESSAGE = "customer-message-SENTINEL-91bc"
PROVIDER_SECRET = "PROVIDER-SECRET-DETAIL-4d2e"

SENSITIVE_VALUES = (TEST_API_KEY, SESSION_ID, CUSTOMER_MESSAGE, PROVIDER_SECRET)

# Internal detail a failing banking backend might put in an exception.
INTERNAL_SECRET = "core-banking db=10.0.0.5 user=svc_bank SENTINEL-5a9f"


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


def _model_response(parts: list[types.Part]) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(role="model", parts=parts))
        ]
    )


def text_response(text: str) -> types.GenerateContentResponse:
    return _model_response([types.Part(text=text)])


def function_call_response(
    *calls: tuple[str, dict | None],
    thought_signature: bytes | None = None,
) -> types.GenerateContentResponse:
    """A model turn requesting (name, args) tool calls.

    Like Gemini, only the first function-call part carries the signature.
    """
    return _model_response(
        [
            types.Part(
                function_call=types.FunctionCall(
                    id=f"call-{index}", name=name, args=args
                ),
                thought_signature=thought_signature if index == 0 else None,
            )
            for index, (name, args) in enumerate(calls)
        ]
    )


def empty_response() -> types.GenerateContentResponse:
    return types.GenerateContentResponse(candidates=[])


def everything_sent(call: dict) -> str:
    """Everything one recorded model request sent: contents and config."""
    sent = [content.model_dump_json() for content in call["contents"]]
    sent.append(call["config"].model_dump_json())
    return "\n".join(sent)


def function_responses(call: dict) -> list[types.FunctionResponse]:
    """Tool results the backend sent at the end of a recorded model request."""
    return [
        part.function_response
        for part in call["contents"][-1].parts
        if part.function_response
    ]


class FakeModels:
    """Stands in for client.models. Fails loudly unless configured."""

    def __init__(self):
        self.calls = []
        self._behaviour = None

    def returns(self, text):
        self._behaviour = lambda **_: text_response(text)

    def raises(self, exc: BaseException):
        def behaviour(**_):
            raise exc

        self._behaviour = behaviour

    def runs(self, behaviour):
        """behaviour(model=, contents=, config=) -> GenerateContentResponse."""
        self._behaviour = behaviour

    def script(self, *steps):
        """Answer successive requests in order; exception steps are raised."""
        remaining = list(steps)

        def behaviour(**_):
            if not remaining:
                raise AssertionError("Fake Gemini script exhausted")

            step = remaining.pop(0)

            if isinstance(step, BaseException):
                raise step

            return step

        self._behaviour = behaviour

    def modes(self):
        """Function-calling mode of each recorded request."""
        return [
            call["config"].tool_config.function_calling_config.mode
            for call in self.calls
        ]

    def generate_content(self, *, model, contents, config=None):
        # Copy: the agent loop appends to the same list in later rounds.
        self.calls.append(
            {"model": model, "contents": list(contents), "config": config}
        )

        if self._behaviour is None:
            raise AssertionError(
                "Unexpected Gemini call: configure the fake_gemini fixture first"
            )

        return self._behaviour(model=model, contents=contents, config=config)


class FakeClient:
    def __init__(self):
        self.models = FakeModels()
