"""Application-owned agent loop for one /chat turn.

The model only requests tool calls. This loop enforces the tool budget and
hands each call to the ToolExecutor, which validates, authorizes and runs it.
The loop itself never sees the customer identity.
"""

from dataclasses import dataclass

from google.genai import types

from app import llm
from app.conversation import Message
from app.tools import ToolExecutor, declarations


@dataclass(frozen=True)
class AgentLimits:
    max_tool_rounds: int = 3
    max_tool_calls_per_round: int = 3
    max_tool_calls_total: int = 5


LIMITS = AgentLimits()


def build_contents(history: list[Message], message: str) -> list[types.Content]:
    contents = [
        types.Content(
            role="user" if item.role == "user" else "model",
            parts=[types.Part(text=item.content)],
        )
        for item in history
    ]
    contents.append(types.Content(role="user", parts=[types.Part(text=message)]))

    return contents


def run_turn(
    history: list[Message],
    message: str,
    executor: ToolExecutor,
    limits: AgentLimits = LIMITS,
) -> str:
    contents = build_contents(history, message)
    tool_declarations = declarations()
    tool_rounds = 0

    # Ends after at most max_tool_rounds + 1 model requests: every loop
    # either returns, raises, or uses up one tool round.
    while True:
        tools_enabled = (
            tool_rounds < limits.max_tool_rounds
            and executor.calls_used < limits.max_tool_calls_total
        )

        response = llm.generate(contents, tool_declarations, tools_enabled=tools_enabled)
        calls = llm.function_calls(response)

        if not calls:
            return llm.reply_text(response)

        if not tools_enabled:
            # Tools were disabled (mode NONE) but the model still asked.
            raise llm.EmptyModelReply

        tool_rounds += 1

        # Replay the model's own Content unchanged: its function-call parts
        # may carry a thought_signature that Gemini requires back.
        contents.append(response.candidates[0].content)
        contents.append(
            types.Content(
                role="user",
                parts=[
                    _handle_call(call, index, executor, limits)
                    for index, call in enumerate(calls)
                ],
            )
        )


def _handle_call(
    call: types.FunctionCall,
    index: int,
    executor: ToolExecutor,
    limits: AgentLimits,
) -> types.Part:
    # Every requested call gets a response part, even when it is refused.
    if (
        index >= limits.max_tool_calls_per_round
        or executor.calls_used >= limits.max_tool_calls_total
    ):
        return executor.reject_over_budget(call)

    return executor.execute(call)
