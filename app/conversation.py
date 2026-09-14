"""In-memory conversation state for /chat.

History holds customer and assistant text only. Tool calls are kept in a
separate audit list, and raw tool results are never stored.
"""

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal

from app.tools import ToolAuditRecord

# How many stored MESSAGES (not user/assistant turns) are sent to the model.
HISTORY_WINDOW_MESSAGES = 20


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


class ConversationBusyError(Exception):
    """Another request for the same session is still being processed."""


class ConversationStore:
    def __init__(self, history_window: int = HISTORY_WINDOW_MESSAGES):
        if history_window <= 0:
            raise ValueError("history_window must be positive")

        self._history_window = history_window
        # threading.Lock, not asyncio.Lock: /chat is a sync endpoint that
        # FastAPI runs on threadpool threads. The lock guards the dicts and
        # the busy set, and is never held while the model is called.
        self._lock = threading.Lock()
        self._messages: dict[str, list[Message]] = {}
        self._tool_calls: dict[str, list[ToolAuditRecord]] = {}
        self._busy: set[str] = set()

    @contextmanager
    def turn(self, session_id: str) -> Iterator[None]:
        """Use a session exclusively for one request, or fail immediately."""
        with self._lock:
            if session_id in self._busy:
                raise ConversationBusyError
            self._busy.add(session_id)

        try:
            yield
        finally:
            with self._lock:
                self._busy.discard(session_id)

    def recent_messages(self, session_id: str) -> list[Message]:
        with self._lock:
            recent = self._messages.get(session_id, [])[-self._history_window:]

        # History sent to the model must start with a customer message.
        while recent and recent[0].role != "user":
            recent = recent[1:]

        return recent

    def commit_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        with self._lock:
            self._messages.setdefault(session_id, []).extend(
                [Message("user", user_text), Message("assistant", assistant_text)]
            )

    def record_tool_calls(self, session_id: str, records: list[ToolAuditRecord]) -> None:
        if not records:
            return

        with self._lock:
            self._tool_calls.setdefault(session_id, []).extend(records)

    def messages(self, session_id: str) -> list[Message]:
        with self._lock:
            return list(self._messages.get(session_id, []))

    def tool_calls(self, session_id: str) -> list[ToolAuditRecord]:
        with self._lock:
            return list(self._tool_calls.get(session_id, []))

    def session_ids(self) -> set[str]:
        """Sessions that have conversation history."""
        with self._lock:
            return set(self._messages)

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()
            self._tool_calls.clear()
            self._busy.clear()


store = ConversationStore()
