import ipaddress
import logging
import os
import socket

import pytest

from tests.fakes import TEST_API_KEY, FakeClient

# This must run before app.main is imported. app.main refuses to start
# without a key, and load_dotenv() never overrides a variable that is
# already set, so the real key in .env is never loaded into the tests.
os.environ["GEMINI_API_KEY"] = TEST_API_KEY

from fastapi.testclient import TestClient  # noqa: E402

from app import conversation, llm, main  # noqa: E402


class NetworkBlockedError(RuntimeError):
    pass


_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _ensure_loopback(address):
    # Non-tuple addresses are local (AF_UNIX) sockets.
    if not isinstance(address, tuple):
        return

    host = address[0]

    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass

    raise NetworkBlockedError(f"Test attempted a network connection to {host!r}")


def _guarded_connect(self, address):
    _ensure_loopback(address)
    return _real_connect(self, address)


def _guarded_connect_ex(self, address):
    _ensure_loopback(address)
    return _real_connect_ex(self, address)


@pytest.fixture(autouse=True, scope="session")
def block_network():
    """Fail any test that opens a non-loopback connection.

    Loopback stays allowed because asyncio on Windows uses a local
    socketpair internally.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(socket.socket, "connect", _guarded_connect)
        mp.setattr(socket.socket, "connect_ex", _guarded_connect_ex)
        yield


@pytest.fixture(autouse=True)
def fake_gemini(monkeypatch):
    """Replace the real Gemini client for every test."""
    fake_client = FakeClient()
    monkeypatch.setattr(llm, "client", fake_client)
    return fake_client.models


@pytest.fixture(autouse=True)
def clean_sessions():
    conversation.store.clear()
    yield
    conversation.store.clear()


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def client_no_raise():
    """A client that returns 500 responses instead of re-raising app bugs."""
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture
def all_logs(caplog):
    caplog.set_level(logging.DEBUG)
    return caplog
